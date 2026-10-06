"""Strict outer-fold epoch and TTA evaluation shared by V9 and V10.

Only native validation images select epochs, TTA families, and class-bias strength.
The three fixed stress conditions are evaluated with those native-fitted parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch

from calibrate import class_bias_from_logits
from evaluate_tta_v6 import _search_mix_macro, macro_metrics, strict_stratified_folds
from v6_core import head_mid_tail_accuracy
from v7_views import VIEW_NAMES


CONDITIONS = ("native", "resize384", "jpeg75", "combined")
FAMILY_WEIGHTS: dict[str, dict[str, float]] = {
    "two_view": {"center": 0.5, "hflip": 0.5},
    "four_view": {name: 0.25 for name in VIEW_NAMES},
}
ALPHAS = np.linspace(0.0, 1.0, 21, dtype=np.float32)
ViewAccessor = Callable[[int, str, bool], Mapping[str, np.ndarray]]


@dataclass
class StrictEvaluation:
    summary: dict[str, Any]
    oof_by_condition: dict[str, np.ndarray]
    full_native_logits: np.ndarray
    class_bias: np.ndarray


def mix_views(views: Mapping[str, np.ndarray], family: str) -> np.ndarray:
    if family not in FAMILY_WEIGHTS:
        raise ValueError(f"unknown V10 TTA family: {family}")
    weights = FAMILY_WEIGHTS[family]
    missing = set(weights) - set(views)
    if missing:
        raise ValueError(f"missing TTA views: {sorted(missing)}")
    result = np.zeros_like(np.asarray(views["center"]), dtype=np.float32)
    for name, weight in weights.items():
        logits = np.asarray(views[name], dtype=np.float32)
        if logits.shape != result.shape or not np.isfinite(logits).all():
            raise ValueError(f"invalid {name} logits")
        result += float(weight) * logits
    return result


def select_epoch(
    native_two_by_epoch: Mapping[int, np.ndarray], labels: np.ndarray, indices: np.ndarray
) -> tuple[int, list[dict[str, float | int]]]:
    """Choose a weight using only uncalibrated center+hflip on given rows."""
    if not native_two_by_epoch or len(indices) == 0:
        raise ValueError("epoch selection requires checkpoints and validation rows")
    scores: list[dict[str, float | int]] = []
    best_key: tuple[float, float, int] | None = None
    best_epoch = -1
    for epoch in sorted(native_two_by_epoch):
        metrics = macro_metrics(native_two_by_epoch[epoch][indices], labels[indices])
        score = {
            "epoch": int(epoch),
            "macro_accuracy": float(metrics["macro_accuracy"]),
            "macro_nll": float(metrics["macro_nll"]),
        }
        scores.append(score)
        key = (score["macro_accuracy"], -score["macro_nll"], -int(epoch))
        if best_key is None or key > best_key:
            best_key, best_epoch = key, int(epoch)
    return best_epoch, scores


def fit_fixed_tta(
    views: Mapping[str, np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    folds: int,
    device: torch.device,
) -> dict[str, Any]:
    """Select one of two fixed families and alpha by CV within ``indices``."""
    local_labels = np.asarray(labels[indices], dtype=np.int64)
    fold_ids = strict_stratified_folds(local_labels, folds=folds, seed=2026)
    candidates = []
    for family in FAMILY_WEIGHTS:
        mixed = mix_views(views, family)[indices]
        search = _search_mix_macro(
            mixed, mixed, local_labels, fold_ids,
            np.asarray([1.0], dtype=np.float32), ALPHAS, device,
        )
        candidates.append({
            "family": family,
            "alpha": float(search["alpha"]),
            "inner_cv_macro_accuracy": float(search["cv_macro_accuracy"]),
            "inner_cv_macro_nll": float(search["cv_macro_nll"]),
        })
    # Accuracy, then NLL, then smaller alpha; fixed two-view wins an exact tie.
    winner = max(
        candidates,
        key=lambda item: (
            item["inner_cv_macro_accuracy"], -item["inner_cv_macro_nll"],
            -item["alpha"], item["family"] == "two_view",
        ),
    )
    return {**winner, "view_weights": FAMILY_WEIGHTS[winner["family"]], "candidates": candidates}


def fitted_bias(native_mixed_train: np.ndarray, alpha: float, device: torch.device) -> np.ndarray:
    values = torch.as_tensor(native_mixed_train, dtype=torch.float32, device=device)
    return (
        float(alpha) * class_bias_from_logits(values)
    ).detach().cpu().numpy().astype(np.float32)


def _metrics(logits: np.ndarray, labels: np.ndarray, class_counts: np.ndarray) -> dict[str, float]:
    result = macro_metrics(logits, labels)
    result["micro_accuracy"] = result.pop("accuracy")
    groups = head_mid_tail_accuracy(
        torch.as_tensor(logits, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
        class_counts,
    )
    result.update({f"{name}_accuracy": float(value) for name, value in groups.items()})
    return result


def strict_nested_evaluation(
    get_views: ViewAccessor,
    epoch_numbers: list[int],
    labels: np.ndarray,
    class_counts: np.ndarray,
    *,
    device: torch.device,
) -> StrictEvaluation:
    """Run 5 outer folds; no held-out labels enter any parameter choice."""
    labels = np.asarray(labels, dtype=np.int64)
    if len(epoch_numbers) not in (12, 24) or sorted(set(epoch_numbers)) != list(range(1, len(epoch_numbers) + 1)):
        raise ValueError("strict evaluation requires all 12 or 24 consecutive epoch checkpoints")
    if labels.ndim != 1 or len(labels) < 5:
        raise ValueError("invalid validation labels")
    all_indices = np.arange(len(labels), dtype=np.int64)
    outer_ids = strict_stratified_folds(labels, folds=5, seed=2026)
    native_two: dict[int, np.ndarray] = {}
    expected_shape = (len(labels), len(class_counts))
    for epoch in epoch_numbers:
        views = get_views(epoch, "native", False)
        mixed = mix_views(views, "two_view")
        if mixed.shape != expected_shape:
            raise ValueError(f"epoch {epoch} native logits have wrong shape")
        native_two[epoch] = mixed

    oof = {condition: np.empty(expected_shape, dtype=np.float32) for condition in CONDITIONS}
    outer_folds: list[dict[str, Any]] = []
    for outer in range(5):
        train = all_indices[outer_ids != outer]
        held_out = all_indices[outer_ids == outer]
        epoch, scores = select_epoch(native_two, labels, train)
        native_views = get_views(epoch, "native", True)
        fitted = fit_fixed_tta(native_views, labels, train, folds=4, device=device)
        native_mixed = mix_views(native_views, fitted["family"])
        bias = fitted_bias(native_mixed[train], fitted["alpha"], device)
        for condition in CONDITIONS:
            condition_views = native_views if condition == "native" else get_views(epoch, condition, True)
            mixed = mix_views(condition_views, fitted["family"])
            if mixed.shape != expected_shape:
                raise ValueError(f"epoch {epoch} {condition} logits have wrong shape")
            oof[condition][held_out] = mixed[held_out] + bias
        outer_folds.append({
            "outer_fold": int(outer),
            "train_rows": int(len(train)),
            "held_out_rows": int(len(held_out)),
            "selected_epoch": int(epoch),
            "epoch_scores": scores,
            "selected_family": fitted["family"],
            "alpha": fitted["alpha"],
            "view_weights": fitted["view_weights"],
            "inner_cv_macro_accuracy": fitted["inner_cv_macro_accuracy"],
            "inner_cv_macro_nll": fitted["inner_cv_macro_nll"],
            "inner_candidates": fitted["candidates"],
        })
    selected_epoch, full_epoch_scores = select_epoch(native_two, labels, all_indices)
    full_views = get_views(selected_epoch, "native", True)
    fitted = fit_fixed_tta(full_views, labels, all_indices, folds=5, device=device)
    full_mixed = mix_views(full_views, fitted["family"])
    class_bias = fitted_bias(full_mixed, fitted["alpha"], device)
    full_logits = full_mixed + class_bias
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_views = full_views if condition == "native" else get_views(selected_epoch, condition, True)
        selected_logits = mix_views(condition_views, fitted["family"]) + class_bias
        conditions[condition] = {
            "outer_metrics": _metrics(oof[condition], labels, class_counts),
            "selected_full_validation": _metrics(selected_logits, labels, class_counts),
        }
    summary = {
        "method": "v10_strict_outer5_epoch_inner4_fixed_tta_bias",
        "outer_fold_ids": outer_ids.tolist(),
        "outer_folds": outer_folds,
        "selected": {
            "epoch": int(selected_epoch),
            "family": fitted["family"],
            "alpha": fitted["alpha"],
            "view_weights": fitted["view_weights"],
            "inner_cv_macro_accuracy": fitted["inner_cv_macro_accuracy"],
            "inner_cv_macro_nll": fitted["inner_cv_macro_nll"],
            "epoch_scores": full_epoch_scores,
        },
        "conditions": conditions,
    }
    return StrictEvaluation(summary, oof, full_logits.astype(np.float32), class_bias)
