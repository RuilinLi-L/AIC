"""Joint epoch/TTA/calibration selection, with untouched outer-fold scoring."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from evaluate_tta_v6 import _search_mix_macro, strict_stratified_folds
from v10_evaluation import (
    ALPHAS, CONDITIONS, FAMILY_WEIGHTS, StrictEvaluation, _metrics, fitted_bias,
    mix_views,
)


METHOD = "v20_strict_outer5_inner4_joint_epoch_tta_bias"
PROTOCOL = {
    "method": METHOD,
    "version": 1,
    "outer_folds": 5,
    "inner_folds": 4,
    "final_inner_folds": 5,
    "seed": 2026,
    "epochs": list(range(1, 25)),
    "families": FAMILY_WEIGHTS,
    "alphas": [float(value) for value in ALPHAS],
    "metric_aggregation": "pooled_inner_oof_per_class_then_class_mean",
    "bias_fit": "inner_training_logits_only_then_refit_on_outer_training_logits",
    "tie_break": ["macro_accuracy_desc", "macro_nll_asc", "alpha_asc", "two_view_first", "epoch_asc"],
    "stress_selection": False,
}


def selection_key(candidate):
    return (
        candidate["inner_cv_macro_accuracy"], -candidate["inner_cv_macro_nll"],
        -candidate["alpha"], candidate["family"] == "two_view", -candidate["epoch"],
    )


def fit_joint_epoch_tta(
    native_mixed: Mapping[int, Mapping[str, np.ndarray]], labels: np.ndarray,
    indices: np.ndarray, *, folds: int, device: torch.device,
) -> dict:
    """Select all three parameters from inner OOF scores on exactly ``indices``.

    ``_search_mix_macro`` fits the marginal bias on each inner training partition
    and accumulates held-out correct/NLL counts by class across all inner folds.
    It does not average fold macro scores, which would overcount sparse classes.
    Keeping its best alpha for each epoch/family is equivalent to ranking the
    complete Cartesian grid under the same lexicographic ordering.
    """
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("joint selection requires unique nonempty fit indices")
    if indices.min() < 0 or indices.max() >= len(labels):
        raise ValueError("joint selection indices outside validation rows")
    local_labels = labels[indices]
    fold_ids = strict_stratified_folds(local_labels, folds=folds, seed=2026)
    candidates = []
    for epoch in sorted(native_mixed):
        families = native_mixed[epoch]
        if set(families) != set(FAMILY_WEIGHTS):
            raise ValueError("joint selection requires both fixed TTA families")
        for family in FAMILY_WEIGHTS:
            mixed = np.asarray(families[family], dtype=np.float32)
            if mixed.ndim != 2 or len(mixed) != len(labels) or not np.isfinite(mixed).all():
                raise ValueError("invalid native logits for joint selection")
            local = mixed[indices]
            search = _search_mix_macro(
                local, local, local_labels, fold_ids,
                np.asarray([1.0], dtype=np.float32), ALPHAS, device,
            )
            candidates.append({
                "epoch": int(epoch), "family": family, "alpha": float(search["alpha"]),
                "inner_cv_macro_accuracy": float(search["cv_macro_accuracy"]),
                "inner_cv_macro_nll": float(search["cv_macro_nll"]),
            })
    if not candidates:
        raise ValueError("joint selection requires epoch logits")
    winner = max(candidates, key=selection_key)
    return {**winner, "view_weights": dict(FAMILY_WEIGHTS[winner["family"]]),
            "candidates": candidates, "inner_fold_ids": fold_ids.tolist()}


def strict_nested_evaluation(get_views, epoch_numbers, labels, class_counts, *, device):
    """Evaluate a joint selection procedure on five outer folds, then freeze it."""
    if sorted(epoch_numbers) != list(range(1, 25)) or len(epoch_numbers) != 24:
        raise ValueError("V20 joint evaluation requires all 24 consecutive epoch checkpoints")
    labels = np.asarray(labels, dtype=np.int64)
    class_counts = np.asarray(class_counts)
    if (labels.ndim != 1 or len(labels) < 5 or class_counts.ndim != 1
            or not len(class_counts) or labels.min() < 0 or labels.max() >= len(class_counts)):
        raise ValueError("invalid joint validation labels or class counts")
    indices = np.arange(len(labels), dtype=np.int64)
    outer_ids = strict_stratified_folds(labels, folds=5, seed=2026)
    shape = (len(labels), len(class_counts))
    native_mixed = {}
    for epoch in sorted(epoch_numbers):
        # Always request native four views for every epoch. The cache retains
        # at most one raw epoch; only two mixed arrays per epoch stay on CPU.
        views = get_views(epoch, "native", True)
        native_mixed[epoch] = {}
        for family in FAMILY_WEIGHTS:
            mixed = mix_views(views, family)
            if mixed.shape != shape:
                raise ValueError(f"epoch {epoch} native logits have wrong shape")
            native_mixed[epoch][family] = mixed
    oof = {condition: np.empty(shape, dtype=np.float32) for condition in CONDITIONS}
    outer_folds = []
    for outer in range(5):
        train, held_out = indices[outer_ids != outer], indices[outer_ids == outer]
        fitted = fit_joint_epoch_tta(native_mixed, labels, train, folds=4, device=device)
        epoch, family = fitted["epoch"], fitted["family"]
        native = native_mixed[epoch][family]
        bias = fitted_bias(native[train], fitted["alpha"], device)
        for condition in CONDITIONS:
            mixed = native if condition == "native" else mix_views(get_views(epoch, condition, True), family)
            if mixed.shape != shape:
                raise ValueError(f"epoch {epoch} {condition} logits have wrong shape")
            oof[condition][held_out] = mixed[held_out] + bias
        outer_folds.append({
            "outer_fold": outer, "train_rows": len(train), "held_out_rows": len(held_out),
            "selected_epoch": epoch, "selected_family": family, "alpha": fitted["alpha"],
            "view_weights": fitted["view_weights"],
            "inner_cv_macro_accuracy": fitted["inner_cv_macro_accuracy"],
            "inner_cv_macro_nll": fitted["inner_cv_macro_nll"],
            "inner_candidates": fitted["candidates"], "inner_fold_ids": fitted["inner_fold_ids"],
        })
    fitted = fit_joint_epoch_tta(native_mixed, labels, indices, folds=5, device=device)
    epoch, family = fitted["epoch"], fitted["family"]
    full_mixed = native_mixed[epoch][family]
    class_bias = fitted_bias(full_mixed, fitted["alpha"], device)
    full_logits = full_mixed + class_bias
    conditions = {}
    for condition in CONDITIONS:
        mixed = full_mixed if condition == "native" else mix_views(get_views(epoch, condition, True), family)
        conditions[condition] = {
            "outer_metrics": _metrics(oof[condition], labels, class_counts),
            "selected_full_validation": _metrics(mixed + class_bias, labels, class_counts),
        }
    selected = {key: fitted[key] for key in ("epoch", "family", "alpha", "view_weights",
                                            "inner_cv_macro_accuracy", "inner_cv_macro_nll")}
    selected.update(inner_candidates=fitted["candidates"], inner_fold_ids=fitted["inner_fold_ids"])
    summary = {
        "method": METHOD, "evaluation_protocol": PROTOCOL,
        "class_training_support": class_counts.tolist(),
        "outer_fold_ids": outer_ids.tolist(), "outer_folds": outer_folds,
        "selected": selected, "conditions": conditions,
    }
    return StrictEvaluation(summary, oof, full_logits.astype(np.float32), class_bias)
