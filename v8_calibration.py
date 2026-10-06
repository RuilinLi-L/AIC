"""Nested out-of-fold TTA selection with a bias fitted only on each training fold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from calibrate import class_bias_from_logits
from evaluate_tta_v6 import _grid, _search_mix_macro, macro_metrics, strict_stratified_folds
from evaluate_tta_v7 import greedy_view_search
from v7_views import VIEW_NAMES, normalize_view_weights


@dataclass
class CalibrationResult:
    summary: dict[str, Any]
    oof_logits: np.ndarray
    full_logits: np.ndarray
    class_bias: np.ndarray


def _mix(views: dict[str, np.ndarray], weights: dict[str, float], indices: np.ndarray) -> np.ndarray:
    result = None
    for name, weight in weights.items():
        value = float(weight) * np.asarray(views[name][indices], dtype=np.float32)
        result = value if result is None else result + value
    if result is None:
        raise ValueError("TTA mix has no views")
    return result


def _fit_family(
    views: dict[str, np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
    family: str,
    device: torch.device,
    folds: int,
) -> dict[str, Any]:
    train_labels = labels[indices]
    inner_ids = strict_stratified_folds(train_labels, folds=folds, seed=2026)
    if family == "two_view":
        result = _search_mix_macro(
            views["center"][indices], views["hflip"][indices], train_labels, inner_ids,
            _grid(1.0, 0.05), _grid(2.0, 0.05), device,
        )
        weights = normalize_view_weights({
            "center": result["original_weight"], "hflip": result["hflip_weight"]
        })
    elif family == "four_view":
        result = greedy_view_search(
            {name: views[name][indices] for name in VIEW_NAMES}, train_labels, inner_ids, device
        )
        weights = result["view_weights"]
    else:
        raise ValueError(f"unknown calibration family: {family}")
    return {
        "view_weights": weights,
        "alpha": float(result["alpha"]),
        "inner_cv_macro_accuracy": float(result["cv_macro_accuracy"]),
        "inner_cv_macro_nll": float(result["cv_macro_nll"]),
    }


def _bias(logits: np.ndarray, alpha: float) -> np.ndarray:
    raw = class_bias_from_logits(torch.as_tensor(logits, dtype=torch.float32)).numpy()
    return (float(alpha) * raw).astype(np.float32)


def nested_calibration(
    views: dict[str, np.ndarray], labels: np.ndarray, device: torch.device
) -> CalibrationResult:
    if set(views) != set(VIEW_NAMES):
        raise ValueError("V8 calibration requires the four named views")
    labels = np.asarray(labels, dtype=np.int64)
    shapes = {tuple(value.shape) for value in views.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] != len(labels):
        raise ValueError("V8 validation views or labels have inconsistent shapes")
    if not all(np.isfinite(value).all() for value in views.values()):
        raise ValueError("V8 validation logits contain nonfinite values")
    fold_ids = strict_stratified_folds(labels, folds=5, seed=2026)
    all_indices = np.arange(len(labels), dtype=np.int64)
    families = ("two_view", "four_view")
    family_results: dict[str, dict[str, Any]] = {}
    for family in families:
        oof = np.empty_like(views["center"], dtype=np.float32)
        fold_details = []
        for outer in range(5):
            train = all_indices[fold_ids != outer]
            held_out = all_indices[fold_ids == outer]
            params = _fit_family(views, labels, train, family, device, folds=4)
            train_mixed = _mix(views, params["view_weights"], train)
            fold_bias = _bias(train_mixed, params["alpha"])
            oof[held_out] = _mix(views, params["view_weights"], held_out) + fold_bias
            fold_details.append({
                "outer_fold": outer,
                "train_rows": int(len(train)),
                "held_out_rows": int(len(held_out)),
                **params,
            })
        family_results[family] = {
            "metrics": macro_metrics(oof, labels),
            "folds": fold_details,
            "oof": oof,
        }
    two = family_results["two_view"]["metrics"]["macro_accuracy"]
    four = family_results["four_view"]["metrics"]["macro_accuracy"]
    selected_family = "four_view" if four > two + 0.0005 else "two_view"
    fitted = _fit_family(views, labels, all_indices, selected_family, device, folds=5)
    mixed = _mix(views, fitted["view_weights"], all_indices)
    bias = _bias(mixed, fitted["alpha"])
    full_logits = mixed + bias
    summary = {
        "method": "v8_nested_5x4_macro_tta_bias",
        "selected_family": selected_family,
        "selected": fitted,
        "outer_fold_ids": fold_ids.tolist(),
        "families": {
            name: {"outer_metrics": value["metrics"], "folds": value["folds"]}
            for name, value in family_results.items()
        },
        "selected_outer_metrics": family_results[selected_family]["metrics"],
        "selected_full_validation": macro_metrics(full_logits, labels),
    }
    return CalibrationResult(
        summary,
        family_results[selected_family]["oof"],
        full_logits.astype(np.float32),
        bias,
    )
