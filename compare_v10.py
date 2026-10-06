"""Compare V10 with the selected V9 run using identical strict outer-fold rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_v7 import stratified_paired_bootstrap


CONDITIONS = ("native", "resize384", "jpeg75", "combined")


def load_evaluation(directory: Path) -> tuple[dict, np.ndarray, list[str], dict[str, np.ndarray]]:
    summary = json.loads((directory / "strict_eval.json").read_text(encoding="utf-8"))
    labels = np.load(directory / "val_labels.npy", allow_pickle=False)
    row_keys = json.loads((directory / "val_row_keys.json").read_text(encoding="utf-8"))
    if labels.ndim != 1 or len(labels) != len(row_keys):
        raise ValueError(f"invalid V10 validation labels/row keys in {directory}")
    logits = {}
    for name in CONDITIONS:
        value = np.load(directory / f"val_oof_{name}.npy", allow_pickle=False)
        if value.ndim != 2 or value.shape[0] != len(labels) or not np.isfinite(value).all():
            raise ValueError(f"invalid V10 {name} OOF logits in {directory}")
        logits[name] = value
    return summary, labels, row_keys, logits


def compare(
    baseline_dir: Path, candidate_dir: Path, bootstrap_repeats: int = 2000
) -> dict:
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    baseline, baseline_labels, baseline_keys, baseline_logits = load_evaluation(baseline_dir)
    candidate, labels, row_keys, candidate_logits = load_evaluation(candidate_dir)
    if row_keys != baseline_keys or not np.array_equal(labels, baseline_labels):
        raise ValueError("V9 and V10 strict evaluations have different validation rows/labels")
    if (
        not isinstance(baseline.get("dataset_signature"), str)
        or baseline["dataset_signature"] != candidate.get("dataset_signature")
    ):
        raise ValueError("V9 and V10 strict evaluations have different dataset signatures")
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("validation row keys are not unique")
    if "outer_fold_ids" in baseline and "outer_fold_ids" in candidate:
        if baseline["outer_fold_ids"] != candidate["outer_fold_ids"]:
            raise ValueError("V9 and V10 strict evaluations use different outer folds")

    conditions = {}
    for index, name in enumerate(CONDITIONS):
        base_values = baseline_logits[name]
        candidate_values = candidate_logits[name]
        if base_values.shape != candidate_values.shape:
            raise ValueError(f"V9 and V10 {name} logits have different shapes")
        base_metrics = baseline["conditions"][name]["outer_metrics"]
        candidate_metrics = candidate["conditions"][name]["outer_metrics"]
        base_pred = base_values.argmax(axis=1)
        candidate_pred = candidate_values.argmax(axis=1)
        for metrics, predictions in ((base_metrics, base_pred), (candidate_metrics, candidate_pred)):
            actual_micro = float(np.mean(predictions == labels))
            if not np.isclose(actual_micro, metrics["micro_accuracy"], atol=1e-6):
                raise ValueError(f"{name} summary and OOF logits disagree on micro accuracy")
            actual_macro = float(np.mean([
                np.mean(predictions[labels == label] == label) for label in np.unique(labels)
            ]))
            if not np.isclose(actual_macro, metrics["macro_accuracy"], atol=1e-6):
                raise ValueError(f"{name} summary and OOF logits disagree on macro accuracy")
        paired = stratified_paired_bootstrap(
            {"labels": labels, "predictions": base_pred,
             "macro_accuracy": base_metrics["macro_accuracy"]},
            {"labels": labels, "predictions": candidate_pred,
             "macro_accuracy": candidate_metrics["macro_accuracy"]},
            bootstrap_repeats,
            seed=2026 + index,
        )
        conditions[name] = {
            "baseline_outer_metrics": base_metrics,
            "candidate_outer_metrics": candidate_metrics,
            "macro_delta": float(candidate_metrics["macro_accuracy"] - base_metrics["macro_accuracy"]),
            "micro_delta": float(candidate_metrics["micro_accuracy"] - base_metrics["micro_accuracy"]),
            "tail_delta": float(candidate_metrics["tail_accuracy"] - base_metrics["tail_accuracy"]),
            "paired_class_bootstrap": paired,
        }

    native = conditions["native"]
    combined = conditions["combined"]
    recommended = bool(
        native["macro_delta"] >= -0.002
        and native["micro_delta"] >= 0.0
        and combined["macro_delta"] >= 0.010
        and combined["tail_delta"] >= -0.005
        and combined["paired_class_bootstrap"]["probability_delta_gt_zero"] > 0.95
    )
    return {
        "baseline": str(baseline_dir.resolve()),
        "candidate": str(candidate_dir.resolve()),
        "validation_rows": int(len(labels)),
        "conditions": conditions,
        "recommended_for_submission": recommended,
        "status": "recommended_for_online_test" if recommended else "experimental",
        "decision_rule": (
            "native macro>=-0.002 and native micro>=0; "
            "combined macro>=+0.010 and combined tail>=-0.005; "
            "P(combined macro delta>0)>0.95"
        ),
        "online_improvement_confirmed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
