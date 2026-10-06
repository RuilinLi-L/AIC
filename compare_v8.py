"""Compare V8 and the submitted V7 validation model under identical nested CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_v7 import stratified_paired_bootstrap
from robust_clip import resolve_device
from v6_core import head_mid_tail_accuracy
from v7_data import load_manifest_dataset
from v7_views import VIEW_NAMES
from v8_calibration import nested_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def compare_metrics(logits: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(axis=1)
    classes = np.unique(labels)
    macro = float(np.mean([(predictions[labels == label] == label).mean() for label in classes]))
    group = head_mid_tail_accuracy(torch.as_tensor(logits), torch.as_tensor(labels), counts)
    return {
        "macro_accuracy": macro,
        "accuracy": float((predictions == labels).mean()),
        **{f"{name}_accuracy": float(value) for name, value in group.items()},
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap count must be positive")
    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    baseline_keys = json.loads((baseline / "val_row_keys.json").read_text(encoding="utf-8"))
    candidate_keys = json.loads((candidate / "val_row_keys.json").read_text(encoding="utf-8"))
    labels = np.load(candidate / "val_labels.npy")
    if baseline_keys != candidate_keys or not np.array_equal(
        np.load(baseline / "val_labels.npy"), labels
    ):
        raise ValueError("V7 and V8 must have identical validation rows and labels")
    manifest = load_manifest_dataset(args.data_manifest, args.train_dir, "partial")
    row_labels = np.asarray([row[1] for row in manifest.rows], dtype=np.int64)
    clean_train = np.asarray(
        [index for index in manifest.train_indices if manifest.clean_mask[index]], dtype=np.int64
    )
    counts = np.bincount(row_labels[clean_train], minlength=len(manifest.class_names))
    baseline_oof_path = candidate / "v7_nested_oof_logits.npy"
    baseline_summary_path = candidate / "v7_nested_summary.json"
    if baseline_oof_path.is_file() and baseline_summary_path.is_file():
        baseline_oof = np.load(baseline_oof_path)
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    else:
        views = {name: np.load(baseline / f"val_{name}_logits.npy") for name in VIEW_NAMES}
        calibration = nested_calibration(views, labels, resolve_device(args.device))
        baseline_oof = calibration.oof_logits
        baseline_summary = calibration.summary
        np.save(baseline_oof_path, baseline_oof)
        baseline_summary_path.write_text(json.dumps(baseline_summary, indent=2), encoding="utf-8")
    candidate_oof = np.load(candidate / "val_oof_logits.npy")
    if baseline_oof.shape != candidate_oof.shape or baseline_oof.shape[0] != len(labels):
        raise ValueError("V7/V8 OOF logits shape differs")
    baseline_metrics = compare_metrics(baseline_oof, labels, counts)
    candidate_metrics = compare_metrics(candidate_oof, labels, counts)
    reference = {
        "labels": labels,
        "predictions": baseline_oof.argmax(axis=1),
        "macro_accuracy": baseline_metrics["macro_accuracy"],
    }
    challenger = {
        "labels": labels,
        "predictions": candidate_oof.argmax(axis=1),
        "macro_accuracy": candidate_metrics["macro_accuracy"],
    }
    bootstrap = stratified_paired_bootstrap(reference, challenger, args.bootstrap, seed=2026)
    macro_delta = candidate_metrics["macro_accuracy"] - baseline_metrics["macro_accuracy"]
    accuracy_delta = candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
    tail_delta = candidate_metrics["tail_accuracy"] - baseline_metrics["tail_accuracy"]
    recommended = bool(
        macro_delta >= 0.005 and accuracy_delta >= 0.0 and tail_delta >= -0.005
        and bootstrap["probability_delta_gt_zero"] > 0.95
    )
    payload = {
        "baseline": str(baseline.resolve()),
        "candidate": str(candidate.resolve()),
        "baseline_selected_family": baseline_summary["selected_family"],
        "baseline_outer_metrics": baseline_metrics,
        "candidate_outer_metrics": candidate_metrics,
        "macro_delta": macro_delta,
        "accuracy_delta": accuracy_delta,
        "tail_delta": tail_delta,
        "paired_class_bootstrap": bootstrap,
        "recommended_for_submission": recommended,
        "decision_rule": "macro>=+0.005, accuracy>=0, tail>=-0.005, P(macro delta>0)>0.95",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
