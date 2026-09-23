"""Rank V7 runs and compute class-stratified paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CV_TIE_TOLERANCE = 0.0005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_run(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    calibration = json.loads((path / "calibration.json").read_text(encoding="utf-8"))
    labels = np.load(path / "val_labels.npy")
    logits = np.load(path / "val_selected_logits.npy")
    row_keys = json.loads((path / "val_row_keys.json").read_text(encoding="utf-8"))
    if logits.shape[0] != len(labels) or len(row_keys) != len(labels):
        raise ValueError(f"inconsistent validation artifacts in {path}")
    selected = calibration["selected"]
    full = calibration["selected_full_validation"]
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "policy": calibration.get("config", {}).get("conflict_policy"),
        "sampler": calibration.get("config", {}).get("sampler"),
        "selected_epoch": calibration.get("selected_epoch"),
        "cv_macro_accuracy": float(selected["cv_macro_accuracy"]),
        "cv_macro_nll": float(selected["cv_macro_nll"]),
        "macro_accuracy": float(full["macro_accuracy"]),
        "macro_nll": float(full["macro_nll"]),
        "tail_accuracy": float(full["tail_accuracy"]),
        "mid_accuracy": float(full["mid_accuracy"]),
        "head_accuracy": float(full["head_accuracy"]),
        "labels": labels,
        "predictions": logits.argmax(axis=1),
        "row_keys": row_keys,
    }


def _better(left: dict[str, Any], right: dict[str, Any]) -> bool:
    delta = left["cv_macro_accuracy"] - right["cv_macro_accuracy"]
    if delta > CV_TIE_TOLERANCE:
        return True
    if delta < -CV_TIE_TOLERANCE:
        return False
    tail_delta = left["tail_accuracy"] - right["tail_accuracy"]
    if abs(tail_delta) > 1e-12:
        return tail_delta > 0.0
    if left["cv_macro_nll"] != right["cv_macro_nll"]:
        return left["cv_macro_nll"] < right["cv_macro_nll"]
    return left["name"] < right["name"]


def stratified_paired_bootstrap(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    repeats: int,
    seed: int,
) -> dict[str, float]:
    labels = reference["labels"]
    reference_correct = reference["predictions"] == labels
    candidate_correct = candidate["predictions"] == labels
    rng = np.random.default_rng(seed)
    deltas = np.zeros(repeats, dtype=np.float64)
    classes = np.unique(labels)
    for label in classes:
        indices = np.where(labels == label)[0]
        paired = candidate_correct[indices].astype(np.float32) - reference_correct[indices].astype(
            np.float32
        )
        draws = rng.integers(0, len(indices), size=(repeats, len(indices)))
        deltas += paired[draws].mean(axis=1)
    deltas /= len(classes)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "observed_delta": float(candidate["macro_accuracy"] - reference["macro_accuracy"]),
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_delta_gt_zero": float((deltas > 0.0).mean()),
        "repeats": int(repeats),
    }


def main() -> None:
    args = parse_args()
    if len(args.run_dirs) < 2 or args.bootstrap < 1:
        raise ValueError("compare_v7 requires at least two runs and a positive bootstrap count")
    runs = [load_run(value) for value in args.run_dirs]
    first = runs[0]
    for run in runs[1:]:
        if run["row_keys"] != first["row_keys"] or not np.array_equal(run["labels"], first["labels"]):
            raise ValueError("V7 runs do not use the exact same clean validation split/order")
    ranked: list[dict[str, Any]] = []
    for run in runs:
        inserted = False
        for index, incumbent in enumerate(ranked):
            if _better(run, incumbent):
                ranked.insert(index, run)
                inserted = True
                break
        if not inserted:
            ranked.append(run)
    winner = ranked[0]
    comparisons = {
        run["name"]: stratified_paired_bootstrap(
            first, run, args.bootstrap, args.seed + index
        )
        for index, run in enumerate(runs[1:], 1)
    }
    public_keys = (
        "name",
        "path",
        "policy",
        "sampler",
        "selected_epoch",
        "cv_macro_accuracy",
        "cv_macro_nll",
        "macro_accuracy",
        "macro_nll",
        "tail_accuracy",
        "mid_accuracy",
        "head_accuracy",
    )
    payload = {
        "selection_rule": "cv macro; within 0.05 percentage points prefer tail, then cv macro NLL",
        "baseline": first["name"],
        "winner": winner["name"],
        "recommended_conflict_policy": winner["policy"],
        "ranked": [{key: run[key] for key in public_keys} for run in ranked],
        "paired_bootstrap_vs_baseline": comparisons,
        "acceptance_vs_baseline": {
            run["name"]: {
                "macro_delta": float(run["macro_accuracy"] - first["macro_accuracy"]),
                "tail_delta": float(run["tail_accuracy"] - first["tail_accuracy"]),
                "passes": bool(
                    run["macro_accuracy"] - first["macro_accuracy"] >= 0.003
                    or (
                        run["tail_accuracy"] - first["tail_accuracy"] >= 0.01
                        and run["macro_accuracy"] - first["macro_accuracy"] >= -0.001
                    )
                ),
            }
            for run in runs[1:]
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
