"""Report V12 versus V9 under the same validation and calibration procedure."""

import argparse
import json
from pathlib import Path

import numpy as np

from compare_v10 import load_evaluation


def compare(baseline_dir: Path, candidate_dir: Path) -> dict:
    baseline, baseline_labels, baseline_keys, baseline_logits = load_evaluation(baseline_dir)
    candidate, labels, keys, candidate_logits = load_evaluation(candidate_dir)
    if keys != baseline_keys or not np.array_equal(labels, baseline_labels):
        raise ValueError("baseline and V12 must use the same validation rows and labels")
    for field in ("dataset_signature", "outer_fold_ids", "method", "precision"):
        if field not in baseline or baseline[field] != candidate.get(field):
            raise ValueError(f"baseline and V12 differ in {field}")
    conditions = {}
    for name in baseline_logits:
        if baseline_logits[name].shape != candidate_logits[name].shape:
            raise ValueError("validation logit shapes differ")
        base = baseline["conditions"][name]["outer_metrics"]
        new = candidate["conditions"][name]["outer_metrics"]
        conditions[name] = {
            "baseline": base, "v12": new,
            "delta_percentage_points": {
                metric: 100.0 * (new[metric] - base[metric])
                for metric in ("macro_accuracy", "micro_accuracy", "tail_accuracy")
            },
        }
    return {
        "baseline": str(baseline_dir.resolve()), "candidate": str(candidate_dir.resolve()),
        "baseline_format_version": baseline.get("format_version"),
        "candidate_format_version": candidate.get("format_version"),
        "conditions": conditions,
        "status": "experimental_pending_online_evaluation",
        "online_improvement_confirmed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
