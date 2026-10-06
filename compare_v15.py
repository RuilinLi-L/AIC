"""Compare V15 to the frozen V14 winner; bootstrap is diagnostic only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_v10 import compare as paired_comparison


def compare(baseline_dir: Path, candidate_dir: Path, bootstrap_repeats: int = 2000) -> dict:
    baseline_dir, candidate_dir = Path(baseline_dir), Path(candidate_dir)
    summaries = [json.loads((path / "strict_eval.json").read_text())
                 for path in (baseline_dir, candidate_dir)]
    for key in ("outer_fold_ids", "precision", "stress_version", "image_size"):
        if key not in summaries[0] or key not in summaries[1] or summaries[0][key] != summaries[1][key]:
            raise ValueError(f"baseline and V15 {key} differ or are missing")
    result = paired_comparison(baseline_dir, candidate_dir, bootstrap_repeats)
    native = result["conditions"]["native"]
    accepted = bool(native["macro_delta"] >= .002 - 1e-12
                    and native["tail_delta"] >= -.003 - 1e-12
                    and all(result["conditions"][name]["macro_delta"] >= -.003 - 1e-12
                            for name in ("resize384", "jpeg75", "combined")))
    result.update({
        "candidate_accepted": accepted,
        "recommended_for_submission": accepted,
        "status": "recommended_for_online_test" if accepted else "fallback_to_v14",
        "decision_rule": "native macro >= +0.002; native tail and each stress macro >= -0.003",
        "bootstrap_role": "diagnostic_only_not_a_selection_gate",
        "required_macro_gain_pp": .2, "allowed_regression_pp": .3,
        "macro_gain_pp": 100 * native["macro_delta"],
        "micro_gain_pp": 100 * native["micro_delta"],
        "tail_gain_pp": 100 * native["tail_delta"],
        "stress_gain_pp": {name: 100 * result["conditions"][name]["macro_delta"]
                           for name in ("resize384", "jpeg75", "combined")},
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
