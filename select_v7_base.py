"""Select the clean/partial base for experiment 3 using the locked V7 rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TOLERANCE = 0.0005


def metrics(run_dir: str) -> dict:
    path = Path(run_dir)
    payload = json.loads((path / "validation_metrics.json").read_text(encoding="utf-8"))
    selected = payload["selected"]
    return {
        "run_dir": str(path.resolve()),
        "name": path.name,
        "cv_macro_accuracy": float(selected["cv_macro_accuracy"]),
        "cv_macro_nll": float(selected["cv_macro_nll"]),
        "tail_accuracy": float(selected["tail_accuracy"]),
        "selected_epoch": int(payload["selected_epoch"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--partial", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    clean = metrics(args.clean)
    partial = metrics(args.partial)
    delta = partial["cv_macro_accuracy"] - clean["cv_macro_accuracy"]
    if delta > TOLERANCE:
        winner, policy, reason = partial, "partial", "higher_cv_macro"
    elif delta < -TOLERANCE:
        winner, policy, reason = clean, "drop", "higher_cv_macro"
    elif partial["tail_accuracy"] > clean["tail_accuracy"]:
        winner, policy, reason = partial, "partial", "cv_tie_higher_tail"
    elif partial["tail_accuracy"] < clean["tail_accuracy"]:
        winner, policy, reason = clean, "drop", "cv_tie_higher_tail"
    elif partial["cv_macro_nll"] < clean["cv_macro_nll"]:
        winner, policy, reason = partial, "partial", "tail_tie_lower_cv_nll"
    else:
        winner, policy, reason = clean, "drop", "deterministic_clean_tie_break"
    result = {
        "rule": "cv macro; within 0.05 percentage points prefer tail, then cv macro NLL",
        "clean": clean,
        "partial": partial,
        "winner": winner["name"],
        "conflict_policy": policy,
        "reason": reason,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"V7_TAIL_CONFLICT_POLICY={policy}", flush=True)


if __name__ == "__main__":
    main()
