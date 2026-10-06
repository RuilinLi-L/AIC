"""Compare two V17 noise-treatment candidates against the fixed V15 baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_v10 import compare as paired_comparison, load_evaluation
from select_v14 import read_evaluation
from v15_model import validate_checkpoint_state_metadata as validate_v15_state
from v17_model import validate_checkpoint_state_metadata as validate_v17_state


INPUT_KEYS = ("baseline_v15", "candidate_a", "candidate_b")
CANDIDATE_KEYS = INPUT_KEYS[1:]
EXPECTED = {"baseline_v15": (15, "expanded_mlp"),
            "candidate_a": (17, "agreement_recovery"),
            "candidate_b": (17, "dynamic_prototype")}
CONDITIONS = ("native", "resize384", "jpeg75", "combined")


def choose(summaries):
    """Deterministic native-only gate; pressure conditions remain diagnostics."""
    metric = lambda key, name: summaries[key]["conditions"]["native"]["outer_metrics"][name]
    baseline_key = "baseline_v15"
    eligible = {
        key: (metric(key, "macro_accuracy") - metric(baseline_key, "macro_accuracy") >= .002 - 1e-12
              and metric(key, "tail_accuracy") - metric(baseline_key, "tail_accuracy") >= -.003 - 1e-12)
        for key in CANDIDATE_KEYS
    }
    accepted = [key for key, value in eligible.items() if value]
    winner = max(accepted, key=lambda key: (
        metric(key, "macro_accuracy"), metric(key, "tail_accuracy"),
        -metric(key, "macro_nll"), key == "candidate_a")) if accepted else baseline_key
    return baseline_key, eligible, winner


def compare(baseline, candidate_a, candidate_b, bootstrap_repeats=2000):
    paths = dict(zip(INPUT_KEYS, map(lambda p: Path(p).resolve(),
                                    (baseline, candidate_a, candidate_b))))
    if len(set(paths.values())) != 3:
        raise ValueError("V17 comparison requires three distinct evaluation directories")
    loaded = {key: load_evaluation(path) for key, path in paths.items()}
    sources = {key: read_evaluation(path)["checkpoint"] for key, path in paths.items()}
    summaries = {key: value[0] for key, value in loaded.items()}
    reference = loaded["baseline_v15"]
    for key, (summary, labels, rows, logits) in loaded.items():
        version, recipe = EXPECTED[key]
        checkpoint = sources[key]
        if (checkpoint.get("format_version") != version or checkpoint["config"].get("recipe") != recipe
                or summary.get("format_version") != version or summary.get("recipe") != recipe):
            raise ValueError(f"{key} requires format {version} recipe {recipe}")
        (validate_v15_state if version == 15 else validate_v17_state)(checkpoint)
        for field in ("class_names", "dataset_signature"):
            if sources[key][field] != sources["baseline_v15"][field]:
                raise ValueError(f"{key} checkpoint {field} differs")
        if sources[key]["config"]["base_model_identity"] != sources["baseline_v15"]["config"]["base_model_identity"]:
            raise ValueError(f"{key} official base_model_identity differs")
        for field in ("dataset_signature", "outer_fold_ids", "precision", "stress_version", "image_size"):
            if field not in summary or field not in reference[0] or summary[field] != reference[0][field]:
                raise ValueError(f"{key} {field} differs or is missing")
        if rows != reference[2] or not np.array_equal(labels, reference[1]):
            raise ValueError(f"{key} validation rows/labels differ")
        if len(set(rows)) != len(rows):
            raise ValueError("validation row keys are not unique")
        if (not np.issubdtype(labels.dtype, np.integer) or not len(labels)
                or np.any(labels < 0) or np.any(labels >= len(checkpoint["class_names"]))):
            raise ValueError(f"{key} invalid validation label indices")
        if (len(summary["outer_fold_ids"]) != len(labels)
                or set(summary["outer_fold_ids"]) != set(range(5))):
            raise ValueError(f"{key} requires exactly five outer folds covering every row")
        for condition in CONDITIONS:
            metrics = summary["conditions"][condition]["outer_metrics"]
            if not all(np.isfinite(metrics[name]) for name in (
                    "macro_accuracy", "tail_accuracy", "macro_nll", "micro_accuracy")):
                raise ValueError(f"nonfinite {key} {condition} metrics")
            if logits[condition].shape != reference[3][condition].shape:
                raise ValueError(f"{key} {condition} logits shape differs")
            if logits[condition].shape[1] != len(checkpoint["class_names"]):
                raise ValueError(f"{key} {condition} logits class dimension differs")
            prediction = logits[condition].argmax(1)
            macro = np.mean([np.mean(prediction[labels == label] == label) for label in np.unique(labels)])
            if (not np.isclose(macro, metrics["macro_accuracy"], atol=1e-6)
                    or not np.isclose(np.mean(prediction == labels), metrics["micro_accuracy"], atol=1e-6)):
                raise ValueError(f"{key} {condition} summary and OOF logits disagree")
    baseline_key, eligible, winner_key = choose(summaries)
    candidates = {}
    for key in CANDIDATE_KEYS:
        paired = paired_comparison(paths[baseline_key], paths[key], bootstrap_repeats)
        native = paired["conditions"]["native"]
        paired.update(candidate_accepted=eligible[key], recommended_for_submission=eligible[key],
                      status="eligible" if eligible[key] else "rejected",
                      decision_rule="native macro >= +0.002; native tail >= -0.003; stress diagnostic only",
                      macro_gain_pp=100 * native["macro_delta"],
                      tail_gain_pp=100 * native["tail_delta"],
                      micro_gain_pp=100 * native["micro_delta"],
                      stress_gain_pp={name: 100 * paired["conditions"][name]["macro_delta"] for name in CONDITIONS[1:]})
        candidates[key] = paired
    return {
        "format_version": 17, "inputs": {key: str(path) for key, path in paths.items()},
        "baseline_key": baseline_key, "baseline": str(paths[baseline_key]),
        "baseline_native_metrics": summaries[baseline_key]["conditions"]["native"]["outer_metrics"],
        "candidates": candidates, "winner_key": winner_key, "winner": str(paths[winner_key]),
        "candidate_accepted": winner_key in CANDIDATE_KEYS, "refit_required": winner_key in CANDIDATE_KEYS,
        "decision_rule": "fixed V15 expanded_mlp baseline; gain >= 0.2 pp and tail loss <= 0.3 pp; then macro, tail, lower NLL, A",
        "stress_role": "diagnostic_only_not_a_selection_gate",
        "bootstrap_role": "diagnostic_only_not_a_selection_gate",
        "online_improvement_confirmed": False,
    }


def add_arguments(parser):
    parser.add_argument("--baseline", "--baseline-v15", dest="baseline_v15", required=True, type=Path)
    for key in CANDIDATE_KEYS:
        parser.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    result = compare(*(getattr(args, key) for key in INPUT_KEYS), args.bootstrap)
    if any(args.output.resolve().is_relative_to(Path(value)) for value in result["inputs"].values()):
        raise ValueError("comparison output must not overwrite an input evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
