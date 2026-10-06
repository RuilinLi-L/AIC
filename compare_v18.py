"""Compare V18 candidates under the frozen standard or matched-control resource plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_v10 import compare as paired_comparison, load_evaluation
from select_v14 import read_evaluation
from v15_model import validate_checkpoint_state_metadata as validate_v15_state
from v18_core import file_sha256, validate_recipe_config
from v18_model import RESOLUTION_VERSION, checkpoint_resolution, validate_checkpoint_state_metadata
from v18_runtime import scientific_config

INPUT_KEYS = ("baseline_v15", "candidate_a", "candidate_b")
CANDIDATE_KEYS = INPUT_KEYS[1:]
EXPECTED = {"baseline_v15": (15, "expanded_mlp"),
            "candidate_a": (18, "resolution384"), "candidate_b": (18, "rank32")}
CONDITIONS = ("native", "resize384", "jpeg75", "combined")
RESOURCE_BRANCHES = ("standard", "matched_control")
GATES = {"macro_gain": .002, "tail_tolerance": .003, "overall_noninferiority": True}
FROZEN_V15_NATIVE = {"macro_accuracy": .7546586990356445, "tail_accuracy": .707298755645752}
DECISION_RULE = "native macro gain >= 0.002; tail loss <= 0.003; micro >= reference without rounding; then macro, tail, lower NLL, A"


def expected_for_branch(branch):
    if branch not in RESOURCE_BRANCHES:
        raise ValueError("invalid V18 resource branch")
    return {**EXPECTED, "candidate_b": (18, "matched_control" if branch == "matched_control" else "rank32")}


def choose(summaries, resource_branch="standard"):
    """Apply all three gates against V15, and additionally the matched control."""
    expected_for_branch(resource_branch)
    metric = lambda key, name: summaries[key]["conditions"]["native"]["outer_metrics"][name]
    def passes(key, reference):
        return (metric(key, "macro_accuracy") - metric(reference, "macro_accuracy") >= .002 - 1e-12
                and metric(key, "tail_accuracy") - metric(reference, "tail_accuracy") >= -.003 - 1e-12
                and metric(key, "micro_accuracy") >= metric(reference, "micro_accuracy"))
    eligible = {key: passes(key, "baseline_v15") for key in CANDIDATE_KEYS}
    if resource_branch == "matched_control":
        eligible["candidate_a"] = eligible["candidate_a"] and passes("candidate_a", "candidate_b")
        eligible["candidate_b"] = False
    accepted = [key for key, value in eligible.items() if value]
    winner = max(accepted, key=lambda key: (
        metric(key, "macro_accuracy"), metric(key, "tail_accuracy"),
        -metric(key, "macro_nll"), key == "candidate_a")) if accepted else "baseline_v15"
    return "baseline_v15", eligible, winner


def _read_resource_plan(path):
    # Lazy import keeps command-line pipeline helpers free from import cycles.
    from v18_pipeline_support import read_resource_plan
    return read_resource_plan(path)


def resource_binding(resource_json, paths, items, resource_branch=None):
    if resource_json is None:
        raise ValueError("V18 comparison requires a frozen resource JSON")
    resource_path = Path(resource_json).resolve()
    plan = _read_resource_plan(resource_path)
    branch = plan["branch"]
    expected = expected_for_branch(branch)
    if resource_branch is not None and resource_branch != branch:
        raise ValueError("resource branch differs from frozen resource plan")
    digest = file_sha256(resource_path)
    if plan.get("gates") != GATES or plan.get("activation_checkpointing") is not False:
        raise ValueError("resource plan changes approved V18 gates or checkpointing")
    baseline = items["baseline_v15"]
    frozen_baseline = plan["baseline"]
    if (Path(frozen_baseline["directory"]).resolve() not in (paths["baseline_v15"], paths["baseline_v15"].parent)
            or frozen_baseline["model_sha256"] != baseline["checkpoint_sha256"]
            or frozen_baseline["evaluation_sha256"] != file_sha256(paths["baseline_v15"] / "strict_eval.json")):
        raise ValueError("fixed V15 baseline differs from frozen resource plan")
    actual_baseline = baseline["summary"]["conditions"]["native"]["outer_metrics"]
    if any(abs(actual_baseline[name] - value) > 1e-12 for name, value in FROZEN_V15_NATIVE.items()):
        raise ValueError("fixed V15 baseline native metrics differ from the approved historical evaluation")
    for key in CANDIDATE_KEYS:
        checkpoint = items[key]["checkpoint"]
        config = checkpoint["config"]
        entry = plan["candidates"][key]
        if entry["recipe"] != expected[key][1] or config.get("recipe") != entry["recipe"]:
            raise ValueError(f"{key} recipe differs from resource branch")
        if Path(entry["run_dir"]).resolve() not in (paths[key], paths[key].parent):
            raise ValueError(f"{key} evaluation directory differs from frozen run directory")
        for field in ("batch_size", "gradient_accumulation", "workers", "image_size", "lora_rank"):
            if config.get(field) != entry[field]:
                raise ValueError(f"{key} {field} differs from frozen resource plan")
        if (config.get("resource_json") != str(resource_path)
                or config.get("resource_sha256") != digest or config.get("resource_branch") != branch):
            raise ValueError(f"{key} resource provenance differs from frozen plan")
        if (checkpoint["dataset_signature"] != plan["dataset_signature"]
                or config["base_model_identity"] != plan["base_model_identity"]):
            raise ValueError(f"{key} dataset or official base differs from resource plan")
    return {"resource_json": str(resource_path), "resource_sha256": digest, "resource_branch": branch}


def validate_inputs(baseline, candidate_a, candidate_b, *, resource_json, resource_branch=None):
    paths = dict(zip(INPUT_KEYS, (Path(p).resolve() for p in (baseline, candidate_a, candidate_b))))
    if len(set(paths.values())) != 3:
        raise ValueError("V18 comparison requires three distinct evaluation directories")
    loaded = {key: load_evaluation(path) for key, path in paths.items()}
    items = {key: read_evaluation(path) for key, path in paths.items()}
    binding = resource_binding(resource_json, paths, items, resource_branch)
    expected = expected_for_branch(binding["resource_branch"])
    reference = loaded["baseline_v15"]
    reference_checkpoint = items["baseline_v15"]["checkpoint"]
    for key, (summary, labels, rows, logits) in loaded.items():
        checkpoint = items[key]["checkpoint"]
        config = checkpoint["config"]
        version, recipe = expected[key]
        if (checkpoint.get("format_version") != version or config.get("recipe") != recipe
                or summary.get("format_version") != version or summary.get("recipe") != recipe):
            raise ValueError(f"{key} requires format {version} recipe {recipe}")
        (validate_v15_state if version == 15 else validate_checkpoint_state_metadata)(checkpoint)
        for field in ("class_names", "dataset_signature"):
            if checkpoint[field] != reference_checkpoint[field]:
                raise ValueError(f"{key} checkpoint {field} differs")
        for field in ("base_model_identity", "conflict_policy"):
            if config[field] != reference_checkpoint["config"][field]:
                raise ValueError(f"{key} {field} differs")
        size = checkpoint_resolution(checkpoint)
        if summary.get("image_size") != size:
            raise ValueError(f"{key} evaluation image_size differs from its checkpoint")
        if version == 18:
            validate_recipe_config(config)
            calibration = checkpoint["calibration"]
            if calibration.get("selected_epoch") != checkpoint["selected_epoch"]:
                raise ValueError(f"{key} calibration epoch differs from checkpoint")
            for field in ("image_size", "zoom_shortest_edge", "lora_rank"):
                if calibration.get(field) != config[field] or summary.get(field) != config[field]:
                    raise ValueError(f"{key} calibration/evaluation {field} differs from checkpoint")
            for field, value in binding.items():
                if summary.get(field) != value:
                    raise ValueError(f"{key} evaluation {field} differs from resource plan")
            if (summary.get("preprocessing_version") != RESOLUTION_VERSION
                    or json.loads(json.dumps(summary.get("scientific_config"))) !=
                    json.loads(json.dumps(scientific_config(config)))):
                raise ValueError(f"{key} evaluation scientific configuration differs from checkpoint")
        original = torch.load(summary["source_epoch_checkpoint"], map_location="cpu", weights_only=False)
        for field in ("format_version", "training_stage", "selected_epoch", "dataset_signature", "class_names", "config"):
            if json.loads(json.dumps(checkpoint[field])) != json.loads(json.dumps(original[field])):
                raise ValueError(f"{key} calibrated source {field} differs from the selected epoch")
        for field in ("dataset_signature", "outer_fold_ids", "precision", "stress_version"):
            if field not in summary or field not in reference[0] or summary[field] != reference[0][field]:
                raise ValueError(f"{key} {field} differs or is missing")
        if rows != reference[2] or not np.array_equal(labels, reference[1]):
            raise ValueError(f"{key} validation rows/labels differ")
        if len(set(rows)) != len(rows):
            raise ValueError("validation row keys are not unique")
        if (not np.issubdtype(labels.dtype, np.integer) or not len(labels)
                or np.any(labels < 0) or np.any(labels >= len(checkpoint["class_names"]))):
            raise ValueError(f"{key} invalid validation label indices")
        if len(summary["outer_fold_ids"]) != len(labels) or set(summary["outer_fold_ids"]) != set(range(5)):
            raise ValueError(f"{key} requires exactly five outer folds covering every row")
        for condition in CONDITIONS:
            metrics = summary["conditions"][condition]["outer_metrics"]
            if not all(np.isfinite(metrics[name]) for name in ("macro_accuracy", "tail_accuracy", "macro_nll", "micro_accuracy")):
                raise ValueError(f"nonfinite {key} {condition} metrics")
            if logits[condition].shape != reference[3][condition].shape:
                raise ValueError(f"{key} {condition} logits shape differs")
            if logits[condition].shape[1] != len(checkpoint["class_names"]):
                raise ValueError(f"{key} {condition} logits class dimension differs")
            prediction = logits[condition].argmax(1)
            macro = np.mean([np.mean(prediction[labels == label] == label) for label in np.unique(labels)])
            if (not np.isclose(macro, metrics["macro_accuracy"], atol=1e-6, rtol=0)
                    or not np.isclose(np.mean(prediction == labels), metrics["micro_accuracy"], atol=1e-6, rtol=0)):
                raise ValueError(f"{key} {condition} summary and OOF logits disagree")
    return paths, items, loaded, binding


def compare(baseline, candidate_a, candidate_b, bootstrap_repeats=2000, *, resource_json, resource_branch=None):
    paths, items, loaded, binding = validate_inputs(baseline, candidate_a, candidate_b,
        resource_json=resource_json, resource_branch=resource_branch)
    summaries = {key: value[0] for key, value in loaded.items()}
    baseline_key, eligible, winner_key = choose(summaries, binding["resource_branch"])
    candidates = {}
    for key in CANDIDATE_KEYS:
        paired = paired_comparison(paths[baseline_key], paths[key], bootstrap_repeats)
        native = paired["conditions"]["native"]
        control = binding["resource_branch"] == "matched_control" and key == "candidate_b"
        paired.update(candidate_accepted=eligible[key], recommended_for_submission=eligible[key],
                      status="control_only" if control else ("eligible" if eligible[key] else "rejected"),
                      decision_rule=DECISION_RULE, macro_gain_pp=100 * native["macro_delta"],
                      tail_gain_pp=100 * native["tail_delta"], micro_gain_pp=100 * native["micro_delta"],
                      stress_gain_pp={name: 100 * paired["conditions"][name]["macro_delta"] for name in CONDITIONS[1:]})
        candidates[key] = paired
    matched = (paired_comparison(paths["candidate_b"], paths["candidate_a"], bootstrap_repeats)
               if binding["resource_branch"] == "matched_control" else None)
    if matched is not None:
        matched.update(candidate_accepted=eligible["candidate_a"], recommended_for_submission=eligible["candidate_a"],
                       decision_rule=DECISION_RULE, status="eligible" if eligible["candidate_a"] else "rejected")
    labels = loaded[baseline_key][1]
    class_names = items[baseline_key]["checkpoint"]["class_names"]
    support = np.bincount(labels, minlength=len(class_names)).tolist()
    per_class = {}
    for key, (_, _, _, logits) in loaded.items():
        per_class[key] = {}
        for condition in CONDITIONS:
            correct = np.bincount(labels, weights=logits[condition].argmax(1) == labels, minlength=len(class_names))
            per_class[key][condition] = [float(correct[label] / count) if count else None
                                         for label, count in enumerate(support)]
    return {"format_version": 18, **binding, "inputs": {key: str(path) for key, path in paths.items()},
        "class_names": class_names, "class_validation_support": support, "per_class_accuracy": per_class,
        "baseline_key": baseline_key, "baseline": str(paths[baseline_key]),
        "baseline_native_metrics": summaries[baseline_key]["conditions"]["native"]["outer_metrics"],
        "candidates": candidates, "matched_control_comparison": matched,
        "winner_key": winner_key, "winner": str(paths[winner_key]),
        "candidate_accepted": winner_key in CANDIDATE_KEYS, "refit_required": winner_key in CANDIDATE_KEYS,
        "gates": GATES, "decision_rule": DECISION_RULE + "; fixed V15 expanded_mlp; matched branch A must also pass control",
        "stress_role": "diagnostic_only_not_a_selection_gate", "bootstrap_role": "diagnostic_only_not_a_selection_gate",
        "online_improvement_confirmed": False}


def add_arguments(parser):
    parser.add_argument("--baseline", "--baseline-v15", dest="baseline_v15", required=True, type=Path)
    for key in CANDIDATE_KEYS:
        parser.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--resource-json", required=True, type=Path)
    parser.add_argument("--resource-branch", choices=RESOURCE_BRANCHES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    result = compare(*(getattr(args, key) for key in INPUT_KEYS), args.bootstrap,
                     resource_json=args.resource_json, resource_branch=args.resource_branch)
    if (any(args.output.resolve().is_relative_to(Path(value)) for value in result["inputs"].values())
            or args.output.resolve() == args.resource_json.resolve()):
        raise ValueError("comparison output must not overwrite an input artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
