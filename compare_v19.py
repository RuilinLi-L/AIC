"""Compare V19 candidates to the frozen V18 rank32 validation baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_v10 import compare as paired_comparison, load_evaluation
from select_v14 import read_evaluation
from v18_core import validate_recipe_config as validate_v18_recipe
from v18_model import (RESOLUTION_VERSION as V18_RESOLUTION_VERSION,
                       validate_checkpoint_state_metadata as validate_v18_state)
from v18_runtime import scientific_config as v18_scientific_config
from v19_core import file_sha256, validate_recipe_config
from v19_model import RESOLUTION_VERSION, checkpoint_resolution, validate_checkpoint_state_metadata
from v19_runtime import scientific_config

INPUT_KEYS = ("baseline_v18", "candidate_a", "candidate_b")
CANDIDATE_KEYS = INPUT_KEYS[1:]
EXPECTED = {"baseline_v18": (18, "rank32"),
            "candidate_a": (19, "resolution384_rank32"), "candidate_b": (19, "rank32_dropout")}
CONDITIONS = ("native", "resize384", "jpeg75", "combined")
RESOURCE_BRANCHES = ("standard", "deduplicated_control")
GATES = {"macro_gain": .002, "tail_tolerance": .003, "overall_noninferiority": True}
FROZEN_V18_NATIVE = {"macro_accuracy": .7591118812561035,
                     "tail_accuracy": .7103818655014038, "micro_accuracy": .7704015970230103}
DECISION_RULE = "native macro gain >= 0.002; tail loss <= 0.003; micro >= reference without rounding; then macro, tail, lower NLL, A"
GEOMETRY_FIELDS = ("image_size", "zoom_shortest_edge", "lora_rank")
DROPOUT_FIELDS = ("lora_dropout", "mlp_lora_dropout")


def expected_for_branch(branch):
    if branch not in RESOURCE_BRANCHES:
        raise ValueError("invalid V19 resource branch")
    result = dict(EXPECTED)
    if branch == "deduplicated_control":
        result["baseline_v18"] = (19, "rank32_control")
    return result


def choose(summaries, resource_branch="standard"):
    """Apply the frozen gates to available candidates; missing runs cannot win."""
    expected_for_branch(resource_branch)
    metric = lambda key, name: summaries[key]["conditions"]["native"]["outer_metrics"][name]
    floor = FROZEN_V18_NATIVE if resource_branch == "deduplicated_control" else None
    eligible = {key: key in summaries and (
        metric(key, "macro_accuracy") - metric("baseline_v18", "macro_accuracy") >= .002 - 1e-12
        and metric(key, "tail_accuracy") - metric("baseline_v18", "tail_accuracy") >= -.003 - 1e-12
        and metric(key, "micro_accuracy") >= metric("baseline_v18", "micro_accuracy")
        and (floor is None or (
            metric(key, "macro_accuracy") - floor["macro_accuracy"] >= .002 - 1e-12
            and metric(key, "tail_accuracy") - floor["tail_accuracy"] >= -.003 - 1e-12
            and metric(key, "micro_accuracy") >= floor["micro_accuracy"])))
        for key in CANDIDATE_KEYS}
    accepted = [key for key, value in eligible.items() if value]
    winner = max(accepted, key=lambda key: (
        metric(key, "macro_accuracy"), metric(key, "tail_accuracy"),
        -metric(key, "macro_nll"), key == "candidate_a")) if accepted else "baseline_v18"
    return "baseline_v18", eligible, winner


def _read_resource_plan(path):
    # The resource reader also verifies genuine CUDA-OOM evidence for unavailable runs.
    from v19_pipeline_support import read_resource_plan
    return read_resource_plan(path)


def _audit_binding(plan):
    binding = plan.get("audit", {})
    path = Path(binding.get("path", "")).resolve()
    if not path.is_file() or file_sha256(path) != binding.get("sha256"):
        raise ValueError("V19 generalization audit missing or changed")
    audit = json.loads(path.read_text())
    if (audit.get("format_version") != 19 or audit.get("kind") != "v19_generalization_audit"
            or audit.get("status") != "passed"
            or audit.get("dataset_signature") != plan.get("dataset_signature")
            or type(audit.get("decoded_cross_split_groups")) is not int
            or type(audit.get("decoded_cross_split_pairs")) is not int
            or audit["decoded_cross_split_groups"] != 0 or audit["decoded_cross_split_pairs"] != 0):
        raise ValueError("V19 generalization audit must pass without decoded train/validation overlap")
    return {"path": str(path), "sha256": binding["sha256"]}


def resource_binding(resource_json, paths, items, resource_branch=None):
    if resource_json is None:
        raise ValueError("V19 comparison requires a frozen resource JSON")
    resource_path = Path(resource_json).resolve()
    plan = _read_resource_plan(resource_path)
    branch = plan["branch"]
    expected = expected_for_branch(branch)
    if resource_branch is not None and resource_branch != branch:
        raise ValueError("resource branch differs from frozen resource plan")
    if (plan.get("format_version") != 19 or plan.get("gates") != GATES
            or plan.get("activation_checkpointing") is not False):
        raise ValueError("resource plan changes approved V19 gates or checkpointing")
    audit = _audit_binding(plan)
    digest = file_sha256(resource_path)
    baseline = items["baseline_v18"]
    frozen_baseline = plan["baseline"]
    if branch == "standard":
        if (Path(frozen_baseline["directory"]).resolve() not in (paths["baseline_v18"], paths["baseline_v18"].parent)
                or frozen_baseline["model_sha256"] != baseline["checkpoint_sha256"]
                or frozen_baseline["evaluation_sha256"] != file_sha256(paths["baseline_v18"] / "strict_eval.json")):
            raise ValueError("fixed V18 baseline differs from frozen resource plan")
    else:
        entry = plan["control"]
        config = baseline["checkpoint"]["config"]
        if (entry["status"] != "runnable" or Path(entry["run_dir"]).resolve() not in
                (paths["baseline_v18"], paths["baseline_v18"].parent)
                or config.get("recipe") != "rank32_control"
                or config.get("resource_json") != str(resource_path)
                or config.get("resource_sha256") != digest
                or config.get("resource_branch") != branch
                or config.get("audit_json") != audit["path"]
                or config.get("audit_sha256") != audit["sha256"]
                or any(config.get(field) != entry[field] for field in
                       ("batch_size", "gradient_accumulation", "workers", "image_size", "lora_rank"))):
            raise ValueError("deduplicated matched control differs from frozen resource plan")
    reference = baseline["checkpoint"]
    if (reference["dataset_signature"] != plan["dataset_signature"]
            or reference["config"]["base_model_identity"] != plan["base_model_identity"]):
        raise ValueError("fixed V18 baseline dataset or official base differs from resource plan")
    metrics = baseline["summary"]["conditions"]["native"]["outer_metrics"]
    if branch == "standard" and any(abs(metrics[name] - value) > 1e-12 for name, value in FROZEN_V18_NATIVE.items()):
        raise ValueError("fixed V18 baseline native metrics differ from the approved historical evaluation")
    statuses = {}
    for key in CANDIDATE_KEYS:
        entry = plan["candidates"][key]
        status = entry.get("status")
        if entry.get("recipe") != expected[key][1] or status not in ("runnable", "resource_infeasible"):
            raise ValueError(f"{key} recipe or status differs from the frozen resource plan")
        statuses[key] = status
        if status == "resource_infeasible":
            if key in items:
                raise ValueError(f"{key} resource_infeasible candidate must not supply an evaluation")
            continue
        if key not in items:
            raise ValueError(f"runnable {key} is missing its evaluation; only verified CUDA OOM permits absence")
        checkpoint = items[key]["checkpoint"]
        config = checkpoint["config"]
        if config.get("recipe") != entry["recipe"]:
            raise ValueError(f"{key} recipe differs from resource plan")
        if Path(entry["run_dir"]).resolve() not in (paths[key], paths[key].parent):
            raise ValueError(f"{key} evaluation directory differs from frozen run directory")
        for field in ("batch_size", "gradient_accumulation", "workers", "image_size", "lora_rank"):
            if config.get(field) != entry[field]:
                raise ValueError(f"{key} {field} differs from frozen resource plan")
        if (config.get("resource_json") != str(resource_path)
                or config.get("resource_sha256") != digest or config.get("resource_branch") != branch):
            raise ValueError(f"{key} resource provenance differs from frozen plan")
        if config.get("audit_json") != audit["path"] or config.get("audit_sha256") != audit["sha256"]:
            raise ValueError(f"{key} generalization audit differs from frozen plan")
        if (checkpoint["dataset_signature"] != plan["dataset_signature"]
                or config["base_model_identity"] != plan["base_model_identity"]):
            raise ValueError(f"{key} dataset or official base differs from resource plan")
    return {"resource_json": str(resource_path), "resource_sha256": digest,
            "resource_branch": branch, "audit": audit, "candidate_statuses": statuses}


def validate_inputs(baseline, candidate_a=None, candidate_b=None, *, resource_json, resource_branch=None):
    paths = {key: Path(value).resolve() for key, value in
             zip(INPUT_KEYS, (baseline, candidate_a, candidate_b)) if value is not None}
    if "baseline_v18" not in paths or len(set(paths.values())) != len(paths):
        raise ValueError("V19 comparison requires a baseline and distinct evaluation directories")
    loaded = {key: load_evaluation(path) for key, path in paths.items()}
    items = {key: read_evaluation(path) for key, path in paths.items()}
    binding = resource_binding(resource_json, paths, items, resource_branch)
    reference = loaded["baseline_v18"]
    reference_checkpoint = items["baseline_v18"]["checkpoint"]
    for key, (summary, labels, rows, logits) in loaded.items():
        checkpoint = items[key]["checkpoint"]
        config = checkpoint["config"]
        version, recipe = expected_for_branch(binding["resource_branch"])[key]
        if (checkpoint.get("format_version") != version or config.get("recipe") != recipe
                or summary.get("format_version") != version or summary.get("recipe") != recipe):
            raise ValueError(f"{key} requires format {version} recipe {recipe}")
        (validate_v18_state if version == 18 else validate_checkpoint_state_metadata)(checkpoint)
        (validate_v18_recipe if version == 18 else validate_recipe_config)(config)
        for field in ("class_names", "dataset_signature"):
            if checkpoint[field] != reference_checkpoint[field]:
                raise ValueError(f"{key} checkpoint {field} differs")
        for field in ("base_model_identity", "conflict_policy"):
            if config[field] != reference_checkpoint["config"][field]:
                raise ValueError(f"{key} {field} differs")
        if summary.get("image_size") != checkpoint_resolution(checkpoint):
            raise ValueError(f"{key} evaluation image_size differs from its checkpoint")
        calibration = checkpoint["calibration"]
        if calibration.get("selected_epoch") != checkpoint["selected_epoch"]:
            raise ValueError(f"{key} calibration epoch differs from checkpoint")
        for field in GEOMETRY_FIELDS + (DROPOUT_FIELDS if version == 19 else ()):
            if calibration.get(field) != config[field] or summary.get(field) != config[field]:
                raise ValueError(f"{key} calibration/evaluation {field} differs from checkpoint")
        if version == 19:
            for field in ("resource_json", "resource_sha256", "resource_branch"):
                if summary.get(field) != binding[field]:
                    raise ValueError(f"{key} evaluation {field} differs from resource plan")
            for field in ("audit_json", "audit_sha256"):
                if summary.get(field) != config[field]:
                    raise ValueError(f"{key} evaluation {field} differs from checkpoint")
        preprocessing = V18_RESOLUTION_VERSION if version == 18 else RESOLUTION_VERSION
        scientific = v18_scientific_config if version == 18 else scientific_config
        if (summary.get("preprocessing_version") != preprocessing
                or json.loads(json.dumps(summary.get("scientific_config"))) !=
                json.loads(json.dumps(scientific(config)))):
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


def compare(baseline, candidate_a=None, candidate_b=None, bootstrap_repeats=2000, *, resource_json, resource_branch=None):
    paths, items, loaded, binding = validate_inputs(baseline, candidate_a, candidate_b,
        resource_json=resource_json, resource_branch=resource_branch)
    summaries = {key: value[0] for key, value in loaded.items()}
    baseline_key, eligible, winner_key = choose(summaries, binding["resource_branch"])
    candidates = {}
    for key in CANDIDATE_KEYS:
        if key not in paths:
            candidates[key] = {"candidate_accepted": False, "recommended_for_submission": False,
                               "status": "resource_infeasible", "decision_rule": DECISION_RULE,
                               "reason": "CUDA OOM established by the frozen engineering benchmark"}
            continue
        paired = paired_comparison(paths[baseline_key], paths[key], bootstrap_repeats)
        native = paired["conditions"]["native"]
        paired.update(candidate_accepted=eligible[key], recommended_for_submission=eligible[key],
                      status="eligible" if eligible[key] else "rejected", decision_rule=DECISION_RULE,
                      macro_gain_pp=100 * native["macro_delta"], tail_gain_pp=100 * native["tail_delta"],
                      micro_gain_pp=100 * native["micro_delta"],
                      stress_gain_pp={name: 100 * paired["conditions"][name]["macro_delta"] for name in CONDITIONS[1:]})
        candidates[key] = paired
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
    return {"format_version": 19, **binding,
        "inputs": {key: str(paths[key]) if key in paths else None for key in INPUT_KEYS},
        "class_names": class_names, "class_validation_support": support, "per_class_accuracy": per_class,
        "baseline_key": baseline_key, "baseline": str(paths[baseline_key]),
        "baseline_native_metrics": summaries[baseline_key]["conditions"]["native"]["outer_metrics"],
        "candidates": candidates, "winner_key": winner_key, "winner": str(paths[winner_key]),
        "candidate_accepted": winner_key in CANDIDATE_KEYS, "refit_required": winner_key in CANDIDATE_KEYS,
        "gates": GATES, "decision_rule": DECISION_RULE + (
            "; deduplicated matched control and original V18 absolute floors" if binding["resource_branch"] == "deduplicated_control"
            else "; fixed V18 rank32") + "; only verified CUDA OOM permits a missing candidate",
        "stress_role": "diagnostic_only_not_a_selection_gate", "bootstrap_role": "diagnostic_only_not_a_selection_gate",
        "online_improvement_confirmed": False}


def add_arguments(parser):
    parser.add_argument("--baseline", "--baseline-v18", dest="baseline_v18", required=True, type=Path)
    for key in CANDIDATE_KEYS:
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
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
    if (any(args.output.resolve().is_relative_to(Path(value)) for value in result["inputs"].values() if value is not None)
            or args.output.resolve() in (args.resource_json.resolve(), Path(result["audit"]["path"]))):
        raise ValueError("comparison output must not overwrite an input artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
