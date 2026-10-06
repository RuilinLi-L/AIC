"""Compare four complete joint-evaluated candidates against the fixed V19 control."""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from compare_v10 import compare as paired_comparison, load_evaluation
from select_v14 import read_evaluation
from v19_core import file_sha256
from v10_evaluation import _metrics as metrics_from_logits
from v20_evaluation import METHOD, PROTOCOL

INPUT_KEYS = ("control", "v19_resolution384", "v19_dropout", "v20_dora", "v20_loraplus")
CANDIDATE_KEYS = INPUT_KEYS[1:]
EXPECTED = {"control": (19, "rank32_control"),
            "v19_resolution384": (19, "resolution384_rank32"),
            "v19_dropout": (19, "rank32_dropout"),
            "v20_dora": (20, "dora_rank32"), "v20_loraplus": (20, "loraplus_rank32")}
CONDITIONS = ("native", "resize384", "jpeg75", "combined")
GATES = {"macro_gain": .002, "tail_tolerance": .003, "overall_noninferiority": True}
TIE_ORDER = ("v20_loraplus", "v20_dora", "v19_dropout", "v19_resolution384")
GEOMETRY_FIELDS = ("image_size", "zoom_shortest_edge", "lora_rank", "lora_dropout", "mlp_lora_dropout")
DECISION_RULE = "native macro >= control + 0.002; tail >= control - 0.003; micro >= control; rank macro, tail, lower NLL, LoRA+, DoRA, dropout, 384"


def _json_value(value):
    return json.loads(json.dumps(value))


def choose(summaries, resource_branch="deduplicated_control"):
    if resource_branch != "deduplicated_control" or "control" not in summaries:
        raise ValueError("V20 requires the fixed deduplicated rank32 control")
    def metric(key, field):
        value = summaries[key]["conditions"]["native"]["outer_metrics"][field]
        if not np.isfinite(value):
            raise ValueError(f"nonfinite {key}/{field}")
        return value
    eligible = {}
    for key in CANDIDATE_KEYS:
        eligible[key] = key in summaries and (
            metric(key, "macro_accuracy") - metric("control", "macro_accuracy") >= .002 - 1e-12
            and metric(key, "tail_accuracy") - metric("control", "tail_accuracy") >= -.003 - 1e-12
            and metric(key, "micro_accuracy") >= metric("control", "micro_accuracy"))
    accepted = [key for key in CANDIDATE_KEYS if eligible[key]]
    winner = max(accepted, key=lambda key: (metric(key, "macro_accuracy"), metric(key, "tail_accuracy"),
        -metric(key, "macro_nll"), -TIE_ORDER.index(key))) if accepted else "fallback_v18"
    return "control", eligible, winner


def _read_resource_plan(path):
    from v20_pipeline_support import read_resource_plan
    return read_resource_plan(path)


def resource_entries(plan):
    return {"control": plan["control"],
            "v19_resolution384": plan["dependencies"]["resolution384_rank32"],
            "v19_dropout": plan["dependencies"]["rank32_dropout"],
            "v20_dora": plan["candidates"]["candidate_a"],
            "v20_loraplus": plan["candidates"]["candidate_b"]}


def _check_hash(path, digest, description):
    path = Path(path)
    if not path.is_file() or file_sha256(path) != digest:
        raise ValueError(f"{description} missing or changed")
    return path


def _validate_model(checkpoint):
    version = checkpoint["format_version"]
    import_module(f"v{version}_model").validate_checkpoint_state_metadata(checkpoint)
    import_module(f"v{version}_core").validate_recipe_config(checkpoint["config"])


def _validate_source(item, entry, plan, resource_path):
    checkpoint, summary = item["checkpoint"], item["summary"]
    config, version = checkpoint["config"], checkpoint["format_version"]
    _validate_model(checkpoint)
    run = Path(entry["run_dir"]).resolve()
    if summary.get("source_run_dir") != str(run):
        raise ValueError("evaluation source run differs from frozen resource plan")
    resource_binding = plan["v19_resource"] if version == 19 else {
        "path": str(resource_path), "sha256": file_sha256(resource_path)}
    source_resource = _check_hash(resource_binding["path"], resource_binding["sha256"], "source resource plan")
    if (config.get("resource_json") != str(source_resource.resolve())
            or config.get("resource_sha256") != resource_binding["sha256"]
            or config.get("resource_branch") != "deduplicated_control"):
        raise ValueError("checkpoint source resource provenance differs")
    original_plan = json.loads(source_resource.read_text())
    original_entries = list(original_plan.get("candidates", {}).values()) + [original_plan.get("control", {})]
    original_entry = next((value for value in original_entries if value.get("recipe") == config["recipe"]), None)
    if original_entry is None or Path(original_entry["run_dir"]).resolve() != run:
        raise ValueError("checkpoint source recipe/run binding differs")
    for field in ("batch_size", "gradient_accumulation", "workers", *GEOMETRY_FIELDS):
        if field in original_entry and config.get(field) != original_entry[field]:
            raise ValueError(f"checkpoint source {field} differs")
    if (config.get("batch_size") != 256 or config.get("gradient_accumulation") != 1
            or config.get("epochs") != 24 or config.get("schedule_epochs") != 24
            or config.get("stage") != "validate"):
        raise ValueError("joint comparison requires a complete 24-epoch validation configuration")
    for field, expected in (("dataset_signature", plan["dataset_signature"]),):
        if checkpoint[field] != expected or summary.get(field) != expected:
            raise ValueError(f"evaluation {field} differs from frozen resource plan")
    if config.get("base_model_identity") != plan["base_model_identity"]:
        raise ValueError("checkpoint official base differs from frozen resource plan")
    for field, expected in (("audit_json", plan["audit"]["path"]), ("audit_sha256", plan["audit"]["sha256"])):
        if config.get(field) != expected or summary.get(field) != expected:
            raise ValueError(f"evaluation {field} differs from frozen audit")
    for field in ("resource_json", "resource_sha256", "resource_branch"):
        if summary.get(field) != config[field]:
            raise ValueError(f"evaluation {field} differs from source checkpoint")
    scientific = import_module(f"v{version}_runtime").scientific_config(config)
    preprocessing = import_module(f"v{version}_model").RESOLUTION_VERSION
    if (_json_value(summary.get("scientific_config")) != _json_value(scientific)
            or summary.get("preprocessing_version") != preprocessing):
        raise ValueError("evaluation scientific configuration differs from checkpoint")
    hashes = summary.get("epoch_checkpoint_sha256", {})
    if set(hashes) != {str(epoch) for epoch in range(1, 25)}:
        raise ValueError("joint evaluation requires all 24 source epoch hashes")
    for epoch in range(1, 25):
        _check_hash(run / "validation_epochs" / f"epoch_{epoch:02d}.pt", hashes[str(epoch)], "source epoch checkpoint")
    epoch = checkpoint["selected_epoch"]
    if (isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 24
            or summary.get("source_epoch_checkpoint") != str(run / "validation_epochs" / f"epoch_{epoch:02d}.pt")
            or summary.get("checkpoint_sha256") != hashes[str(epoch)]):
        raise ValueError("selected source epoch binding differs")
    original = torch.load(summary["source_epoch_checkpoint"], map_location="cpu", weights_only=False)
    for field in ("format_version", "training_stage", "selected_epoch", "dataset_signature", "class_names", "config"):
        if _json_value(checkpoint[field]) != _json_value(original[field]):
            raise ValueError(f"calibrated source {field} differs from selected epoch")
    calibration = checkpoint["calibration"]
    if (summary.get("method") != METHOD or calibration.get("method") != METHOD
            or _json_value(summary.get("evaluation_protocol")) != _json_value(PROTOCOL)
            or _json_value(calibration.get("evaluation_protocol")) != _json_value(PROTOCOL)
            or summary.get("evaluation_format_version") != 20
            or summary.get("source_format_version") != version):
        raise ValueError("evaluation must use the frozen V20 joint protocol")
    for field in GEOMETRY_FIELDS:
        if summary.get(field) != config[field] or calibration.get(field) != config[field]:
            raise ValueError(f"calibration/evaluation {field} differs from checkpoint")
    if calibration.get("selected_epoch") != epoch:
        raise ValueError("calibration epoch differs from checkpoint")
    if summary.get("class_names") != checkpoint["class_names"] or summary.get("base_model_identity") != plan["base_model_identity"]:
        raise ValueError("evaluation class names or official base differ")
    row_keys = json.loads((Path(item["root"]) / "val_row_keys.json").read_text())
    row_digest = hashlib.sha256(json.dumps(row_keys, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if summary.get("validation_row_keys_sha256") != row_digest:
        raise ValueError("validation row keys missing or changed")


def validate_inputs(control, v19_resolution384=None, v19_dropout=None, v20_dora=None, v20_loraplus=None,
                    *, resource_json, resource_branch=None):
    if resource_branch not in (None, "deduplicated_control"):
        raise ValueError("V20 only supports the deduplicated control branch")
    resource_path = Path(resource_json).resolve()
    plan = _read_resource_plan(resource_path)
    if (plan.get("format_version") != 20 or plan.get("kind") != "v20_resource_plan"
            or plan.get("branch") != "deduplicated_control" or plan.get("gates") != GATES):
        raise ValueError("invalid V20 resource plan or gates")
    paths = {key: Path(value).resolve() for key, value in zip(INPUT_KEYS,
        (control, v19_resolution384, v19_dropout, v20_dora, v20_loraplus)) if value is not None}
    if "control" not in paths or len(paths) != len(set(paths.values())):
        raise ValueError("V20 requires a control and distinct evaluation directories")
    entries = resource_entries(plan)
    statuses = {}
    for key, entry in entries.items():
        status = entry.get("status", "runnable" if key == "control" else None)
        if status not in ("runnable", "resource_infeasible") or (key == "control" and status != "runnable"):
            raise ValueError(f"{key} invalid resource status; incomplete dependencies must wait")
        if key in CANDIDATE_KEYS:
            statuses[key] = status
        if status == "resource_infeasible" and key in paths:
            raise ValueError(f"{key} resource_infeasible candidate must not supply an evaluation")
        if status == "runnable" and (key not in paths or not (paths[key] / "strict_eval.json").is_file()):
            raise ValueError(f"runnable {key} is incomplete; wait for its joint evaluation")
    loaded = {key: load_evaluation(path) for key, path in paths.items()}
    items = {key: read_evaluation(path) for key, path in paths.items()}
    reference = loaded["control"]
    reference_checkpoint = items["control"]["checkpoint"]
    for key, (summary, labels, rows, logits) in loaded.items():
        checkpoint = items[key]["checkpoint"]
        version, recipe = EXPECTED[key]
        if (checkpoint.get("format_version") != version or checkpoint["config"].get("recipe") != recipe
                or summary.get("format_version") != version or summary.get("recipe") != recipe):
            raise ValueError(f"{key} requires format {version} recipe {recipe}")
        _validate_source(items[key], entries[key], plan, resource_path)
        for field in ("dataset_signature", "class_names"):
            if checkpoint[field] != reference_checkpoint[field]:
                raise ValueError(f"{key} checkpoint {field} differs")
        for field in ("dataset_signature", "outer_fold_ids", "precision", "stress_version", "evaluation_protocol", "class_training_support"):
            if field not in summary or _json_value(summary[field]) != _json_value(reference[0].get(field)):
                raise ValueError(f"{key} {field} differs or is missing")
        if rows != reference[2] or not np.array_equal(labels, reference[1]):
            raise ValueError(f"{key} validation rows/labels differ")
        if (len(set(rows)) != len(rows) or not np.issubdtype(labels.dtype, np.integer)
                or not len(labels) or np.any(labels < 0) or np.any(labels >= len(checkpoint["class_names"]))):
            raise ValueError("invalid unique validation rows or label indices")
        if len(summary["outer_fold_ids"]) != len(labels) or set(summary["outer_fold_ids"]) != set(range(5)):
            raise ValueError("exactly five outer folds must cover every validation row")
        class_support = np.asarray(summary["class_training_support"])
        if (class_support.shape != (len(checkpoint["class_names"]),)
                or not np.issubdtype(class_support.dtype, np.integer) or np.any(class_support < 0)
                or not np.any(class_support > 0)):
            raise ValueError("invalid frozen training class support")
        for condition in CONDITIONS:
            metrics = summary["conditions"][condition]["outer_metrics"]
            if not all(np.isfinite(metrics[name]) for name in ("macro_accuracy", "tail_accuracy", "macro_nll", "micro_accuracy")):
                raise ValueError(f"nonfinite {key}/{condition} metrics")
            if logits[condition].shape != (len(labels), len(checkpoint["class_names"])):
                raise ValueError(f"{key} logits dimensions differ")
            measured = metrics_from_logits(logits[condition], labels, class_support)
            if any(not np.isclose(measured[name], metrics[name], atol=1e-6, rtol=0)
                   for name in ("macro_accuracy", "micro_accuracy", "tail_accuracy", "macro_nll")):
                raise ValueError(f"{key} summary and OOF logits disagree")
    return paths, items, loaded, {"resource_json": str(resource_path), "resource_sha256": file_sha256(resource_path),
        "resource_branch": plan["branch"], "audit": plan["audit"], "candidate_statuses": statuses,
        "historical_refit": plan["historical_refit"]}


def compare(control, v19_resolution384=None, v19_dropout=None, v20_dora=None, v20_loraplus=None,
            bootstrap_repeats=2000, *, resource_json, resource_branch=None):
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    paths, items, loaded, binding = validate_inputs(control, v19_resolution384, v19_dropout, v20_dora,
        v20_loraplus, resource_json=resource_json, resource_branch=resource_branch)
    summaries = {key: value[0] for key, value in loaded.items()}
    baseline_key, eligible, winner_key = choose(summaries)
    candidates = {}
    for key in CANDIDATE_KEYS:
        if key not in paths:
            candidates[key] = {"candidate_accepted": False, "recommended_for_submission": False,
                "status": "resource_infeasible", "reason": "explicit frozen resource failure; no evaluation supplied"}
            continue
        paired = paired_comparison(paths["control"], paths[key], bootstrap_repeats)
        native = paired["conditions"]["native"]
        paired.update(candidate_accepted=eligible[key], recommended_for_submission=eligible[key],
            status="eligible" if eligible[key] else "rejected", decision_rule=DECISION_RULE,
            macro_gain_pp=100 * native["macro_delta"], tail_gain_pp=100 * native["tail_delta"],
            micro_gain_pp=100 * native["micro_delta"])
        candidates[key] = paired
    labels, names = loaded["control"][1], items["control"]["checkpoint"]["class_names"]
    support = np.bincount(labels, minlength=len(names)).tolist()
    per_class = {}
    for key, (_, _, _, logits) in loaded.items():
        per_class[key] = {}
        for condition in CONDITIONS:
            correct = np.bincount(labels, weights=logits[condition].argmax(1) == labels, minlength=len(names))
            per_class[key][condition] = [float(correct[label] / count) if count else None for label, count in enumerate(support)]
    accepted = winner_key in CANDIDATE_KEYS
    return {"format_version": 20, **binding,
        "inputs": {key: str(paths[key]) if key in paths else None for key in INPUT_KEYS},
        "class_names": names, "class_validation_support": support, "per_class_accuracy": per_class,
        "dataset_signature": items["control"]["checkpoint"]["dataset_signature"],
        "baseline_key": baseline_key, "baseline": str(paths[baseline_key]),
        "baseline_native_metrics": summaries[baseline_key]["conditions"]["native"]["outer_metrics"],
        "candidates": candidates, "winner_key": winner_key, "winner": str(paths[winner_key]) if accepted else None,
        "candidate_accepted": accepted, "refit_required": accepted,
        "gates": GATES, "decision_rule": DECISION_RULE, "evaluation_protocol": PROTOCOL,
        "stress_role": "diagnostic_only_not_a_selection_gate", "bootstrap_role": "diagnostic_only_not_a_selection_gate",
        "online_improvement_confirmed": False}


def add_arguments(parser):
    parser.add_argument("--control", "--baseline", dest="control", required=True, type=Path)
    for key in CANDIDATE_KEYS:
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    parser.add_argument("--resource-json", required=True, type=Path)
    parser.add_argument("--resource-branch", choices=["deduplicated_control"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    result = compare(*(getattr(args, key) for key in INPUT_KEYS), args.bootstrap,
                     resource_json=args.resource_json, resource_branch=args.resource_branch)
    output = args.output.resolve()
    if (any(output.is_relative_to(Path(value)) for value in result["inputs"].values() if value)
            or output in (args.resource_json.resolve(), Path(result["audit"]["path"]))):
        raise ValueError("comparison output must not overwrite source artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
