"""Freeze a V19/V20 winning refit or preserve the historical V18 full-refit package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_v20 import (INPUT_KEYS, CANDIDATE_KEYS, CONDITIONS, GATES, PROTOCOL,
                         _check_hash, _json_value, _read_resource_plan, add_arguments,
                         choose, compare, validate_inputs)
from v19_core import file_sha256, model_identity
from v7_views import normalize_view_weights


def _artifact_hashes(root):
    root = Path(root)
    names = ["strict_eval.json", "val_labels.npy", "val_row_keys.json", "val_selected_native.npy",
             *(f"val_oof_{condition}.npy" for condition in CONDITIONS)]
    return {name: file_sha256(root / name) for name in names}


def _fallback(plan, directory=None):
    frozen = plan["historical_refit"]
    root = Path(frozen["directory"]).resolve()
    if directory is not None and Path(directory).resolve() != root:
        raise ValueError("fallback V18 directory differs from frozen historical full refit")
    provenance_path = _check_hash(root / "provenance.json", frozen["provenance_sha256"], "fallback provenance")
    provenance = json.loads(provenance_path.read_text())
    checkpoint_path = Path(provenance["checkpoint"]).resolve()
    for path, field in ((checkpoint_path, "checkpoint_sha256"),
                        (root / "pred_results.csv", "csv_sha256"), (root / "pred_results.zip", "zip_sha256")):
        _check_hash(path, frozen[field], "historical V18 " + field)
        if provenance.get(field) != frozen[field]:
            raise ValueError("fallback provenance artifact binding differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("format_version") != 18 or checkpoint.get("training_stage") != "refit"
            or checkpoint.get("config", {}).get("recipe") != "rank32"
            or checkpoint["config"].get("base_model_identity") != plan["base_model_identity"]):
        raise ValueError("fallback must be the frozen V18 rank32 full-refit model")
    from v18_model import validate_checkpoint_state_metadata
    validate_checkpoint_state_metadata(checkpoint)
    return {"root": str(root), "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": frozen["checkpoint_sha256"], "checkpoint": checkpoint,
            "binding": {**frozen, "directory": str(root), "source_checkpoint": str(checkpoint_path),
                        "recipe": "rank32", "source_format_version": 18, "refit_required": False}}


def _payload(comparison, items, fallback):
    accepted = comparison["refit_required"]
    winner = items[comparison["winner_key"]] if accepted else fallback
    checkpoint = winner["checkpoint"]
    return {
        "format_version": 20, "kind": "v20_refit_selection",
        **{key: comparison[key] for key in ("resource_json", "resource_sha256", "resource_branch", "audit", "candidate_statuses")},
        "gates": GATES, "evaluation_protocol": PROTOCOL,
        "recipe": checkpoint["config"]["recipe"],
        "candidate_accepted": accepted, "refit_required": accepted,
        "selected_stage": "validation" if accepted else "fallback",
        "selected_candidate": comparison["winner_key"] if accepted else None,
        "selected_evaluation": winner["root"] if accepted else None,
        "source_format_version": checkpoint["format_version"],
        "selected_epoch": checkpoint["selected_epoch"], "schedule_epochs": 24,
        "augmentation": checkpoint["config"]["augmentation"],
        "dataset_signature": checkpoint["dataset_signature"], "class_names": checkpoint["class_names"],
        "comparison_dataset_signature": comparison["dataset_signature"],
        "base_model_identity": checkpoint["config"]["base_model_identity"],
        "source_checkpoint": winner["checkpoint_path"], "source_checkpoint_sha256": winner["checkpoint_sha256"],
        "training_config": checkpoint["config"], "calibration": checkpoint["calibration"],
        "class_bias": torch.as_tensor(checkpoint["class_bias"]).tolist(),
        "evaluation_sha256": {key: file_sha256(Path(item["root"]) / "strict_eval.json") for key, item in items.items()},
        "evaluation_artifact_sha256": {key: _artifact_hashes(item["root"]) for key, item in items.items()},
        "comparison": comparison, "baseline_recipe": "rank32_control", "fallback": fallback["binding"],
        "initialization": "official_base_only" if accepted else "reuse_historical_full_refit",
        "online_improvement_confirmed": False,
        "evaluation_scope": "noisy holdout; joint inner-fold epoch/TTA/bias selection; outer folds share training",
    }


def make_selection(control, v19_resolution384=None, v19_dropout=None, v20_dora=None, v20_loraplus=None,
                   identity=None, *, bootstrap_repeats=2000, resource_json, resource_branch=None, fallback_v18=None):
    values = (control, v19_resolution384, v19_dropout, v20_dora, v20_loraplus)
    comparison = compare(*values, bootstrap_repeats, resource_json=resource_json, resource_branch=resource_branch)
    _, items, _, _ = validate_inputs(*values, resource_json=resource_json, resource_branch=resource_branch)
    plan = _read_resource_plan(resource_json)
    fallback = _fallback(plan, fallback_v18)
    result = _payload(comparison, items, fallback)
    if identity is not None and result["base_model_identity"] != identity:
        raise ValueError("V20 selection official base identity differs")
    _validate_calibration(result)
    return result


def _validate_calibration(selected):
    epoch = selected["selected_epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 24 or selected["schedule_epochs"] != 24:
        raise ValueError("selected epoch outside frozen 24-epoch schedule")
    calibration = selected["calibration"]
    weights = calibration["view_weights"]
    if (not isinstance(weights, dict) or not weights
            or any(not np.isfinite(value) or value < 0 for value in weights.values())
            or abs(sum(weights.values()) - 1) > 1e-6):
        raise ValueError("invalid frozen view weights")
    normalize_view_weights(weights)
    if not np.isfinite(calibration["alpha"]) or not 0 <= calibration["alpha"] <= 1:
        raise ValueError("invalid frozen calibration strength")
    if calibration.get("selected_epoch") != epoch:
        raise ValueError("frozen calibration epoch differs")
    bias = np.asarray(selected["class_bias"], dtype=np.float32)
    if bias.shape != (len(selected["class_names"]),) or not np.isfinite(bias).all():
        raise ValueError("invalid frozen class bias")


def read_decision(path, *, dataset_signature=None, class_names=None, base_identity=None):
    selected = json.loads(Path(path).read_text())
    if selected.get("format_version") != 20 or selected.get("kind") != "v20_refit_selection":
        raise ValueError("not a V20 refit selection")
    accepted = selected.get("candidate_accepted")
    if not isinstance(accepted, bool) or selected.get("refit_required") is not accepted:
        raise ValueError("V20 requires matching boolean acceptance and refit fields")
    _check_hash(selected["source_checkpoint"], selected["source_checkpoint_sha256"], "selection source checkpoint")
    _check_hash(selected["resource_json"], selected["resource_sha256"], "frozen V20 resource plan")
    comparison = selected["comparison"]
    for key in ("resource_json", "resource_sha256", "resource_branch", "audit", "candidate_statuses", "gates", "evaluation_protocol"):
        if _json_value(comparison.get(key)) != _json_value(selected.get(key)):
            raise ValueError(f"selection/comparison {key} differs")
    if selected.get("gates") != GATES or _json_value(selected.get("evaluation_protocol")) != _json_value(PROTOCOL):
        raise ValueError("frozen gates or joint evaluation protocol changed")
    inputs = comparison["inputs"]
    active = {key for key, value in inputs.items() if value is not None}
    if (set(inputs) != set(INPUT_KEYS) or set(selected["evaluation_sha256"]) != active
            or set(selected["evaluation_artifact_sha256"]) != active):
        raise ValueError("frozen evaluations differ from available inputs")
    for key in active:
        _check_hash(Path(inputs[key]) / "strict_eval.json", selected["evaluation_sha256"][key], "frozen evaluation summary")
        if _artifact_hashes(inputs[key]) != selected["evaluation_artifact_sha256"][key]:
            raise ValueError("frozen evaluation artifact missing or changed")
    paths, items, loaded, binding = validate_inputs(*(inputs[key] for key in INPUT_KEYS),
        resource_json=selected["resource_json"], resource_branch=selected["resource_branch"])
    baseline_key, eligible, winner_key = choose({key: value[0] for key, value in loaded.items()})
    if (accepted != (winner_key in CANDIDATE_KEYS) or comparison.get("refit_required") is not accepted
            or comparison.get("candidate_accepted") is not accepted or comparison.get("winner_key") != winner_key
            or comparison.get("baseline_key") != baseline_key or comparison.get("baseline") != str(paths[baseline_key])
            or comparison.get("winner") != (str(paths[winner_key]) if accepted else None)
            or comparison.get("dataset_signature") != items[baseline_key]["checkpoint"]["dataset_signature"]
            or set(comparison["candidates"]) != set(CANDIDATE_KEYS)):
        raise ValueError("V20 decision differs from frozen evaluation gate")
    for key in CANDIDATE_KEYS:
        expected_status = "resource_infeasible" if key not in items else "eligible" if eligible[key] else "rejected"
        if (comparison["candidates"][key].get("candidate_accepted") is not eligible[key]
                or comparison["candidates"][key].get("status") != expected_status):
            raise ValueError("candidate status differs from frozen evaluation gate")
    for field in ("audit", "candidate_statuses", "historical_refit"):
        if comparison.get(field) != binding[field]:
            raise ValueError(f"frozen resource {field} differs")
    fallback = _fallback(_read_resource_plan(selected["resource_json"]), selected["fallback"]["directory"])
    expected = _payload(comparison, items, fallback)
    if _json_value(selected) != _json_value(expected):
        raise ValueError("selection source, configuration, calibration, or frozen refit data differs")
    for value, key in ((dataset_signature, "dataset_signature"), (class_names, "class_names"), (base_identity, "base_model_identity")):
        if value is not None and selected[key] != value:
            raise ValueError(f"refit selection {key} mismatch")
    _validate_calibration(selected)
    return selected


def load_selection(path, **kwargs):
    selected = read_decision(path, **kwargs)
    if not selected["refit_required"]:
        raise ValueError("V20 candidates rejected; reuse the historical V18 fallback without refit")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--fallback-v18", type=Path, help="Frozen historical V18 full-refit delivery directory")
    parser.add_argument("--model-dir", help="Optionally verify installed official base weights")
    args = parser.parse_args()
    identity = model_identity(args.model_dir) if args.model_dir else None
    selected = make_selection(*(getattr(args, key) for key in INPUT_KEYS), identity,
        bootstrap_repeats=args.bootstrap, resource_json=args.resource_json, resource_branch=args.resource_branch,
        fallback_v18=args.fallback_v18)
    output = args.output.resolve()
    if (any(output.is_relative_to(Path(value)) for value in selected["comparison"]["inputs"].values() if value)
            or output.is_relative_to(Path(selected["fallback"]["directory"]))
            or output in (Path(selected["source_checkpoint"]), Path(selected["resource_json"]), Path(selected["audit"]["path"]))):
        raise ValueError("selection output must not overwrite source artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    read_decision(temporary, base_identity=identity)
    temporary.replace(output)
    print(json.dumps({"candidate_accepted": selected["candidate_accepted"], "refit_required": selected["refit_required"],
        "recipe": selected["recipe"], "selected_epoch": selected["selected_epoch"], "selection": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
