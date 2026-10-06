"""Freeze a winning V19 refit or reuse the fixed V18 rank32 validation model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_v19 import INPUT_KEYS, CANDIDATE_KEYS, EXPECTED, CONDITIONS, GATES, add_arguments, choose, compare, validate_inputs
from v19_core import file_sha256, model_identity
from v7_views import normalize_view_weights


def _json_value(value):
    return json.loads(json.dumps(value))


def _artifact_hashes(root):
    root = Path(root)
    names = ["strict_eval.json", "val_labels.npy", "val_row_keys.json",
             *(f"val_oof_{condition}.npy" for condition in CONDITIONS)]
    return {name: file_sha256(root / name) for name in names}


def _read_inputs(paths, resource_json, resource_branch=None):
    return validate_inputs(*(paths[key] for key in INPUT_KEYS),
        resource_json=resource_json, resource_branch=resource_branch)[1]


def make_selection(baseline, candidate_a=None, candidate_b=None, identity=None, *, bootstrap_repeats=2000, resource_json, resource_branch=None):
    paths = {key: str(value) if value is not None else None
             for key, value in zip(INPUT_KEYS, (baseline, candidate_a, candidate_b))}
    comparison = compare(*(paths[key] for key in INPUT_KEYS), bootstrap_repeats,
                         resource_json=resource_json, resource_branch=resource_branch)
    items = _read_inputs(paths, resource_json, comparison["resource_branch"])
    winner = items[comparison["winner_key"]]
    baseline = items[comparison["baseline_key"]]
    checkpoint = winner["checkpoint"]
    actual_identity = checkpoint["config"]["base_model_identity"]
    if identity is not None and identity != actual_identity:
        raise ValueError("V19 selection official base identity differs")
    accepted = comparison["refit_required"]
    return {
        "format_version": 19, "kind": "v19_refit_selection",
        "resource_json": comparison["resource_json"], "resource_sha256": comparison["resource_sha256"],
        "resource_branch": comparison["resource_branch"], "gates": GATES,
        "audit": comparison["audit"], "candidate_statuses": comparison["candidate_statuses"],
        "recipe": checkpoint["config"]["recipe"],
        "candidate_accepted": accepted, "refit_required": accepted,
        "selected_stage": "validation" if accepted else "fallback",
        "selected_candidate": comparison["winner_key"] if accepted else None,
        "selected_evaluation": winner["root"], "source_format_version": checkpoint["format_version"],
        "selected_epoch": checkpoint["selected_epoch"], "schedule_epochs": 24,
        "augmentation": checkpoint["config"]["augmentation"],
        "dataset_signature": checkpoint["dataset_signature"], "class_names": checkpoint["class_names"],
        "base_model_identity": actual_identity,
        "source_checkpoint": winner["checkpoint_path"], "source_checkpoint_sha256": winner["checkpoint_sha256"],
        "training_config": checkpoint["config"], "calibration": checkpoint["calibration"],
        "class_bias": torch.as_tensor(checkpoint["class_bias"]).tolist(),
        "evaluation_sha256": {key: file_sha256(Path(item["root"]) / "strict_eval.json") for key, item in items.items()},
        "evaluation_artifact_sha256": {key: _artifact_hashes(item["root"]) for key, item in items.items()},
        "comparison": comparison, "baseline_recipe": baseline["checkpoint"]["config"]["recipe"],
        "fallback": {"recipe": baseline["checkpoint"]["config"]["recipe"],
                     "source_checkpoint": baseline["checkpoint_path"],
                     "source_checkpoint_sha256": baseline["checkpoint_sha256"],
                     "evaluation_directory": baseline["root"], "refit_required": False},
        "online_improvement_confirmed": False,
        "evaluation_scope": "noisy holdout; folds select epoch/calibration, not independent training",
    }


def read_decision(path, *, dataset_signature=None, class_names=None, base_identity=None):
    selected = json.loads(Path(path).read_text())
    if selected.get("format_version") != 19 or selected.get("kind") != "v19_refit_selection":
        raise ValueError("not a V19 refit selection")
    accepted = selected.get("candidate_accepted")
    if not isinstance(accepted, bool) or selected.get("refit_required") is not accepted:
        raise ValueError("V19 decision requires matching boolean acceptance and refit fields")
    if selected.get("selected_stage") != ("validation" if accepted else "fallback"):
        raise ValueError("V19 selected stage differs from refit decision")
    epoch = selected["selected_epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 24:
        raise ValueError("selected epoch outside frozen schedule")
    if selected["schedule_epochs"] != 24:
        raise ValueError("refit must preserve the 24-epoch validation learning-rate schedule")
    for value, key in ((dataset_signature, "dataset_signature"), (class_names, "class_names"),
                       (base_identity, "base_model_identity")):
        if value is not None and selected[key] != value:
            raise ValueError(f"refit selection {key} mismatch")
    source = Path(selected["source_checkpoint"])
    if not source.is_file() or file_sha256(source) != selected["source_checkpoint_sha256"]:
        raise ValueError("selection source checkpoint missing or changed")
    comparison = selected["comparison"]
    resource_path = Path(selected["resource_json"])
    if not resource_path.is_file() or file_sha256(resource_path) != selected["resource_sha256"]:
        raise ValueError("V19 frozen resource plan missing or changed")
    if selected.get("gates") != GATES or comparison.get("gates") != GATES:
        raise ValueError("V19 frozen selection gates changed")
    for field in ("resource_json", "resource_sha256", "resource_branch"):
        if comparison.get(field) != selected[field]:
            raise ValueError(f"V19 selection/comparison {field} mismatch")
    for field in ("audit", "candidate_statuses"):
        if comparison.get(field) != selected.get(field):
            raise ValueError(f"V19 selection/comparison {field} mismatch")
    active_keys = {key for key, value in comparison["inputs"].items() if value is not None}
    if (set(comparison["inputs"]) != set(INPUT_KEYS)
            or set(selected["evaluation_sha256"]) != active_keys
            or set(selected["evaluation_artifact_sha256"]) != active_keys):
        raise ValueError("V19 frozen evaluation artifacts do not match available inputs")
    for key in INPUT_KEYS:
        if comparison["inputs"][key] is None:
            continue
        summary_path = Path(comparison["inputs"][key]) / "strict_eval.json"
        if not summary_path.is_file() or file_sha256(summary_path) != selected["evaluation_sha256"][key]:
            raise ValueError("V19 frozen evaluation summary missing or changed")
        if _artifact_hashes(summary_path.parent) != selected["evaluation_artifact_sha256"][key]:
            raise ValueError("V19 frozen evaluation artifact missing or changed")
    _, items, _, binding = validate_inputs(*(comparison["inputs"][key] for key in INPUT_KEYS),
        resource_json=selected["resource_json"], resource_branch=selected["resource_branch"])
    for field in ("audit", "candidate_statuses"):
        if selected.get(field) != binding[field]:
            raise ValueError(f"V19 selection {field} differs from frozen resource plan")
    summaries = {key: item["summary"] for key, item in items.items()}
    baseline_key, eligible, winner_key = choose(summaries, selected["resource_branch"])
    expected_accepted = winner_key in CANDIDATE_KEYS
    if (accepted != expected_accepted or comparison.get("candidate_accepted") != accepted
            or comparison.get("refit_required") != accepted or comparison["winner_key"] != winner_key
            or comparison["baseline_key"] != baseline_key
            or selected["selected_candidate"] != (winner_key if accepted else None)
            or any(comparison["candidates"][key]["candidate_accepted"] != eligible[key] for key in CANDIDATE_KEYS)):
        raise ValueError("V19 decision differs from frozen evaluation gate")
    for key in CANDIDATE_KEYS:
        expected_status = ("resource_infeasible" if key not in items else
                           "eligible" if eligible[key] else "rejected")
        if comparison["candidates"][key].get("status") != expected_status:
            raise ValueError("V19 candidate status differs from frozen evaluation gate")
    winner, baseline = items[winner_key], items[baseline_key]
    checkpoint = winner["checkpoint"]
    if (selected["source_checkpoint"] != winner["checkpoint_path"]
            or selected["source_checkpoint_sha256"] != winner["checkpoint_sha256"]
            or selected["selected_evaluation"] != winner["root"]
            or comparison["winner"] != winner["root"] or comparison["baseline"] != baseline["root"]):
        raise ValueError("V19 decision differs from frozen winner source")
    for field in ("dataset_signature", "class_names", "selected_epoch"):
        if selected[field] != checkpoint[field]:
            raise ValueError(f"selection/source {field} mismatch")
    if (selected["source_format_version"] != checkpoint["format_version"]
            or selected["recipe"] != checkpoint["config"]["recipe"]
            or _json_value(selected["training_config"]) != _json_value(checkpoint["config"])
            or selected["augmentation"] != checkpoint["config"]["augmentation"]
            or selected["base_model_identity"] != checkpoint["config"]["base_model_identity"]):
        raise ValueError("selection training configuration differs from frozen source")
    calibration = selected["calibration"]
    if calibration != checkpoint["calibration"]:
        raise ValueError("selection calibration differs from frozen source")
    weights = calibration["view_weights"]
    if (any(not np.isfinite(value) or value < 0 for value in weights.values())
            or abs(sum(weights.values()) - 1) > 1e-6):
        raise ValueError("invalid frozen view weights")
    normalize_view_weights(weights)
    if not np.isfinite(calibration["alpha"]) or not 0 <= calibration["alpha"] <= 1:
        raise ValueError("invalid frozen calibration strength")
    bias = np.asarray(selected["class_bias"], dtype=np.float32)
    if (bias.shape != (len(selected["class_names"]),) or not np.isfinite(bias).all()
            or not np.array_equal(bias, torch.as_tensor(checkpoint["class_bias"]).float().numpy())):
        raise ValueError("selection class bias differs from frozen source")
    expected_fallback = {"recipe": baseline["checkpoint"]["config"]["recipe"],
                         "source_checkpoint": baseline["checkpoint_path"],
                         "source_checkpoint_sha256": baseline["checkpoint_sha256"],
                         "evaluation_directory": baseline["root"], "refit_required": False}
    if selected["fallback"] != expected_fallback or selected["baseline_recipe"] != expected_fallback["recipe"]:
        raise ValueError("V19 fallback differs from frozen baseline")
    return selected


def load_selection(path, **kwargs):
    selected = read_decision(path, **kwargs)
    if not selected["refit_required"]:
        raise ValueError("V19 candidates rejected; reuse the fallback without refit")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--model-dir", help="Optionally verify installed official base weights")
    args = parser.parse_args()
    identity = model_identity(args.model_dir) if args.model_dir else None
    selected = make_selection(*(getattr(args, key) for key in INPUT_KEYS), identity,
                              bootstrap_repeats=args.bootstrap, resource_json=args.resource_json,
                              resource_branch=args.resource_branch)
    output = args.output.resolve()
    if (any(output.is_relative_to(Path(value)) for value in selected["comparison"]["inputs"].values() if value is not None)
            or output == Path(selected["source_checkpoint"])
            or output == Path(selected["resource_json"])
            or output == Path(selected["audit"]["path"])):
        raise ValueError("selection output must not overwrite a source artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    read_decision(temporary, base_identity=identity)
    temporary.replace(output)
    print(json.dumps({"candidate_accepted": selected["candidate_accepted"],
                      "refit_required": selected["refit_required"], "recipe": selected["recipe"],
                      "selected_epoch": selected["selected_epoch"], "selection": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
