"""Freeze a winning V16 refit or reuse the stronger existing validation model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_v16 import INPUT_KEYS, CONDITIONS, add_arguments, choose, compare
from select_v14 import read_evaluation
from v15_model import validate_checkpoint_state_metadata as validate_v15_state
from v16_core import file_sha256, model_identity, recipe_config, validate_recipe_config
from v16_model import validate_checkpoint_state_metadata
from v7_views import normalize_view_weights


EXPECTED = {"baseline_expanded": (13, "expanded"), "baseline_v15": (15, "expanded_mlp"),
            "candidate_a": (16, "mlp_light"), "candidate_b": (16, "mlp_light_ln")}


def _json_value(value):
    return json.loads(json.dumps(value))


def _read_inputs(paths):
    items = {key: read_evaluation(paths[key]) for key in INPUT_KEYS}
    reference = items["baseline_expanded"]["checkpoint"]
    for key, item in items.items():
        checkpoint = item["checkpoint"]
        version, recipe = EXPECTED[key]
        if checkpoint.get("format_version") != version or checkpoint["config"].get("recipe") != recipe:
            raise ValueError(f"{key} requires format {version} recipe {recipe}")
        for field in ("dataset_signature", "class_names"):
            if checkpoint[field] != reference[field]:
                raise ValueError(f"{key} {field} differs from the baseline")
        for field in ("base_model_identity", "conflict_policy"):
            if checkpoint["config"][field] != reference["config"][field]:
                raise ValueError(f"{key} {field} differs from the baseline")
        if version == 16:
            validate_recipe_config(checkpoint["config"])
            validate_checkpoint_state_metadata(checkpoint)
        elif version == 15:
            validate_v15_state(checkpoint)
        source = Path(item["summary"]["source_epoch_checkpoint"])
        original = torch.load(source, map_location="cpu", weights_only=False)
        for field in ("format_version", "training_stage", "selected_epoch", "dataset_signature", "class_names", "config"):
            if _json_value(checkpoint[field]) != _json_value(original[field]):
                raise ValueError(f"{key} calibrated source {field} differs from the selected epoch")
    return items


def make_selection(baseline_expanded, baseline_v15, candidate_a, candidate_b, identity=None, *, bootstrap_repeats=2000):
    paths = dict(zip(INPUT_KEYS, map(str, (baseline_expanded, baseline_v15, candidate_a, candidate_b))))
    items = _read_inputs(paths)
    comparison = compare(*(paths[key] for key in INPUT_KEYS), bootstrap_repeats)
    winner = items[comparison["winner_key"]]
    baseline = items[comparison["baseline_key"]]
    checkpoint = winner["checkpoint"]
    actual_identity = checkpoint["config"]["base_model_identity"]
    if identity is not None and identity != actual_identity:
        raise ValueError("V16 selection official base identity differs")
    accepted = comparison["refit_required"]
    return {
        "format_version": 16, "kind": "v16_refit_selection",
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
    if selected.get("format_version") != 16 or selected.get("kind") != "v16_refit_selection":
        raise ValueError("not a V16 refit selection")
    accepted = selected.get("candidate_accepted")
    if not isinstance(accepted, bool) or selected.get("refit_required") is not accepted:
        raise ValueError("V16 decision requires matching boolean acceptance and refit fields")
    if selected.get("selected_stage") != ("validation" if accepted else "fallback"):
        raise ValueError("V16 selected stage differs from refit decision")
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
    for key in INPUT_KEYS:
        summary_path = Path(comparison["inputs"][key]) / "strict_eval.json"
        if not summary_path.is_file() or file_sha256(summary_path) != selected["evaluation_sha256"][key]:
            raise ValueError("V16 frozen evaluation summary missing or changed")
    items = _read_inputs(comparison["inputs"])
    summaries = {key: item["summary"] for key, item in items.items()}
    baseline_key, eligible, winner_key = choose(summaries)
    expected_accepted = winner_key in INPUT_KEYS[2:]
    if (accepted != expected_accepted or comparison.get("candidate_accepted") != accepted
            or comparison.get("refit_required") != accepted or comparison["winner_key"] != winner_key
            or comparison["baseline_key"] != baseline_key
            or selected["selected_candidate"] != (winner_key if accepted else None)
            or any(comparison["candidates"][key]["candidate_accepted"] != eligible[key] for key in INPUT_KEYS[2:])):
        raise ValueError("V16 decision differs from frozen evaluation gate")
    winner, baseline = items[winner_key], items[baseline_key]
    checkpoint = winner["checkpoint"]
    if (selected["source_checkpoint"] != winner["checkpoint_path"]
            or selected["source_checkpoint_sha256"] != winner["checkpoint_sha256"]
            or selected["selected_evaluation"] != winner["root"]
            or comparison["winner"] != winner["root"] or comparison["baseline"] != baseline["root"]):
        raise ValueError("V16 decision differs from frozen winner source")
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
        raise ValueError("V16 fallback differs from frozen baseline")
    return selected


def load_selection(path, **kwargs):
    selected = read_decision(path, **kwargs)
    if not selected["refit_required"]:
        raise ValueError("V16 candidates rejected; reuse the fallback without refit")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--model-dir", help="Optionally verify installed official base weights")
    args = parser.parse_args()
    identity = model_identity(args.model_dir) if args.model_dir else None
    selected = make_selection(*(getattr(args, key) for key in INPUT_KEYS), identity,
                              bootstrap_repeats=args.bootstrap)
    output = args.output.resolve()
    if (any(output.is_relative_to(Path(value)) for value in selected["comparison"]["inputs"].values())
            or output == Path(selected["source_checkpoint"])):
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
