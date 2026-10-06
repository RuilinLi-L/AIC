"""Compare V14 robust augmentation to V13 expanded and freeze one refit contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from v14_core import file_sha256, model_identity, recipe_config
from v7_views import normalize_view_weights


CONDITIONS = ("native", "resize384", "jpeg75", "combined")


def _json_value(value):
    return json.loads(json.dumps(value))


def _validate_recipe(config, recipe):
    expected = recipe_config(recipe)
    for key, value in expected.items():
        if _json_value(config.get(key)) != _json_value(value):
            raise ValueError(f"selected training configuration differs from frozen recipe: {key}")


def read_evaluation(path):
    root = Path(path).resolve()
    summary = json.loads((root / "strict_eval.json").read_text())
    for condition in CONDITIONS:
        for metric in ("macro_accuracy", "tail_accuracy", "macro_nll"):
            if not np.isfinite(summary["conditions"][condition]["outer_metrics"][metric]):
                raise ValueError(f"nonfinite {condition}/{metric}")
    checkpoint_path = Path(summary["calibrated_checkpoint"])
    if file_sha256(checkpoint_path) != summary.get("calibrated_checkpoint_sha256"):
        raise ValueError("calibrated checkpoint changed after evaluation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("training_stage") != "validation":
        raise ValueError("selection requires held-out validation models")
    if checkpoint.get("dataset_signature") != summary["dataset_signature"]:
        raise ValueError("evaluation/checkpoint dataset mismatch")
    selected = summary["selected"]
    if checkpoint["selected_epoch"] != selected["epoch"]:
        raise ValueError("evaluation/checkpoint selected epoch mismatch")
    for key in ("view_weights", "alpha"):
        if checkpoint["calibration"][key] != selected[key]:
            raise ValueError(f"evaluation/checkpoint {key} mismatch")
    if checkpoint["calibration"]["precision"] != summary["precision"]:
        raise ValueError("evaluation/checkpoint precision mismatch")
    source = Path(summary.get("source_epoch_checkpoint", root / "validation_epochs" /
                              f"epoch_{selected['epoch']:02d}.pt"))
    if not source.is_file() or file_sha256(source) != summary["checkpoint_sha256"]:
        raise ValueError("selected epoch checkpoint changed or is missing")
    original = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint["model"].keys() != original["model"].keys() or any(
            not torch.equal(value, original["model"][name]) for name, value in checkpoint["model"].items()):
        raise ValueError("calibrated weights differ from the selected epoch")
    return {"root": str(root), "summary": summary, "checkpoint": checkpoint,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint_path)}


def choose_candidate(baseline, candidate):
    if candidate["summary"]["dataset_signature"] != baseline["summary"]["dataset_signature"]:
        raise ValueError("candidate and baseline use different validation data")
    reference = baseline["summary"]["conditions"]
    conditions = candidate["summary"]["conditions"]
    gain = lambda condition, metric: (conditions[condition]["outer_metrics"][metric]
                                     - reference[condition]["outer_metrics"][metric])
    macro_gain = gain("native", "macro_accuracy")
    tail_gain = gain("native", "tail_accuracy")
    stresses = {name: gain(name, "macro_accuracy") for name in CONDITIONS[1:]}
    accepted = (macro_gain >= .002 - 1e-12 and tail_gain >= -.003 - 1e-12
                and all(value >= -.003 - 1e-12 for value in stresses.values()))
    comparison = {"baseline": baseline["root"], "candidate": candidate["root"],
                  "eligible": accepted, "macro_gain_pp": 100 * macro_gain,
                  "tail_gain_pp": 100 * tail_gain,
                  "stress_gain_pp": {key: 100 * value for key, value in stresses.items()},
                  "required_macro_gain_pp": .2, "allowed_regression_pp": .3}
    return (candidate if accepted else baseline), comparison


def load_selection(path, *, dataset_signature=None, class_names=None, base_identity=None):
    selection = json.loads(Path(path).read_text())
    if selection.get("format_version") != 14 or selection.get("kind") != "v14_refit_selection":
        raise ValueError("not a V14 refit selection")
    recipe = selection["recipe"]
    expected = recipe_config(recipe)
    epoch = selection["selected_epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 24:
        raise ValueError("selected epoch outside frozen schedule")
    if selection["schedule_epochs"] != 24 or expected["schedule_epochs"] != 24:
        raise ValueError("refit must preserve the 24-epoch validation learning-rate schedule")
    if selection.get("augmentation") != expected["augmentation"]:
        raise ValueError("selection augmentation differs from recipe")
    _validate_recipe(selection["training_config"], recipe)
    for value, key in ((dataset_signature, "dataset_signature"), (class_names, "class_names"),
                       (base_identity, "base_model_identity")):
        if value is not None and selection[key] != value:
            raise ValueError(f"refit selection {key} mismatch")
    source = Path(selection["source_checkpoint"])
    if not source.is_file() or file_sha256(source) != selection["source_checkpoint_sha256"]:
        raise ValueError("selection source checkpoint missing or changed")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required_format = 14 if recipe == "expanded_robust" else 13
    if checkpoint.get("format_version") != required_format or checkpoint.get("training_stage") != "validation":
        raise ValueError("selection source must be the expected validation checkpoint format")
    for key in ("dataset_signature", "class_names", "selected_epoch"):
        if selection[key] != checkpoint[key]:
            raise ValueError(f"selection/source {key} mismatch")
    if _json_value(selection["training_config"]) != _json_value(checkpoint["config"]):
        raise ValueError("selection training configuration differs from frozen source")
    if selection["base_model_identity"] != checkpoint["config"]["base_model_identity"]:
        raise ValueError("selection/source base_model_identity mismatch")
    calibration = selection["calibration"]
    if calibration != checkpoint["calibration"]:
        raise ValueError("selection calibration differs from frozen source")
    weights = calibration["view_weights"]
    if (any(not np.isfinite(value) or value < 0 for value in weights.values())
            or abs(sum(weights.values()) - 1.0) > 1e-6):
        raise ValueError("invalid frozen view weights")
    normalize_view_weights(weights)
    if not np.isfinite(calibration["alpha"]) or not 0 <= calibration["alpha"] <= 1:
        raise ValueError("invalid frozen calibration strength")
    bias = np.asarray(selection["class_bias"], dtype=np.float32)
    if bias.shape != (len(selection["class_names"]),) or not np.isfinite(bias).all():
        raise ValueError("invalid frozen class bias")
    source_bias = torch.as_tensor(checkpoint["class_bias"]).float().numpy()
    if not np.array_equal(bias, source_bias):
        raise ValueError("selection class bias differs from frozen source")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline, candidate = read_evaluation(args.baseline), read_evaluation(args.candidate)
    identity = model_identity(args.model_dir)
    for item, version, recipe in ((baseline, 13, "expanded"), (candidate, 14, "expanded_robust")):
        checkpoint = item["checkpoint"]
        if (checkpoint.get("format_version") != version or checkpoint["config"].get("recipe") != recipe
                or checkpoint["config"]["base_model_identity"] != identity):
            raise ValueError("expected V13 expanded baseline and V14 expanded_robust candidate with identical base weights")
        _validate_recipe(checkpoint["config"], recipe)
        if checkpoint["class_names"] != baseline["checkpoint"]["class_names"]:
            raise ValueError("candidate and baseline classes differ")
        for key in ("precision", "stress_version", "image_size", "outer_fold_ids"):
            if item["summary"].get(key) != baseline["summary"].get(key):
                raise ValueError(f"candidate and baseline {key} differ")
    winner, comparison = choose_candidate(baseline, candidate)
    checkpoint = winner["checkpoint"]
    recipe = checkpoint["config"]["recipe"]
    result = {
        "format_version": 14, "kind": "v14_refit_selection", "recipe": recipe,
        "selected_epoch": checkpoint["selected_epoch"], "schedule_epochs": 24,
        "augmentation": recipe_config(recipe)["augmentation"],
        "dataset_signature": checkpoint["dataset_signature"], "class_names": checkpoint["class_names"],
        "base_model_identity": identity, "source_checkpoint": winner["checkpoint_path"],
        "source_checkpoint_sha256": winner["checkpoint_sha256"],
        "training_config": checkpoint["config"], "calibration": checkpoint["calibration"],
        "class_bias": torch.as_tensor(checkpoint["class_bias"]).tolist(), "comparison": comparison,
        "online_improvement_confirmed": False,
        "evaluation_scope": "noisy holdout; folds select epoch/calibration, not independent training",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    load_selection(temporary, base_identity=identity)
    temporary.replace(output)
    print(json.dumps({"winner": recipe, "selected_epoch": result["selected_epoch"],
                      "comparison": comparison, "selection": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
