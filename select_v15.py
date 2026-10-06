"""Freeze the sole V15 candidate against the already selected V14 baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_v15 import compare
from select_v14 import load_selection as load_v14_selection, read_evaluation
from v15_core import file_sha256, model_identity, recipe_config
from v15_model import validate_checkpoint_state_metadata
from v7_views import normalize_view_weights


CONDITIONS = ("native", "resize384", "jpeg75", "combined")


def _json_value(value):
    return json.loads(json.dumps(value))


def _validate_recipe(config):
    for key, value in recipe_config("expanded_mlp").items():
        if _json_value(config.get(key)) != _json_value(value):
            raise ValueError(f"selected training configuration differs from V15 recipe: {key}")


def read_baseline(selection_path, *, base_identity=None):
    """Resolve the baseline from V14's decision, never from the newest run."""
    selected = load_v14_selection(selection_path, base_identity=base_identity)
    key = "candidate" if selected["recipe"] == "expanded_robust" else "baseline"
    baseline = read_evaluation(selected["comparison"][key])
    if (Path(baseline["checkpoint_path"]).resolve() != Path(selected["source_checkpoint"]).resolve()
            or baseline["checkpoint_sha256"] != selected["source_checkpoint_sha256"]):
        raise ValueError("V14 baseline evaluation differs from the frozen selection source")
    return selected, baseline


def read_decision(path, *, dataset_signature=None, class_names=None, base_identity=None):
    selected = json.loads(Path(path).read_text())
    if selected.get("format_version") != 15 or selected.get("kind") != "v15_refit_selection":
        raise ValueError("not a V15 refit selection")
    if not isinstance(selected.get("candidate_accepted"), bool):
        raise ValueError("V15 decision must record a boolean candidate_accepted")
    if selected.get("recipe") != "expanded_mlp":
        raise ValueError("V15 refit requires the expanded_mlp recipe")
    epoch = selected["selected_epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 24:
        raise ValueError("selected epoch outside frozen schedule")
    if selected["schedule_epochs"] != 24:
        raise ValueError("refit must preserve the 24-epoch validation learning-rate schedule")
    _validate_recipe(selected["training_config"])
    if selected["augmentation"] != recipe_config("expanded_mlp")["augmentation"]:
        raise ValueError("selection augmentation differs from recipe")
    for value, key in ((dataset_signature, "dataset_signature"), (class_names, "class_names"),
                       (base_identity, "base_model_identity")):
        if value is not None and selected[key] != value:
            raise ValueError(f"refit selection {key} mismatch")
    baseline_path = Path(selected["baseline_selection"])
    if not baseline_path.is_file() or file_sha256(baseline_path) != selected["baseline_selection_sha256"]:
        raise ValueError("frozen V14 baseline selection missing or changed")
    source = Path(selected["source_checkpoint"])
    if not source.is_file() or file_sha256(source) != selected["source_checkpoint_sha256"]:
        raise ValueError("selection source checkpoint missing or changed")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 15 or checkpoint.get("training_stage") != "validation":
        raise ValueError("V15 selection source must be a V15 validation checkpoint")
    validate_checkpoint_state_metadata(checkpoint)
    for key in ("dataset_signature", "class_names", "selected_epoch"):
        if selected[key] != checkpoint[key]:
            raise ValueError(f"selection/source {key} mismatch")
    if _json_value(selected["training_config"]) != _json_value(checkpoint["config"]):
        raise ValueError("selection training configuration differs from frozen source")
    if selected["base_model_identity"] != checkpoint["config"]["base_model_identity"]:
        raise ValueError("selection/source base_model_identity mismatch")
    calibration = selected["calibration"]
    if calibration != checkpoint["calibration"]:
        raise ValueError("selection calibration differs from frozen source")
    weights = calibration["view_weights"]
    if (any(not np.isfinite(value) or value < 0 for value in weights.values())
            or abs(sum(weights.values()) - 1.) > 1e-6):
        raise ValueError("invalid frozen view weights")
    normalize_view_weights(weights)
    if not np.isfinite(calibration["alpha"]) or not 0 <= calibration["alpha"] <= 1:
        raise ValueError("invalid frozen calibration strength")
    bias = np.asarray(selected["class_bias"], dtype=np.float32)
    if bias.shape != (len(selected["class_names"]),) or not np.isfinite(bias).all():
        raise ValueError("invalid frozen class bias")
    if not np.array_equal(bias, torch.as_tensor(checkpoint["class_bias"]).float().numpy()):
        raise ValueError("selection class bias differs from frozen source")
    baseline_selection, baseline = read_baseline(baseline_path, base_identity=selected["base_model_identity"])
    fallback = selected["fallback"]
    if (fallback["source_checkpoint"] != baseline["checkpoint_path"]
            or fallback["source_checkpoint_sha256"] != baseline["checkpoint_sha256"]
            or selected["comparison"]["baseline"] != baseline["root"]
            or fallback["recipe"] != baseline_selection["recipe"]
            or selected["baseline_recipe"] != baseline_selection["recipe"]):
        raise ValueError("V15 fallback differs from the frozen V14 source")
    candidate = read_evaluation(selected["comparison"]["candidate"])
    if (candidate["checkpoint_path"] != str(source.resolve())
            or candidate["checkpoint_sha256"] != selected["source_checkpoint_sha256"]):
        raise ValueError("V15 decision evaluation differs from frozen candidate source")
    for item, key in ((baseline, "baseline_evaluation_sha256"), (candidate, "candidate_evaluation_sha256")):
        if file_sha256(Path(item["root"]) / "strict_eval.json") != selected[key]:
            raise ValueError("V15 decision evaluation summary changed after selection")
    delta = lambda condition, metric: (candidate["summary"]["conditions"][condition]["outer_metrics"][metric]
                                       - baseline["summary"]["conditions"][condition]["outer_metrics"][metric])
    accepted = (delta("native", "macro_accuracy") >= .002 - 1e-12
                and delta("native", "tail_accuracy") >= -.003 - 1e-12
                and all(delta(condition, "macro_accuracy") >= -.003 - 1e-12 for condition in CONDITIONS[1:]))
    if selected["candidate_accepted"] != accepted or selected["comparison"]["candidate_accepted"] != accepted:
        raise ValueError("V15 candidate decision differs from the frozen evaluation gate")
    return selected


def load_selection(path, **kwargs):
    selected = read_decision(path, **kwargs)
    if selected["candidate_accepted"] is not True:
        raise ValueError("V15 candidate was rejected; use the frozen V14 fallback without V15 refit")
    return selected


def make_selection(baseline_selection_path, candidate_dir, identity, *, bootstrap_repeats=2000):
    baseline_selected, baseline = read_baseline(baseline_selection_path, base_identity=identity)
    candidate = read_evaluation(candidate_dir)
    checkpoint = candidate["checkpoint"]
    if checkpoint.get("format_version") != 15:
        raise ValueError("V15 selection requires a format-15 candidate")
    validate_checkpoint_state_metadata(checkpoint)
    _validate_recipe(checkpoint["config"])
    if checkpoint["config"]["base_model_identity"] != identity:
        raise ValueError("V15 candidate and baseline must use identical official base weights")
    if checkpoint["class_names"] != baseline["checkpoint"]["class_names"]:
        raise ValueError("V15 candidate and baseline classes differ")
    if checkpoint["config"]["conflict_policy"] != baseline["checkpoint"]["config"]["conflict_policy"]:
        raise ValueError("V15 candidate and baseline conflict policy differs")
    comparison = compare(Path(baseline["root"]), Path(candidate["root"]), bootstrap_repeats)
    selected = {
        "format_version": 15, "kind": "v15_refit_selection", "recipe": "expanded_mlp",
        "candidate_accepted": comparison["candidate_accepted"],
        "selected_epoch": checkpoint["selected_epoch"], "schedule_epochs": 24,
        "augmentation": recipe_config("expanded_mlp")["augmentation"],
        "dataset_signature": checkpoint["dataset_signature"], "class_names": checkpoint["class_names"],
        "base_model_identity": identity, "source_checkpoint": candidate["checkpoint_path"],
        "source_checkpoint_sha256": candidate["checkpoint_sha256"],
        "training_config": checkpoint["config"], "calibration": checkpoint["calibration"],
        "class_bias": torch.as_tensor(checkpoint["class_bias"]).tolist(),
        "baseline_selection": str(Path(baseline_selection_path).resolve()),
        "baseline_selection_sha256": file_sha256(baseline_selection_path),
        "baseline_recipe": baseline_selected["recipe"],
        "baseline_evaluation_sha256": file_sha256(Path(baseline["root"]) / "strict_eval.json"),
        "candidate_evaluation_sha256": file_sha256(Path(candidate["root"]) / "strict_eval.json"),
        "comparison": comparison,
        "fallback": {"recipe": baseline_selected["recipe"],
                     "source_checkpoint": baseline["checkpoint_path"],
                     "source_checkpoint_sha256": baseline["checkpoint_sha256"],
                     "evaluation_directory": baseline["root"], "refit_required": False},
        "online_improvement_confirmed": False,
        "evaluation_scope": "noisy holdout; folds select epoch/calibration, not independent training",
    }
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-selection", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    identity = model_identity(args.model_dir)
    selected = make_selection(args.baseline_selection, args.candidate, identity,
                              bootstrap_repeats=args.bootstrap)
    output = Path(args.output)
    source_paths = (Path(args.baseline_selection), Path(selected["source_checkpoint"]),
                    Path(selected["fallback"]["source_checkpoint"]))
    if any(output.resolve() == path.resolve() for path in source_paths):
        raise ValueError("selection output must not overwrite a source artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    read_decision(temporary, base_identity=identity)
    temporary.replace(output)
    print(json.dumps({"candidate_accepted": selected["candidate_accepted"],
                      "recipe": selected["recipe"], "selected_epoch": selected["selected_epoch"],
                      "comparison": selected["comparison"],
                      "selection": str(output.resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
