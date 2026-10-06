"""Choose a qualified V13 recipe, or V12, and freeze the full-data refit contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from v13_core import RECIPES, file_sha256, model_identity, recipe_config

CONDITIONS = ("native", "resize384", "jpeg75", "combined")


def read_evaluation(path):
    root = Path(path).resolve()
    summary = json.loads((root / "strict_eval.json").read_text())
    for name in CONDITIONS:
        metrics = summary["conditions"][name]["outer_metrics"]
        for key in ("macro_accuracy", "tail_accuracy", "macro_nll"):
            if not np.isfinite(metrics[key]):
                raise ValueError(f"nonfinite {name}/{key}")
    checkpoint_path = Path(summary["calibrated_checkpoint"])
    if summary.get("calibrated_checkpoint_sha256") not in (None, file_sha256(checkpoint_path)):
        raise ValueError("calibrated checkpoint changed after evaluation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("training_stage") != "validation":
        raise ValueError("selection requires held-out validation models, never refit models")
    if checkpoint.get("dataset_signature") != summary["dataset_signature"]:
        raise ValueError("evaluation/checkpoint dataset mismatch")
    selected = summary["selected"]
    if checkpoint["selected_epoch"] != selected["epoch"]:
        raise ValueError("evaluation/checkpoint selected epoch mismatch")
    if checkpoint["calibration"]["view_weights"] != selected["view_weights"]:
        raise ValueError("evaluation/checkpoint view weights mismatch")
    if checkpoint["calibration"]["alpha"] != selected["alpha"]:
        raise ValueError("evaluation/checkpoint calibration strength mismatch")
    # Calibrated weights must be the selected epoch, without weight averaging.
    source = root / "validation_epochs" / f"epoch_{selected['epoch']:02d}.pt"
    if not source.exists() or file_sha256(source) != summary["checkpoint_sha256"]:
        raise ValueError("selected epoch checkpoint has changed or is missing")
    epoch_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint["model"].keys() != epoch_checkpoint["model"].keys() or any(
            not torch.equal(value, epoch_checkpoint["model"][name]) for name, value in checkpoint["model"].items()):
        raise ValueError("calibrated checkpoint weights differ from selected epoch")
    return {"root": str(root), "summary": summary, "checkpoint": checkpoint,
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path)}


def choose_candidate(baseline, candidates):
    """Accuracy gate is in fractions (0.005 == 0.5 percentage points)."""
    reference = baseline["summary"]["conditions"]
    accepted, comparisons = [], []
    for item in candidates:
        if item["summary"]["dataset_signature"] != baseline["summary"]["dataset_signature"]:
            raise ValueError("candidate and V12 use different validation data")
        conditions = item["summary"]["conditions"]
        native = conditions["native"]["outer_metrics"]
        macro_gain = native["macro_accuracy"] - reference["native"]["outer_metrics"]["macro_accuracy"]
        tail_gain = native["tail_accuracy"] - reference["native"]["outer_metrics"]["tail_accuracy"]
        stress_gains = {name: conditions[name]["outer_metrics"]["macro_accuracy"]
                        - reference[name]["outer_metrics"]["macro_accuracy"] for name in CONDITIONS[1:]}
        eligible = (macro_gain >= .005 - 1e-12 and tail_gain >= -.005 - 1e-12
                    and all(value >= -.005 - 1e-12 for value in stress_gains.values()))
        comparisons.append({"run": item["root"], "eligible": eligible,
                            "macro_gain_pp": 100 * macro_gain, "tail_gain_pp": 100 * tail_gain,
                            "stress_gain_pp": {k: 100 * v for k, v in stress_gains.items()}})
        if eligible:
            accepted.append(item)
    def key(item):
        native = item["summary"]["conditions"]["native"]["outer_metrics"]
        count = sum(value.numel() for value in item["checkpoint"]["model"].values())
        return (-native["macro_accuracy"], native["macro_nll"], count,
                item["checkpoint"]["config"]["recipe"])
    return (min(accepted, key=key) if accepted else baseline), comparisons


def load_selection(path, *, dataset_signature=None, class_names=None, base_identity=None):
    selection = json.loads(Path(path).read_text())
    if selection.get("format_version") != 13 or selection.get("kind") != "v13_refit_selection":
        raise ValueError("not a V13 refit selection")
    expected = recipe_config(selection["recipe"])
    if not 1 <= selection["selected_epoch"] <= expected["schedule_epochs"]:
        raise ValueError("selected epoch outside frozen schedule")
    if selection["schedule_epochs"] != expected["schedule_epochs"]:
        raise ValueError("refit must preserve the validation learning-rate schedule")
    for value, key in ((dataset_signature, "dataset_signature"), (class_names, "class_names"),
                       (base_identity, "base_model_identity")):
        if value is not None and selection[key] != value:
            raise ValueError(f"refit selection {key} mismatch")
    bias = np.asarray(selection["class_bias"])
    if bias.shape != (len(selection["class_names"]),) or not np.isfinite(bias).all():
        raise ValueError("invalid frozen class bias")
    source = Path(selection["source_checkpoint"])
    if not source.is_file() or file_sha256(source) != selection["source_checkpoint_sha256"]:
        raise ValueError("selection source checkpoint missing or changed")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidates", nargs=3, required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = read_evaluation(args.baseline)
    if baseline["checkpoint"].get("format_version") != 12:
        raise ValueError("baseline must be V12")
    candidates = [read_evaluation(p) for p in args.candidates]
    if {x["checkpoint"]["config"].get("recipe") for x in candidates} != set(RECIPES):
        raise ValueError("provide all three distinct V13 recipes")
    identity = model_identity(args.model_dir)
    for item in candidates:
        if (item["checkpoint"].get("format_version") != 13
                or item["checkpoint"]["config"]["base_model_identity"] != identity
                or item["checkpoint"]["class_names"] != baseline["checkpoint"]["class_names"]
                or item["summary"]["precision"] != baseline["summary"]["precision"]):
            raise ValueError("candidate base weights, classes, format or precision differ")
    winner, comparisons = choose_candidate(baseline, candidates)
    checkpoint = winner["checkpoint"]
    recipe = checkpoint["config"].get("recipe", "v12_fallback")
    result = {
        "format_version": 13, "kind": "v13_refit_selection", "recipe": recipe,
        "selected_epoch": checkpoint["selected_epoch"],
        "schedule_epochs": recipe_config(recipe)["schedule_epochs"],
        "dataset_signature": checkpoint["dataset_signature"], "class_names": checkpoint["class_names"],
        "base_model_identity": identity, "source_checkpoint": winner["checkpoint_path"],
        "source_checkpoint_sha256": winner["checkpoint_sha256"],
        "training_config": checkpoint["config"], "calibration": checkpoint["calibration"],
        "class_bias": checkpoint["class_bias"].tolist(), "comparisons": comparisons,
        "baseline_online_score": 58.8292, "online_improvement_confirmed": False,
        "evaluation_scope": "noisy holdout; folds select epoch/calibration, not independent training",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"winner": recipe, "selected_epoch": result["selected_epoch"],
                      "comparisons": comparisons, "selection": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
