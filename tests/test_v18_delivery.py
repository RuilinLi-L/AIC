"""V18 resource-bound gates, immutable selections and compatible delivery."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import compare_v18 as comparison
import evaluate_tta_v18 as evaluation
import predict_v18 as prediction
import select_v18 as selection
from test_v15_delivery import BASE_IDENTITY, TinyPredictor, checkpoint as old_checkpoint
from test_v17_delivery import checkpoint as v17_checkpoint
from v18_core import recipe_config, file_sha256
from v18_model import _official_trainable_shapes
from v18_runtime import scientific_config
from v18_views import resolution_processor
from types import SimpleNamespace


def checkpoint(recipe="resolution384", epoch=18, *, branch="standard", resource=None):
    value = old_checkpoint(15, epoch)
    value["format_version"] = 18
    value["config"].update(recipe_config(recipe))
    value["config"].update(resource_branch=branch, resource_json=str(resource or "/frozen/resource.json"),
        resource_sha256=file_sha256(resource) if resource else "resource-hash",
        batch_size=128 if branch == "matched_control" else 256,
        gradient_accumulation=2 if branch == "matched_control" else 1)
    shapes = {**_official_trainable_shapes(2, value["config"]), "logit_scale": ()}
    zero = torch.zeros(())
    value["model"] = {name: zero.expand(shape) for name, shape in shapes.items()}
    value["trainable_parameter_names"] = sorted(key for key in shapes if key != "logit_scale")
    value["calibration"].update(selected_epoch=epoch,
        **{field: value["config"][field] for field in ("image_size", "zoom_shortest_edge", "lora_rank")})
    return value


def fixture(root, key, *, branch="standard", resource=None, correct=350, stress_correct=350, tail=.6, nll=1.2, same_run=False):
    version, recipe = comparison.expected_for_branch(branch)[key]
    ckpt = checkpoint(recipe, branch=branch, resource=resource) if version == 18 else old_checkpoint(version)
    run = root / key; run.mkdir()
    source = run / "epoch_18.pt"; torch.save(ckpt, source)
    out = run if same_run else run / "evaluation"
    out.mkdir(exist_ok=True)
    target = out / "model.pt"; torch.save(ckpt, target)
    labels = np.repeat(np.arange(2), 500)
    np.save(out / "val_labels.npy", labels)
    (out / "val_row_keys.json").write_text(json.dumps([f"row-{i}" for i in range(1000)]))
    conditions = {}
    for condition in comparison.CONDITIONS:
        count = correct if condition == "native" else stress_correct
        pred = 1 - labels
        for label in (0, 1): pred[label * 500:label * 500 + count] = label
        logits = np.zeros((len(labels), 2), dtype=np.float32)
        logits[np.arange(len(labels)), pred] = 4
        np.save(out / f"val_oof_{condition}.npy", logits)
        conditions[condition] = {"outer_metrics": {"macro_accuracy": count / 500,
            "micro_accuracy": count / 500, "tail_accuracy": tail, "macro_nll": nll}}
    summary = {"format_version": version, "recipe": recipe, "dataset_signature": "dataset",
        "calibrated_checkpoint": str(target), "calibrated_checkpoint_sha256": file_sha256(target),
        "checkpoint_sha256": file_sha256(source), "source_epoch_checkpoint": str(source),
        "selected": {"epoch": 18, "view_weights": ckpt["calibration"]["view_weights"], "alpha": 0.},
        "precision": "fp32", "stress_version": evaluation.STRESS_VERSION,
        "image_size": ckpt["config"]["image_size"],
        "outer_fold_ids": (np.arange(1000) % 5).tolist(), "conditions": conditions}
    if version == 18:
        summary.update({field: ckpt["config"][field] for field in
            ("zoom_shortest_edge", "lora_rank", "resource_json", "resource_sha256", "resource_branch")})
        summary.update(scientific_config=scientific_config(ckpt["config"]),
                       preprocessing_version=evaluation.RESOLUTION_VERSION)
    (out / "strict_eval.json").write_text(json.dumps(summary))
    return out


def inputs(root, branch="standard", **options):
    baseline = fixture(root, "baseline_v15", **options.get("baseline_v15", {}))
    plan = {"format_version": 18, "kind": "v18_resource_plan", "branch": branch,
        "activation_checkpointing": False, "gates": comparison.GATES,
        "dataset_signature": "dataset", "base_model_identity": BASE_IDENTITY,
        "baseline": {"directory": str(baseline), "model_sha256": file_sha256(baseline / "model.pt"),
                     "evaluation_sha256": file_sha256(baseline / "strict_eval.json")}, "candidates": {}}
    for key in comparison.CANDIDATE_KEYS:
        spec = recipe_config(comparison.expected_for_branch(branch)[key][1])
        plan["candidates"][key] = {"run_dir": str(root / key), "recipe": spec["recipe"],
            "batch_size": 128 if branch == "matched_control" else 256,
            "gradient_accumulation": 2 if branch == "matched_control" else 1,
            "workers": 8, "image_size": spec["image_size"], "lora_rank": spec["lora_rank"]}
    resource = root / "resource.json"; resource.write_text(json.dumps(plan))
    candidates = [fixture(root, key, branch=branch, resource=resource, **options.get(key, {}))
                  for key in comparison.CANDIDATE_KEYS]
    return [baseline, *candidates], resource


class V18DeliveryTests(unittest.TestCase):
    def setUp(self):
        # Engineering probe validation is exercised by pipeline tests; here real
        # file hashes/configuration binding stay active while small fixtures skip probes.
        self.reader = patch.object(comparison, "_read_resource_plan", side_effect=lambda p: json.loads(Path(p).read_text()))
        self.reader.start(); self.addCleanup(self.reader.stop)
        self.metrics = patch.object(comparison, "FROZEN_V15_NATIVE", {})
        self.metrics.start(); self.addCleanup(self.metrics.stop)

    def test_historical_v15_numeric_anchor_is_checked_separately_from_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, resource = inputs(Path(directory))
            with patch.object(comparison, "FROZEN_V15_NATIVE", {"macro_accuracy": .7546586990356445}), \
                    self.assertRaisesRegex(ValueError, "historical evaluation"):
                comparison.compare(*paths, 5, resource_json=resource)

    def test_gate_boundaries_micro_is_unrounded_and_ties_are_deterministic(self):
        summaries = {key: {"conditions": {"native": {"outer_metrics": {
            "macro_accuracy": .7 if key == "baseline_v15" else .702,
            "micro_accuracy": .75, "tail_accuracy": .6 if key == "baseline_v15" else .597,
            "macro_nll": 1.2}}}} for key in comparison.INPUT_KEYS}
        self.assertEqual(comparison.choose(summaries)[2], "candidate_a")
        a = summaries["candidate_a"]["conditions"]["native"]["outer_metrics"]
        b = summaries["candidate_b"]["conditions"]["native"]["outer_metrics"]
        a["micro_accuracy"] = np.nextafter(.75, 0.)
        self.assertEqual(comparison.choose(summaries)[2], "candidate_b")
        b["tail_accuracy"] = .5969
        self.assertEqual(comparison.choose(summaries)[2], "baseline_v15")
        a["micro_accuracy"] = b["micro_accuracy"] = .75; b["tail_accuracy"] = .597
        for field, value in (("macro_nll", 1.1), ("tail_accuracy", .61), ("macro_accuracy", .72)):
            b[field] = value
            self.assertEqual(comparison.choose(summaries)[2], "candidate_b")

    def test_different_model_sizes_compare_without_relaxing_gate_or_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, resource = inputs(Path(directory), baseline_v15={"correct": 352},
                candidate_a={"correct": 353, "tail": .597, "stress_correct": 200},
                candidate_b={"correct": 353, "tail": .5969})
            actual = comparison.compare(*paths, 10, resource_json=resource)
            self.assertEqual(actual["winner_key"], "candidate_a")
            self.assertAlmostEqual(actual["candidates"]["candidate_a"]["macro_gain_pp"], .2)
            self.assertEqual(actual["stress_role"], "diagnostic_only_not_a_selection_gate")
            self.assertEqual(actual["bootstrap_role"], "diagnostic_only_not_a_selection_gate")
            p = paths[1] / "strict_eval.json"; original = json.loads(p.read_text())
            for field, value in (("image_size", 320), ("lora_rank", 32), ("zoom_shortest_edge", 366),
                                 ("resource_branch", "matched_control"), ("outer_fold_ids", [0]),
                                 ("stress_version", "other"), ("precision", "bf16")):
                p.write_text(json.dumps({**original, field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    comparison.compare(*paths, 10, resource_json=resource)
            p.write_text(json.dumps(original))
            rows = paths[1] / "val_row_keys.json"
            rows.write_text(json.dumps(list(reversed(json.loads(rows.read_text())))))
            with self.assertRaisesRegex(ValueError, "rows/labels"):
                comparison.compare(*paths, 10, resource_json=resource)

    def test_pipeline_evaluation_can_be_saved_directly_in_training_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, resource = inputs(Path(directory), candidate_a={"correct": 355, "same_run": True},
                                     candidate_b={"same_run": True})
            result = comparison.compare(*paths, 5, resource_json=resource)
            self.assertEqual(result["winner_key"], "candidate_a")
            self.assertEqual(result["class_validation_support"], [500, 500])
            self.assertEqual(result["per_class_accuracy"]["candidate_a"]["native"], [.71, .71])

    def test_matched_control_is_never_winner_and_a_must_pass_both_references(self):
        for a, b, winner in ((355, 354, "candidate_a"), (355, 355, "baseline_v15"),
                              (351, 340, "candidate_a"), (350, 340, "baseline_v15"),
                              (355, 356, "baseline_v15")):
            with self.subTest(a=a, b=b), tempfile.TemporaryDirectory() as directory:
                paths, resource = inputs(Path(directory), "matched_control",
                    candidate_a={"correct": a}, candidate_b={"correct": b})
                result = comparison.compare(*paths, 5, resource_json=resource)
                self.assertEqual(result["winner_key"], winner)
                self.assertFalse(result["candidates"]["candidate_b"]["candidate_accepted"])
                self.assertEqual(result["candidates"]["candidate_b"]["status"], "control_only")
                self.assertIsNotNone(result["matched_control_comparison"])

    def test_selection_winners_fallback_frozen_sources_and_artifacts(self):
        for branch, winner in (("standard", "candidate_a"), ("standard", "candidate_b"),
                               ("standard", "baseline_v15"), ("matched_control", "candidate_a")):
            with self.subTest(branch=branch, winner=winner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); paths, resource = inputs(root, branch, **{winner: {"correct": 355}})
                value = selection.make_selection(*paths, resource_json=resource, bootstrap_repeats=5)
                output = root / "selection.json"; output.write_text(json.dumps(value))
                self.assertEqual(selection.read_decision(output)["comparison"]["winner_key"], winner)
                self.assertEqual(value["resource_sha256"], file_sha256(resource))
                if winner == "baseline_v15":
                    with self.assertRaisesRegex(ValueError, "without refit"): selection.load_selection(output)
                else:
                    self.assertEqual(selection.load_selection(output, base_identity=BASE_IDENTITY)["recipe"],
                                     comparison.expected_for_branch(branch)[winner][1])
                for field in ("resource_branch", "calibration", "config", "bias", "gate", "source", "winner"):
                    bad = deepcopy(value)
                    if field == "resource_branch": bad[field] = "standard" if branch != "standard" else "matched_control"
                    elif field == "calibration": bad[field]["alpha"] = .5
                    elif field == "config": bad["training_config"]["batch_size"] = 64
                    elif field == "bias": bad["class_bias"][0] = 1.
                    elif field == "gate": bad["gates"]["overall_noninferiority"] = False
                    elif field == "source": bad["source_checkpoint_sha256"] = "changed"
                    else: bad["comparison"]["winner_key"] = "candidate_b" if winner != "candidate_b" else "candidate_a"
                    output.write_text(json.dumps(bad))
                    with self.subTest(field=field), self.assertRaises(ValueError): selection.read_decision(output)
                output.write_text(json.dumps(value))
                logits = paths[1] / "val_oof_native.npy"; saved = np.load(logits); saved[0] += 1
                np.save(logits, saved)
                with self.assertRaisesRegex(ValueError, "evaluation artifact"): selection.read_decision(output)

    def test_resource_file_hash_and_actual_microbatch_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths, resource = inputs(root)
            with self.assertRaisesRegex(ValueError, "branch"):
                comparison.compare(*paths, 5, resource_json=resource, resource_branch="matched_control")
            original = resource.read_text(); changed = json.loads(original)
            changed["candidates"]["candidate_a"]["batch_size"] = 128
            resource.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "batch_size"):
                comparison.compare(*paths, 5, resource_json=resource)
            resource.write_text(original + " ")
            with self.assertRaisesRegex(ValueError, "provenance"):
                comparison.compare(*paths, 5, resource_json=resource)

    def test_selection_cli_accepts_resource_and_baseline_alias(self):
        for option in ("--baseline", "--baseline-v15"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); paths, resource = inputs(root, candidate_a={"correct": 355})
                output = root / "selection.json"
                argv = ["select_v18.py", option, str(paths[0]), "--candidate-a", str(paths[1]),
                    "--candidate-b", str(paths[2]), "--resource-json", str(resource), "--output", str(output), "--bootstrap", "5"]
                with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()): selection.main()
                self.assertEqual(selection.read_decision(output)["recipe"], "resolution384")

    def test_prediction_rejects_calibration_geometry_or_rank_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "model.pt"
            argv = ["predict_v18.py", "--checkpoint", str(source), "--test-dir", str(root),
                    "--output", str(root / "pred.csv"), "--zip-output", str(root / "pred.zip"), "--device", "cpu"]
            for field, value in (("view_weights", {"center": 2.}), ("view_weights", {"center": float("nan")}),
                                 ("alpha", 1.5), ("selected_epoch", 17), ("image_size", 320),
                                 ("zoom_shortest_edge", 366), ("lora_rank", 32), ("precision", "bf16")):
                current = checkpoint(); current["calibration"][field] = value; torch.save(current, source)
                with self.subTest(field=field), patch.object(sys, "argv", argv), \
                        patch.object(prediction, "build_classifier_from_checkpoint") as build, self.assertRaises(ValueError):
                    prediction.main()
                build.assert_not_called()

    def test_prediction_new_and_legacy_models_validate_zip_and_actual_shape(self):
        processor = SimpleNamespace(image_processor=SimpleNamespace(size={"shortest_edge": 224},
            crop_size={"height": 224, "width": 224}, image_mean=(.5, .5, .5), image_std=(.25, .25, .25)))
        seen = []
        class ShapePredictor(TinyPredictor):
            def forward(self, pixels, _=None):
                seen.append(tuple(pixels.shape[-2:])); return super().forward(pixels)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); images = root / "images"; images.mkdir()
            for index in range(3): Image.new("RGB", (360, 420), (index * 70, 90, 120)).save(images / f"{index}.png")
            for name, value in (("a", checkpoint()), ("b", checkpoint("rank32")),
                                ("old15", old_checkpoint(15)), ("old17", v17_checkpoint())):
                source = root / f"{name}.pt"; torch.save(value, source); out = root / name
                prep = resolution_processor(processor, value["config"]["image_size"])
                argv = ["predict_v18.py", "--checkpoint", str(source), "--model-dir", "official",
                    "--test-dir", str(images), "--output", str(out / "pred_results.csv"),
                    "--zip-output", str(out / "pred_results.zip"), "--device", "cpu",
                    "--expected-rows", "3", "--workers", "0", "--batch-size", "2"]
                seen.clear()
                with patch.object(sys, "argv", argv), patch.object(prediction, "build_classifier_from_checkpoint",
                        return_value=(ShapePredictor(), prep)), redirect_stdout(io.StringIO()): prediction.main()
                self.assertEqual(set(seen), {(value["config"]["image_size"],) * 2})
                prediction.validate_submission(out / "pred_results.csv", out / "pred_results.zip", 3, [f"{n}.png" for n in range(3)])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
