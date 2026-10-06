"""Frozen V20 gates, common-protocol provenance, and V18 package fallback."""
from copy import deepcopy
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import compare_v20 as comparison
import select_v20 as selection
import predict_v20 as prediction
from test_v15_delivery import BASE_IDENTITY
from test_v18_delivery import checkpoint as v18_checkpoint
from v19_core import file_sha256
from v20_evaluation import METHOD, PROTOCOL
from v10_evaluation import _metrics


def binding(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def checkpoint(key, resource, audit, *, epoch=13):
    version, recipe = comparison.EXPECTED[key]
    value = v18_checkpoint("rank32", epoch)
    value["format_version"] = version
    value["config"].update(import_module(f"v{version}_core").recipe_config(recipe))
    value["config"].update(resource_json=str(resource), resource_sha256=file_sha256(resource),
        resource_branch="deduplicated_control", audit_json=str(audit), audit_sha256=file_sha256(audit),
        stage="validate", epochs=24, schedule_epochs=24, workers=8)
    shapes = import_module(f"v{version}_model")._official_trainable_shapes(2, value["config"])
    value["trainable_parameter_names"] = sorted(shapes)
    value["model"] = {name: torch.zeros(()).expand(shape) for name, shape in {**shapes, "logit_scale": ()}.items()}
    value["calibration"].update(method=METHOD, evaluation_protocol=PROTOCOL, selected_epoch=epoch,
        **{field: value["config"][field] for field in comparison.GEOMETRY_FIELDS})
    return value


def make_fixture(root, *, absent=(), winner="v20_loraplus"):
    fallback = root / "historical_refit"; fallback.mkdir()
    historical = v18_checkpoint("rank32", 18)
    historical["training_stage"] = "refit"; historical["config"]["stage"] = "refit"
    historical["dataset_signature"] = "historical_pre_dedup_data"
    historical["calibration"]["frozen_before_refit"] = True
    historical_path = fallback / "model.pt"; torch.save(historical, historical_path)
    (fallback / "pred_results.csv").write_text("one.jpg, 0000\n")
    (fallback / "pred_results.zip").write_bytes(b"frozen-existing-zip")
    provenance = {"checkpoint": str(historical_path), "checkpoint_sha256": file_sha256(historical_path),
                  "csv_sha256": file_sha256(fallback / "pred_results.csv"), "zip_sha256": file_sha256(fallback / "pred_results.zip")}
    (fallback / "provenance.json").write_text(json.dumps(provenance))
    historical_binding = {"directory": str(fallback), "provenance_sha256": file_sha256(fallback / "provenance.json"),
                          **{key: value for key, value in provenance.items() if key.endswith("sha256")}}
    audit = root / "audit.json"; audit.write_text(json.dumps({"status": "passed", "dataset_signature": "dataset"}))
    entries = {}
    for key in comparison.INPUT_KEYS:
        (root / key).mkdir()
        entries[key] = {"recipe": comparison.EXPECTED[key][1], "run_dir": str(root / key),
                        "status": "resource_infeasible" if key in absent else "runnable",
                        "batch_size": 256, "gradient_accumulation": 1, "workers": 8}
    old = {"format_version": 19, "branch": "deduplicated_control", "dataset_signature": "dataset",
        "control": entries["control"], "candidates": {"candidate_a": entries["v19_resolution384"], "candidate_b": entries["v19_dropout"]}}
    oldpath = root / "v19_resource.json"; oldpath.write_text(json.dumps(old))
    plan = {"format_version": 20, "kind": "v20_resource_plan", "branch": "deduplicated_control",
        "gates": comparison.GATES, "dataset_signature": "dataset", "base_model_identity": BASE_IDENTITY,
        "audit": binding(audit), "v19_resource": binding(oldpath), "control": entries["control"],
        "dependencies": {entries[key]["recipe"]: entries[key] for key in comparison.CANDIDATE_KEYS[:2]},
        "candidates": {"candidate_a": entries["v20_dora"], "candidate_b": entries["v20_loraplus"]},
        "historical_refit": historical_binding}
    resource = root / "resource.json"; resource.write_text(json.dumps(plan))
    paths = []
    for key in comparison.INPUT_KEYS:
        if key in absent:
            paths.append(None); continue
        version, recipe = comparison.EXPECTED[key]
        source_resource = oldpath if version == 19 else resource
        value = checkpoint(key, source_resource, audit)
        epoch_dir = root / key / "validation_epochs"; epoch_dir.mkdir()
        hashes = {}
        for epoch in range(1, 25):
            source = epoch_dir / f"epoch_{epoch:02d}.pt"
            value["selected_epoch"] = epoch
            value["calibration"]["selected_epoch"] = epoch
            torch.save(value, source); hashes[str(epoch)] = file_sha256(source)
        value["selected_epoch"] = value["calibration"]["selected_epoch"] = 13
        out = root / key / "joint_evaluation"; out.mkdir()
        calibrated = out / "model.pt"; torch.save(value, calibrated)
        labels = np.repeat(np.arange(2), 50)
        rows = [f"row-{i}" for i in range(100)]
        np.save(out / "val_labels.npy", labels)
        (out / "val_row_keys.json").write_text(json.dumps(rows))
        count = 36 if key == winner else 35
        prediction_labels = 1 - labels
        for label in (0, 1): prediction_labels[label * 50:label * 50 + count] = label
        logits = np.zeros((100, 2), dtype=np.float32)
        logits[np.arange(100), prediction_labels] = 4
        np.save(out / "val_selected_native.npy", logits)
        conditions = {}
        for condition in comparison.CONDITIONS:
            np.save(out / f"val_oof_{condition}.npy", logits)
            conditions[condition] = {"outer_metrics": _metrics(logits, labels, np.array([100, 100]))}
        config = value["config"]
        summary = {"format_version": version, "source_format_version": version, "evaluation_format_version": 20,
            "method": METHOD, "evaluation_protocol": PROTOCOL, "recipe": recipe, "source_run_dir": str(root / key),
            "dataset_signature": value["dataset_signature"], "class_names": value["class_names"],
            "base_model_identity": BASE_IDENTITY, "conditions": conditions, "class_training_support": [100, 100],
            "calibrated_checkpoint": str(calibrated), "calibrated_checkpoint_sha256": file_sha256(calibrated),
            "source_epoch_checkpoint": str(epoch_dir / "epoch_13.pt"), "checkpoint_sha256": hashes["13"],
            "epoch_checkpoint_sha256": hashes, "validation_row_keys_sha256": hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
            "outer_fold_ids": (np.arange(100) % 5).tolist(), "precision": "fp32", "stress_version": "stress",
            "selected": {"epoch": 13, "view_weights": value["calibration"]["view_weights"], "alpha": 0.},
            "scientific_config": import_module(f"v{version}_runtime").scientific_config(config),
            "preprocessing_version": import_module(f"v{version}_model").RESOLUTION_VERSION,
            **{field: config[field] for field in (*comparison.GEOMETRY_FIELDS, "resource_json", "resource_sha256", "resource_branch", "audit_json", "audit_sha256")}}
        (out / "strict_eval.json").write_text(json.dumps(summary))
        paths.append(out)
    return paths, resource, fallback


class V20SelectionTests(unittest.TestCase):
    def setUp(self):
        # Typed OOM, audit and source-bundle validation are covered by the resource
        # tests; this suite keeps real checkpoint, evaluation and package hashes.
        for module in (comparison, selection):
            reader = patch.object(module, "_read_resource_plan", side_effect=lambda path: json.loads(Path(path).read_text()))
            reader.start(); self.addCleanup(reader.stop)

    def test_gate_boundaries_unrounded_micro_and_all_tie_breaks(self):
        summaries = {key: {"conditions": {"native": {"outer_metrics": {
            "macro_accuracy": .7 if key == "control" else .702,
            "tail_accuracy": .6 if key == "control" else .597,
            "micro_accuracy": .75, "macro_nll": 1.2}}}} for key in comparison.INPUT_KEYS}
        for expected in comparison.TIE_ORDER:
            self.assertEqual(comparison.choose(summaries)[2], expected)
            summaries.pop(expected)
        self.assertEqual(comparison.choose(summaries)[2], "fallback_v18")
        candidate = {"conditions": {"native": {"outer_metrics": {
            "macro_accuracy": .702, "tail_accuracy": .597, "micro_accuracy": np.nextafter(.75, 0), "macro_nll": 1.2}}}}
        summaries["v20_dora"] = candidate
        self.assertFalse(comparison.choose(summaries)[1]["v20_dora"])
        candidate["conditions"]["native"]["outer_metrics"].update(micro_accuracy=.75, tail_accuracy=.5969)
        self.assertFalse(comparison.choose(summaries)[1]["v20_dora"])

    def test_ranking_prefers_macro_then_tail_then_lower_nll(self):
        summaries = {key: {"conditions": {"native": {"outer_metrics": {
            "macro_accuracy": .7 if key == "control" else .72,
            "tail_accuracy": .6, "micro_accuracy": .75, "macro_nll": 1.2}}}}
            for key in ("control", "v20_dora", "v20_loraplus")}
        for field, value in (("macro_nll", 1.1), ("tail_accuracy", .61), ("macro_accuracy", .73)):
            with self.subTest(field=field):
                current = deepcopy(summaries)
                current["v20_dora"]["conditions"]["native"]["outer_metrics"][field] = value
                self.assertEqual(comparison.choose(current)[2], "v20_dora")

    def test_v19_and_v20_winners_freeze_joint_epoch_calibration_for_refit(self):
        for winner in ("v19_dropout", "v20_dora", "v20_loraplus"):
            with self.subTest(winner=winner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); paths, resource, fallback = make_fixture(root, winner=winner)
                selected = selection.make_selection(*paths, resource_json=resource, bootstrap_repeats=3)
                output = root / "selection.json"; output.write_text(json.dumps(selected))
                checked = selection.load_selection(output, dataset_signature="dataset", base_identity=BASE_IDENTITY)
                self.assertEqual(checked["selected_candidate"], winner)
                self.assertEqual(checked["source_format_version"], comparison.EXPECTED[winner][0])
                self.assertEqual(checked["selected_epoch"], 13)
                self.assertEqual(checked["calibration"]["evaluation_protocol"], json.loads(json.dumps(PROTOCOL)))
                self.assertEqual(checked["initialization"], "official_base_only")
                self.assertEqual(checked["fallback"]["directory"], str(fallback))

    def test_fallback_reuses_historical_full_refit_without_shared_dataset_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths, resource, fallback = make_fixture(root, winner=None, absent=("v20_dora",))
            value = selection.make_selection(*paths, resource_json=resource, bootstrap_repeats=3, fallback_v18=fallback)
            output = root / "selection.json"; output.write_text(json.dumps(value))
            checked = selection.read_decision(output)
            self.assertFalse(checked["refit_required"])
            self.assertEqual(checked["source_format_version"], 18)
            self.assertEqual(checked["dataset_signature"], "historical_pre_dedup_data")
            self.assertEqual(checked["comparison_dataset_signature"], "dataset")
            self.assertIsNone(checked["selected_evaluation"])
            with self.assertRaisesRegex(ValueError, "without refit"):
                selection.load_selection(output)
            (fallback / "pred_results.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "historical V18"):
                selection.read_decision(output)

    def test_missing_runnable_waits_and_only_explicit_resource_absence_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, resource, _ = make_fixture(Path(directory), absent=("v20_dora",))
            result = comparison.compare(*paths, 3, resource_json=resource)
            self.assertEqual(result["candidates"]["v20_dora"]["status"], "resource_infeasible")
            paths[2] = None
            with self.assertRaisesRegex(ValueError, "wait for its joint evaluation"):
                comparison.compare(*paths, 3, resource_json=resource)

    def test_protocol_data_and_fold_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, resource, _ = make_fixture(Path(directory))
            target = paths[-1] / "strict_eval.json"; original = json.loads(target.read_text())
            for field, value in (("method", "old_two_step"), ("evaluation_protocol", {}),
                ("dataset_signature", "different"), ("outer_fold_ids", [0] * 100),
                ("source_format_version", 19), ("resource_sha256", "changed"), ("image_size", 384),
                ("class_training_support", [99, 101])):
                target.write_text(json.dumps({**original, field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    comparison.compare(*paths, 3, resource_json=resource)
            for field in ("tail_accuracy", "macro_nll"):
                altered = deepcopy(original)
                altered["conditions"]["native"]["outer_metrics"][field] += .01
                target.write_text(json.dumps(altered))
                with self.subTest(metric=field), self.assertRaisesRegex(ValueError, "summary and OOF"):
                    comparison.compare(*paths, 3, resource_json=resource)
            target.write_text(json.dumps(original))
            (paths[-1] / "val_row_keys.json").write_text(json.dumps(["changed"] * 100))
            with self.assertRaises(ValueError):
                comparison.compare(*paths, 3, resource_json=resource)

    def test_frozen_decision_rejects_configuration_calibration_bias_and_winner_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths, resource, _ = make_fixture(root)
            selected = selection.make_selection(*paths, resource_json=resource, bootstrap_repeats=3)
            output = root / "selection.json"
            for field in ("epoch", "calibration", "config", "bias", "winner", "gate"):
                modified = deepcopy(selected)
                if field == "epoch": modified["selected_epoch"] = 14
                elif field == "calibration": modified["calibration"]["alpha"] = .5
                elif field == "config": modified["training_config"]["lora_lr_b"] = .001
                elif field == "bias": modified["class_bias"][0] = 1.
                elif field == "winner": modified["comparison"]["winner_key"] = "v20_dora"
                else: modified["gates"]["macro_gain"] = 0.
                output.write_text(json.dumps(modified))
                with self.subTest(field=field), self.assertRaises(ValueError): selection.read_decision(output)

    def test_mutated_epoch_checkpoint_and_evaluation_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths, resource, _ = make_fixture(root)
            selected = selection.make_selection(*paths, resource_json=resource, bootstrap_repeats=3)
            output = root / "selection.json"; output.write_text(json.dumps(selected))
            original = paths[-1].parent / "validation_epochs" / "epoch_02.pt"
            before = original.read_bytes(); original.write_bytes(before + b"mutation")
            with self.assertRaisesRegex(ValueError, "source epoch checkpoint"):
                selection.read_decision(output)
            original.write_bytes(before)
            logits = paths[0] / "val_oof_native.npy"
            np.save(logits, np.load(logits) + 1)
            with self.assertRaisesRegex(ValueError, "evaluation artifact"):
                selection.read_decision(output)

    def test_v20_prediction_rejects_mismatched_geometry_before_building_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths, _, _ = make_fixture(root)
            value = torch.load(paths[-1] / "model.pt", map_location="cpu", weights_only=False)
            value["calibration"]["lora_dropout"] = .05
            source = root / "bad.pt"; torch.save(value, source)
            argv = ["predict_v20.py", "--checkpoint", str(source), "--test-dir", str(root),
                    "--output", str(root / "out.csv"), "--zip-output", str(root / "out.zip"), "--device", "cpu"]
            with patch.object(sys, "argv", argv), patch.object(prediction, "build_classifier_from_checkpoint") as build:
                with self.assertRaisesRegex(ValueError, "calibration lora_dropout"):
                    prediction.main()
                build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
