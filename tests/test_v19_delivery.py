"""V19 frozen gates, unavailable candidates, audit provenance and prediction."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import compare_v19 as comparison
import evaluate_tta_v19 as evaluation
import predict_v19 as prediction
import select_v19 as selection
from test_v15_delivery import BASE_IDENTITY, TinyPredictor
from test_v18_delivery import checkpoint as v18_checkpoint
from v18_model import RESOLUTION_VERSION as V18_RESOLUTION_VERSION
from v18_runtime import scientific_config as v18_scientific_config
from v19_core import file_sha256, recipe_config
from v19_model import _official_trainable_shapes
from v19_runtime import scientific_config
from v19_views import resolution_processor


def checkpoint(recipe="resolution384_rank32", epoch=18, *, resource=None):
    value = v18_checkpoint("rank32", epoch)
    value["format_version"] = 19
    value["config"].update(recipe_config(recipe))
    value["config"].update(resource_branch="standard",
        resource_json=str(resource or "/frozen/resource.json"),
        resource_sha256=file_sha256(resource) if resource else "resource-hash",
        audit_json="/frozen/audit.json", audit_sha256="audit-hash", batch_size=256,
        gradient_accumulation=1)
    if resource:
        audit = json.loads(Path(resource).read_text())["audit"]
        value["config"].update(audit_json=audit["path"], audit_sha256=audit["sha256"])
    shapes = {**_official_trainable_shapes(2, value["config"]), "logit_scale": ()}
    zero = torch.zeros(())
    value["model"] = {name: zero.expand(shape) for name, shape in shapes.items()}
    value["trainable_parameter_names"] = sorted(name for name in shapes if name != "logit_scale")
    value["calibration"].update(selected_epoch=epoch,
        **{field: value["config"][field] for field in comparison.GEOMETRY_FIELDS + comparison.DROPOUT_FIELDS})
    return value


def fixture(root, key, *, resource=None, correct=350, stress_correct=350, tail=.6, nll=1.2, same_run=False):
    version, recipe = comparison.EXPECTED[key]
    ckpt = v18_checkpoint("rank32") if version == 18 else checkpoint(recipe, resource=resource)
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
        prediction = 1 - labels
        for label in (0, 1): prediction[label * 500:label * 500 + count] = label
        logits = np.zeros((len(labels), 2), dtype=np.float32)
        logits[np.arange(len(labels)), prediction] = 4
        np.save(out / f"val_oof_{condition}.npy", logits)
        conditions[condition] = {"outer_metrics": {"macro_accuracy": count / 500,
            "micro_accuracy": count / 500, "tail_accuracy": tail, "macro_nll": nll}}
    summary = {"format_version": version, "recipe": recipe, "dataset_signature": "dataset",
        "calibrated_checkpoint": str(target), "calibrated_checkpoint_sha256": file_sha256(target),
        "checkpoint_sha256": file_sha256(source), "source_epoch_checkpoint": str(source),
        "selected": {"epoch": 18, "view_weights": ckpt["calibration"]["view_weights"], "alpha": 0.},
        "precision": "fp32", "stress_version": evaluation.STRESS_VERSION,
        "outer_fold_ids": (np.arange(1000) % 5).tolist(), "conditions": conditions,
        **{field: ckpt["config"][field] for field in comparison.GEOMETRY_FIELDS},
        "scientific_config": (v18_scientific_config if version == 18 else scientific_config)(ckpt["config"]),
        "preprocessing_version": V18_RESOLUTION_VERSION if version == 18 else evaluation.RESOLUTION_VERSION}
    if version == 19:
        summary.update({field: ckpt["config"][field] for field in
            (*comparison.DROPOUT_FIELDS, "resource_json", "resource_sha256", "resource_branch", "audit_json", "audit_sha256")})
    (out / "strict_eval.json").write_text(json.dumps(summary))
    return out


def inputs(root, *, absent=(), **options):
    baseline = fixture(root, "baseline_v18", **options.get("baseline_v18", {}))
    audit = root / "audit.json"
    audit.write_text(json.dumps({"format_version": 19, "kind": "v19_generalization_audit", "status": "passed",
        "dataset_signature": "dataset", "decoded_cross_split_groups": 0, "decoded_cross_split_pairs": 0}))
    plan = {"format_version": 19, "kind": "v19_resource_plan", "branch": "standard",
        "activation_checkpointing": False, "gates": comparison.GATES,
        "dataset_signature": "dataset", "base_model_identity": BASE_IDENTITY,
        "audit": {"path": str(audit), "sha256": file_sha256(audit)},
        "baseline": {"directory": str(baseline), "model_sha256": file_sha256(baseline / "model.pt"),
                     "evaluation_sha256": file_sha256(baseline / "strict_eval.json")}, "candidates": {}}
    for key in comparison.CANDIDATE_KEYS:
        spec = recipe_config(comparison.EXPECTED[key][1])
        plan["candidates"][key] = {"run_dir": str(root / key), "recipe": spec["recipe"],
            "status": "resource_infeasible" if key in absent else "runnable",
            "batch_size": 256, "gradient_accumulation": 1, "workers": 8,
            **{field: spec[field] for field in comparison.GEOMETRY_FIELDS + comparison.DROPOUT_FIELDS}}
    resource = root / "resource.json"; resource.write_text(json.dumps(plan))
    candidates = [None if key in absent else fixture(root, key, resource=resource, **options.get(key, {}))
                  for key in comparison.CANDIDATE_KEYS]
    return [baseline, *candidates], resource


class V19DeliveryTests(unittest.TestCase):
    def setUp(self):
        # Full OOM-report/source validation belongs to pipeline-support tests.
        # These fixtures keep actual model/cache/audit hashes and scientific checks.
        reader = patch.object(comparison, "_read_resource_plan", side_effect=lambda p: json.loads(Path(p).read_text()))
        reader.start(); self.addCleanup(reader.stop)
        metrics = patch.object(comparison, "FROZEN_V18_NATIVE", {})
        metrics.start(); self.addCleanup(metrics.stop)

    def test_gate_boundaries_micro_is_unrounded_and_ties_choose_a(self):
        summaries = {key: {"conditions": {"native": {"outer_metrics": {
            "macro_accuracy": .7 if key == "baseline_v18" else .702, "micro_accuracy": .75,
            "tail_accuracy": .6 if key == "baseline_v18" else .597, "macro_nll": 1.2}}}}
            for key in comparison.INPUT_KEYS}
        self.assertEqual(comparison.choose(summaries)[2], "candidate_a")
        a = summaries["candidate_a"]["conditions"]["native"]["outer_metrics"]
        b = summaries["candidate_b"]["conditions"]["native"]["outer_metrics"]
        a["micro_accuracy"] = np.nextafter(.75, 0.)
        self.assertEqual(comparison.choose(summaries)[2], "candidate_b")
        b["tail_accuracy"] = .5969
        self.assertEqual(comparison.choose(summaries)[2], "baseline_v18")
        a["micro_accuracy"] = .75; b["tail_accuracy"] = .597
        for field, value in (("macro_nll", 1.1), ("tail_accuracy", .61), ("macro_accuracy", .72)):
            b[field] = value
            self.assertEqual(comparison.choose(summaries)[2], "candidate_b")

    def test_both_single_or_no_runnable_candidates_keep_fixed_v18_fallback(self):
        for absent in ((), ("candidate_a",), ("candidate_b",), comparison.CANDIDATE_KEYS):
            with self.subTest(absent=absent), tempfile.TemporaryDirectory() as directory:
                paths, resource = inputs(Path(directory), absent=absent,
                    candidate_a={"correct": 355}, candidate_b={"correct": 356})
                result = comparison.compare(*paths, 5, resource_json=resource)
                expected = "candidate_b" if "candidate_b" not in absent else (
                    "candidate_a" if "candidate_a" not in absent else "baseline_v18")
                self.assertEqual(result["winner_key"], expected)
                for key in absent:
                    self.assertIsNone(result["inputs"][key])
                    self.assertEqual(result["candidates"][key]["status"], "resource_infeasible")
                    self.assertNotIn("conditions", result["candidates"][key])

    def test_runnable_missing_candidate_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, resource = inputs(Path(directory))
            with self.assertRaisesRegex(ValueError, "runnable candidate_a"):
                comparison.compare(paths[0], None, paths[2], 5, resource_json=resource)
            plan = json.loads(resource.read_text()); plan["candidates"]["candidate_a"]["status"] = "failed"
            resource.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "status"):
                comparison.compare(paths[0], None, paths[2], 5, resource_json=resource)

    def test_infeasible_candidate_must_not_be_submitted_for_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths, resource = inputs(root, absent=("candidate_a",))
            paths[1] = fixture(root, "candidate_a", resource=resource)
            with self.assertRaisesRegex(ValueError, "must not supply"):
                comparison.compare(*paths, 5, resource_json=resource)

    def test_failed_overlap_or_unbound_audit_blocks_comparison(self):
        for field, value in (("status", "failed"), ("decoded_cross_split_groups", 1),
                             ("decoded_cross_split_pairs", 1), ("decoded_cross_split_pairs", False),
                             ("dataset_signature", "other")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); paths, resource = inputs(root, absent=comparison.CANDIDATE_KEYS)
                audit = root / "audit.json"; data = json.loads(audit.read_text()); data[field] = value
                audit.write_text(json.dumps(data))
                plan = json.loads(resource.read_text()); plan["audit"]["sha256"] = file_sha256(audit)
                resource.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "audit must pass"):
                    comparison.compare(*paths, 5, resource_json=resource)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths, resource = inputs(root)
            (root / "audit.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "audit missing or changed"):
                comparison.compare(*paths, 5, resource_json=resource)

    def test_historical_anchor_geometry_dropout_and_rows_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, resource = inputs(Path(directory), candidate_a={"correct": 351, "tail": .597})
            result = comparison.compare(*paths, 5, resource_json=resource)
            self.assertAlmostEqual(result["candidates"]["candidate_a"]["macro_gain_pp"], .2)
            with patch.object(comparison, "FROZEN_V18_NATIVE", {"macro_accuracy": .7591118812561035}), \
                    self.assertRaisesRegex(ValueError, "historical evaluation"):
                comparison.compare(*paths, 5, resource_json=resource)
            target = paths[1] / "strict_eval.json"; original = json.loads(target.read_text())
            for field, value in (("image_size", 320), ("lora_rank", 16), ("lora_dropout", .05),
                                 ("mlp_lora_dropout", .05), ("precision", "bf16"),
                                 ("outer_fold_ids", [0]), ("audit_sha256", "changed")):
                target.write_text(json.dumps({**original, field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    comparison.compare(*paths, 5, resource_json=resource)
            target.write_text(json.dumps(original))
            rows = paths[1] / "val_row_keys.json"; rows.write_text(json.dumps(list(reversed(json.loads(rows.read_text())))))
            with self.assertRaisesRegex(ValueError, "rows/labels"):
                comparison.compare(*paths, 5, resource_json=resource)

    def test_selection_freezes_sources_and_supports_absent_candidates(self):
        cases = (((), "candidate_a"), ((), "candidate_b"), ((), "baseline_v18"),
                 (("candidate_a",), "candidate_b"), (comparison.CANDIDATE_KEYS, "baseline_v18"))
        for absent, winner in cases:
            with self.subTest(absent=absent, winner=winner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); paths, resource = inputs(root, absent=absent, **{winner: {"correct": 355}})
                value = selection.make_selection(*paths, resource_json=resource, bootstrap_repeats=5)
                output = root / "selection.json"; output.write_text(json.dumps(value))
                self.assertEqual(selection.read_decision(output)["comparison"]["winner_key"], winner)
                self.assertEqual(value["source_format_version"], 18 if winner == "baseline_v18" else 19)
                self.assertEqual(value["fallback"]["recipe"], "rank32")
                if winner == "baseline_v18":
                    with self.assertRaisesRegex(ValueError, "without refit"): selection.load_selection(output)
                else:
                    self.assertEqual(selection.load_selection(output, base_identity=BASE_IDENTITY)["recipe"],
                                     comparison.EXPECTED[winner][1])
                for field in ("audit", "calibration", "config", "bias", "gate", "source", "winner", "status"):
                    bad = deepcopy(value)
                    if field == "audit": bad["audit"]["sha256"] = "changed"
                    elif field == "calibration": bad["calibration"]["alpha"] = .5
                    elif field == "config": bad["training_config"]["lora_dropout"] = .1
                    elif field == "bias": bad["class_bias"][0] = 1.
                    elif field == "gate": bad["gates"]["overall_noninferiority"] = False
                    elif field == "source": bad["source_checkpoint_sha256"] = "changed"
                    elif field == "status": bad["comparison"]["candidates"]["candidate_a"]["status"] = "changed"
                    else: bad["comparison"]["winner_key"] = "candidate_b" if winner != "candidate_b" else "candidate_a"
                    output.write_text(json.dumps(bad))
                    with self.subTest(field=field), self.assertRaises(ValueError): selection.read_decision(output)
                output.write_text(json.dumps(value))
                logits = paths[0] / "val_oof_native.npy"; saved = np.load(logits); saved[0] += 1
                np.save(logits, saved)
                with self.assertRaisesRegex(ValueError, "evaluation artifact"): selection.read_decision(output)

    def test_cli_accepts_missing_infeasible_candidate_and_baseline_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths, resource = inputs(root, absent=("candidate_a",), candidate_b={"correct": 355})
            output = root / "selection.json"
            argv = ["select_v19.py", "--baseline-v18", str(paths[0]), "--candidate-b", str(paths[2]),
                    "--resource-json", str(resource), "--output", str(output), "--bootstrap", "5"]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()): selection.main()
            self.assertEqual(selection.read_decision(output)["recipe"], "rank32_dropout")

    def test_prediction_rejects_calibration_geometry_and_dropout_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "model.pt"
            argv = ["predict_v19.py", "--checkpoint", str(source), "--test-dir", str(root),
                    "--output", str(root / "pred.csv"), "--zip-output", str(root / "pred.zip"), "--device", "cpu"]
            for field, value in (("view_weights", {"center": 2.}), ("alpha", 1.5), ("selected_epoch", 17),
                                 ("image_size", 320), ("lora_rank", 16), ("lora_dropout", .05),
                                 ("mlp_lora_dropout", .05), ("precision", "bf16")):
                current = checkpoint(); current["calibration"][field] = value; torch.save(current, source)
                with self.subTest(field=field), patch.object(sys, "argv", argv), \
                        patch.object(prediction, "build_classifier_from_checkpoint") as build, self.assertRaises(ValueError):
                    prediction.main()
                build.assert_not_called()

    def test_prediction_384_320_dropout_and_v18_fallback_use_actual_geometry(self):
        processor = SimpleNamespace(image_processor=SimpleNamespace(size={"shortest_edge": 224},
            crop_size={"height": 224, "width": 224}, image_mean=(.5, .5, .5), image_std=(.25, .25, .25)))
        seen = []
        class ShapePredictor(TinyPredictor):
            def forward(self, pixels, _=None):
                seen.append(tuple(pixels.shape[-2:])); return super().forward(pixels)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); images = root / "images"; images.mkdir()
            for index in range(3): Image.new("RGB", (360, 420), (index * 70, 90, 120)).save(images / f"{index}.png")
            for name, value in (("a", checkpoint()), ("b", checkpoint("rank32_dropout")),
                                ("baseline", v18_checkpoint("rank32"))):
                source = root / f"{name}.pt"; torch.save(value, source); out = root / name
                prep = resolution_processor(processor, value["config"]["image_size"])
                argv = ["predict_v19.py", "--checkpoint", str(source), "--model-dir", "official",
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
