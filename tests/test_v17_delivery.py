"""V17 native-only selection, immutable provenance, evaluation and submission."""
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

import compare_v17 as comparison
import evaluate_tta_v17 as evaluation
import predict_v17 as prediction
import select_v17 as selection
from test_v15_delivery import BASE_IDENTITY, TinyPredictor, checkpoint as old_checkpoint
from v17_core import recipe_config, file_sha256
from v11_resolution import resolution_processor


def checkpoint(recipe="agreement_recovery", epoch=18):
    value = old_checkpoint(15, epoch)
    value["format_version"] = 17
    value["config"].update(recipe_config(recipe))
    value["trainable_parameter_names"] = sorted(key for key in value["model"] if key != "logit_scale")
    value["calibration"].update(selected_epoch=epoch, image_size=320)
    return value


def fixture(root, key, *, correct=350, stress_correct=350, tail=.6, nll=1.2):
    version, recipe = selection.EXPECTED[key]
    ckpt = checkpoint(recipe) if version == 17 else old_checkpoint(version)
    run = root / key; run.mkdir()
    source = run / "epoch_18.pt"; torch.save(ckpt, source)
    out = run / "evaluation"; out.mkdir()
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
        "precision": "fp32", "stress_version": evaluation.STRESS_VERSION, "image_size": 320,
        "outer_fold_ids": (np.arange(1000) % 5).tolist(), "conditions": conditions}
    (out / "strict_eval.json").write_text(json.dumps(summary))
    return out


def inputs(root, **options):
    return [fixture(root, key, **options.get(key, {})) for key in comparison.INPUT_KEYS]


class V17DeliveryTests(unittest.TestCase):
    def test_gate_uses_fixed_v15_baseline_and_pressure_is_only_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = inputs(Path(directory), baseline_v15={"correct": 352},
                           candidate_a={"correct": 353, "tail": .597, "stress_correct": 200},
                           candidate_b={"correct": 353, "tail": .5969})
            actual = comparison.compare(*paths, bootstrap_repeats=20)
            self.assertEqual(actual["baseline_key"], "baseline_v15")
            self.assertEqual(actual["winner_key"], "candidate_a")
            self.assertTrue(actual["refit_required"])
            self.assertAlmostEqual(actual["candidates"]["candidate_a"]["macro_gain_pp"], .2)
            self.assertFalse(actual["candidates"]["candidate_b"]["candidate_accepted"])
            self.assertEqual(actual["candidates"]["candidate_a"]["conditions"]["native"]["paired_class_bootstrap"]["repeats"], 20)
            self.assertEqual(actual["stress_role"], "diagnostic_only_not_a_selection_gate")

    def test_ties_follow_macro_tail_nll_then_a(self):
        summaries = {key: {"conditions": {"native": {"outer_metrics": {
            "macro_accuracy": .7 if key.startswith("baseline") else .71,
            "tail_accuracy": .6, "macro_nll": 1.2}}}} for key in comparison.INPUT_KEYS}
        self.assertEqual(comparison.choose(summaries)[::2], ("baseline_v15", "candidate_a"))
        b = summaries["candidate_b"]["conditions"]["native"]["outer_metrics"]
        for field, value in (("macro_nll", 1.1), ("tail_accuracy", .61), ("macro_accuracy", .72)):
            b[field] = value
            self.assertEqual(comparison.choose(summaries)[2], "candidate_b")

    def test_comparison_rejects_identity_and_oof_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = inputs(Path(directory))
            p = paths[-1] / "strict_eval.json"; original = json.loads(p.read_text())
            for field, value in (("outer_fold_ids", [0]), ("precision", "bf16"),
                                 ("stress_version", "other"), ("image_size", 224),
                                 ("dataset_signature", "other")):
                p.write_text(json.dumps({**original, field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    comparison.compare(*paths, 10)
            p.write_text(json.dumps(original))
            rows = paths[-1] / "val_row_keys.json"; original_rows = rows.read_text()
            rows.write_text(json.dumps(list(reversed(json.loads(original_rows)))))
            with self.assertRaisesRegex(ValueError, "rows/labels"):
                comparison.compare(*paths, 10)
            rows.write_text(original_rows)
            bad = deepcopy(original); bad["conditions"]["native"]["outer_metrics"]["macro_accuracy"] = .99
            p.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "OOF logits"):
                comparison.compare(*paths, 10)

    def test_selection_both_recipes_and_fallback_refuse_refit(self):
        for winner in ("candidate_a", "candidate_b", "baseline_v15"):
            with self.subTest(winner=winner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); paths = inputs(root, **{winner: {"correct": 355}})
                output = root / "selection.json"
                argv = ["select_v17.py", "--output", str(output), "--bootstrap", "10"]
                for key, path in zip(comparison.INPUT_KEYS, paths):
                    option = "--baseline" if key == "baseline_v15" else "--" + key.replace("_", "-")
                    argv.extend([option, str(path)])
                with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()): selection.main()
                decision = selection.read_decision(output)
                self.assertEqual(decision["comparison"]["winner_key"], winner)
                self.assertEqual(decision["source_format_version"], 15 if winner.startswith("baseline") else 17)
                self.assertEqual(decision["selected_evaluation"], str(paths[comparison.INPUT_KEYS.index(winner)].resolve()))
                if winner.startswith("baseline"):
                    self.assertFalse(decision["refit_required"])
                    self.assertEqual(decision["selected_stage"], "fallback")
                    with self.assertRaisesRegex(ValueError, "without refit"): selection.load_selection(output)
                else:
                    self.assertEqual(selection.load_selection(output, base_identity=BASE_IDENTITY)["recipe"], selection.EXPECTED[winner][1])
                for field in ("recipe", "schedule", "calibration", "bias", "acceptance", "source", "winner"):
                    bad = deepcopy(decision)
                    if field == "recipe": bad["recipe"] = "other"
                    elif field == "schedule": bad["schedule_epochs"] = 18
                    elif field == "calibration": bad["calibration"]["alpha"] = .5
                    elif field == "bias": bad["class_bias"][0] = 1.
                    elif field == "acceptance": bad["candidate_accepted"] = not bad["candidate_accepted"]
                    elif field == "source": bad["source_checkpoint_sha256"] = "changed"
                    else: bad["comparison"]["winner_key"] = "candidate_b" if winner != "candidate_b" else "candidate_a"
                    output.write_text(json.dumps(bad))
                    with self.subTest(field=field), self.assertRaises(ValueError): selection.read_decision(output)
                output.write_text(json.dumps(decision))
                p = paths[0] / "strict_eval.json"; p.write_text(p.read_text() + " ")
                with self.assertRaisesRegex(ValueError, "evaluation summary"): selection.read_decision(output)

    def test_cli_baseline_alias_and_refit_identity_checks(self):
        import argparse
        parser = argparse.ArgumentParser()
        comparison.add_arguments(parser)
        for option in ("--baseline", "--baseline-v15"):
            args = parser.parse_args([option, "v15", "--candidate-a", "a", "--candidate-b", "b", "--output", "out"])
            self.assertEqual(args.baseline_v15, Path("v15"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths = inputs(root, candidate_a={"correct": 355})
            value = selection.make_selection(*paths, bootstrap_repeats=10)
            output = root / "selection.json"; output.write_text(json.dumps(value))
            for kwargs in ({"dataset_signature": "other"}, {"class_names": ["bad"]}, {"base_identity": {"bad": "id"}}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "mismatch"):
                    selection.load_selection(output, **kwargs)

    def test_fixed_baseline_and_candidate_recipes_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = inputs(Path(directory))
            for index, other in ((0, old_checkpoint(13)), (1, checkpoint("dynamic_prototype"))):
                source = paths[index].parent / "epoch_18.pt"
                target = paths[index] / "model.pt"
                original_source, original_target = source.read_bytes(), target.read_bytes()
                summary_path = paths[index] / "strict_eval.json"; original_summary = summary_path.read_text()
                torch.save(other, source); torch.save(other, target)
                summary = json.loads(original_summary)
                summary.update(format_version=other["format_version"], recipe=other["config"]["recipe"],
                               checkpoint_sha256=file_sha256(source), calibrated_checkpoint_sha256=file_sha256(target))
                summary_path.write_text(json.dumps(summary))
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, "requires format"):
                    comparison.compare(*paths, 10)
                source.write_bytes(original_source); target.write_bytes(original_target)
                summary_path.write_text(original_summary)

    def test_prediction_rejects_invalid_calibration_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "model.pt"
            argv = ["predict_v17.py", "--checkpoint", str(source), "--test-dir", str(root),
                    "--output", str(root / "pred.csv"), "--zip-output", str(root / "pred.zip"), "--device", "cpu"]
            for field, value in (("view_weights", {"center": 2.}), ("view_weights", {"center": float("nan")}),
                                 ("view_weights", {"center": -1., "hflip": 2.}), ("alpha", 1.5),
                                 ("selected_epoch", 17), ("image_size", 224), ("precision", "bf16")):
                current = checkpoint(); current["calibration"][field] = value; torch.save(current, source)
                with self.subTest(field=field, value=value), patch.object(sys, "argv", argv), \
                        patch.object(prediction, "build_classifier_from_checkpoint") as build, self.assertRaises(ValueError):
                    prediction.main()
                build.assert_not_called()

    def test_prediction_outputs_cannot_alias_checkpoint_or_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "model.pt"
            base = ["predict_v17.py", "--checkpoint", str(source), "--test-dir", str(root), "--device", "cpu"]
            cases = [(["--output", str(source), "--zip-output", str(root / "out.zip")], "checkpoint"),
                     (["--output", str(root / "same"), "--zip-output", str(root / "same")], "distinct"),
                     (["--output", str(root / "values.npy"), "--zip-output", str(root / "out.zip"),
                       "--logits-output", str(root / "values")], "distinct")]
            for args, message in cases:
                with self.subTest(args=args), patch.object(sys, "argv", base + args), self.assertRaisesRegex(ValueError, message):
                    prediction.main()

    def test_epoch_contract_for_both_recipes_and_mixed_states_rejected(self):
        manifest = SimpleNamespace(signature="dataset", validation_indices=[0, 1],
            rows=[("a", 0, "0000"), ("b", 1, "0001")], class_names=["0000", "0001"])
        for recipe in ("agreement_recovery", "dynamic_prototype"):
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); epochs = root / "validation_epochs"; epochs.mkdir()
                for epoch in range(1, 25): torch.save(checkpoint(recipe, epoch), epochs / f"epoch_{epoch:02d}.pt")
                paths, _ = evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))
                self.assertEqual(len(paths), 24)
                bad = checkpoint(recipe, 24); bad["config"]["recipe"] = "dynamic_prototype" if recipe == "agreement_recovery" else "agreement_recovery"
                torch.save(bad, paths[24])
                with self.assertRaises(ValueError): evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))
                paths[24].unlink()
                with self.assertRaises(FileNotFoundError): evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))

    def test_training_logits_cache_requires_precision_rows_hash_and_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "epoch_01.pt"; source.write_bytes(b"checkpoint")
            logits = np.eye(2, dtype=np.float32)
            metadata = {"center": logits, "hflip": logits, "labels": np.array([0, 1]),
                "row_indices": np.array([0, 1]), "precision": np.asarray("fp32"),
                "format_version": np.asarray(17), "checkpoint_sha256": np.asarray(file_sha256(source))}
            manifest = SimpleNamespace(signature="dataset", class_names=["0000", "0001"])
            def cache():
                return evaluation.LogitCache(output_dir=root / "evaluation", paths={1: source},
                    checkpoints={1: checkpoint(epoch=1)}, manifest=manifest, indices=[0, 1],
                    labels=np.array([0, 1]), model_dir="official", device=torch.device("cpu"),
                    batch_size=2, workers=0, force=False)
            np.savez(root / "epoch_01_logits.npz", **metadata)
            with patch.object(evaluation, "build_classifier_from_checkpoint", side_effect=RuntimeError("recompute")):
                np.testing.assert_array_equal(cache().get_views(1, "native", False)["center"], logits)
                for key, value in (("checkpoint_sha256", np.asarray("changed")), ("precision", np.asarray("bf16")),
                                   ("format_version", np.asarray(15)), ("row_indices", np.array([1, 0]))):
                    np.savez(root / "epoch_01_logits.npz", **{**metadata, key: value})
                    with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "recompute"): cache().get_views(1, "native", False)

    def test_prediction_both_recipes_and_legacy_fallback_validate_zip(self):
        prep = resolution_processor(SimpleNamespace(image_processor=SimpleNamespace(
            size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224},
            image_mean=(.5, .5, .5), image_std=(.25, .25, .25))))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); images = root / "images"; images.mkdir()
            for index in range(3): Image.new("RGB", (360, 420), (index * 70, 90, 120)).save(images / f"{index}.png")
            for name, value in (("a", checkpoint()), ("b", checkpoint("dynamic_prototype")),
                                ("old13", old_checkpoint(13)), ("old15", old_checkpoint(15))):
                source = root / f"{name}.pt"; torch.save(value, source); out = root / name
                argv = ["predict_v17.py", "--checkpoint", str(source), "--model-dir", "official",
                        "--test-dir", str(images), "--output", str(out / "pred_results.csv"),
                        "--zip-output", str(out / "pred_results.zip"), "--device", "cpu",
                        "--expected-rows", "3", "--workers", "0", "--batch-size", "2"]
                with patch.object(sys, "argv", argv), patch.object(prediction, "build_classifier_from_checkpoint", return_value=(TinyPredictor(), prep)), redirect_stdout(io.StringIO()): prediction.main()
                prediction.validate_submission(out / "pred_results.csv", out / "pred_results.zip", 3, [f"{n}.png" for n in range(3)])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
