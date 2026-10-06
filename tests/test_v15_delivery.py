"""V15 evaluation, frozen-baseline selection and compatible submission contracts."""
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

import compare_v15 as comparison
import evaluate_tta_v15 as evaluation
import predict_v15 as prediction
import select_v15 as selection
from v14_core import recipe_config as old_recipe
from v15_core import recipe_config, file_sha256
from v11_resolution import resolution_processor


BASE_IDENTITY = {"config.json": "official", "model.safetensors": "official-weights"}


def model_state():
    # Expanded scalar views preserve official tensor shapes without large fixtures.
    zero = torch.zeros(())
    shapes = {"classifier": (2, 512), "logit_scale": (), "adapter.scale": (),
              "adapter.net.0.weight": (128, 512), "adapter.net.0.bias": (128,),
              "adapter.net.3.weight": (512, 128), "adapter.net.3.bias": (512,)}
    for layer in range(12):
        prefix = f"clip.vision_model.encoder.layers.{layer}."
        for target in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[prefix + f"self_attn.{target}.lora_a"] = (16, 768)
            shapes[prefix + f"self_attn.{target}.lora_b"] = (768, 16)
        shapes[prefix + "mlp.fc1.lora_a"] = (16, 768)
        shapes[prefix + "mlp.fc1.lora_b"] = (3072, 16)
        shapes[prefix + "mlp.fc2.lora_a"] = (16, 3072)
        shapes[prefix + "mlp.fc2.lora_b"] = (768, 16)
    return {name: zero.expand(shape) for name, shape in shapes.items()}


def checkpoint(version=15, epoch=18):
    recipe = {13: "expanded", 14: "expanded_robust", 15: "expanded_mlp"}[version]
    config = (recipe_config if version == 15 else old_recipe)(recipe)
    config.update(stage="validate", image_size=320, zoom_shortest_edge=366,
                  interpolate_pos_encoding=True, conflict_policy="partial", sampler="repeat-factor",
                  base_model_identity=BASE_IDENTITY, feature_cache="frozen.npy", workers=8,
                  prefetch_factor=1, eval_batch_size=256, batch_size=256,
                  gradient_accumulation=1, precision="fp32")
    state = model_state() if version == 15 else {"weight": torch.tensor([1., 2.])}
    return {"format_version": version, "training_stage": "validation", "selected_epoch": epoch,
            "dataset_signature": "dataset", "class_names": ["0000", "0001"],
            "config": config, "validation": {"indices": [0, 1]},
            "model": state, "class_bias": torch.zeros(2),
            "trainable_parameter_names": sorted(name for name in state if name != "logit_scale"),
            "calibration": {"view_weights": {"center": .5, "hflip": .5}, "alpha": 0., "precision": "fp32"}}


def evaluation_fixture(root, version, *, correct=350, stress_correct=None, tail=.6):
    run = root / f"v{version}"; run.mkdir()
    source = run / "epoch_18.pt"
    ckpt = checkpoint(version)
    torch.save(ckpt, source)
    out = run / "evaluation"; out.mkdir()
    calibrated = out / "model.pt"; torch.save(ckpt, calibrated)
    labels = np.repeat(np.arange(2), 500)
    np.save(out / "val_labels.npy", labels)
    (out / "val_row_keys.json").write_text(json.dumps([f"row-{i}" for i in range(1000)]))
    conditions = {}
    for condition in selection.CONDITIONS:
        count = correct if condition == "native" or stress_correct is None else stress_correct
        predicted = 1 - labels
        for label in (0, 1):
            predicted[label * 500:label * 500 + count] = label
        logits = np.zeros((1000, 2), dtype=np.float32)
        logits[np.arange(1000), predicted] = 4
        np.save(out / f"val_oof_{condition}.npy", logits)
        conditions[condition] = {"outer_metrics": {"macro_accuracy": count / 500,
            "micro_accuracy": count / 500, "tail_accuracy": tail, "macro_nll": 1.2}}
    summary = {"format_version": version, "recipe": ckpt["config"]["recipe"],
        "dataset_signature": "dataset", "calibrated_checkpoint": str(calibrated),
        "calibrated_checkpoint_sha256": file_sha256(calibrated), "checkpoint_sha256": file_sha256(source),
        "source_epoch_checkpoint": str(source), "selected": {"epoch": 18,
            "view_weights": ckpt["calibration"]["view_weights"], "alpha": 0.},
        "precision": "fp32", "stress_version": evaluation.STRESS_VERSION, "image_size": 320,
        "outer_fold_ids": (np.arange(1000) % 5).tolist(), "conditions": conditions}
    (out / "strict_eval.json").write_text(json.dumps(summary))
    return out


def baseline_fixture(root, version):
    out = evaluation_fixture(root, version)
    ckpt = checkpoint(version)
    payload = {"format_version": 14, "kind": "v14_refit_selection", "recipe": ckpt["config"]["recipe"],
        "selected_epoch": 18, "schedule_epochs": 24, "augmentation": ckpt["config"]["augmentation"],
        "dataset_signature": "dataset", "class_names": ckpt["class_names"],
        "base_model_identity": BASE_IDENTITY, "source_checkpoint": str(out / "model.pt"),
        "source_checkpoint_sha256": file_sha256(out / "model.pt"), "training_config": ckpt["config"],
        "calibration": ckpt["calibration"], "class_bias": [0., 0.],
        "comparison": {"baseline": str(out) if version == 13 else "unused",
                       "candidate": str(out) if version == 14 else "unused"}}
    path = root / "v14_selection.json"; path.write_text(json.dumps(payload))
    return path, out


class TinyPredictor(torch.nn.Module):
    def forward(self, pixels, _=None):
        values = pixels.mean(dim=(1, 2, 3))
        return torch.stack((values, -values), dim=1), None


class V15DeliveryTests(unittest.TestCase):
    def test_gate_uses_existing_boundaries_and_bootstrap_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = evaluation_fixture(root, 14)
            candidate = evaluation_fixture(root, 15, correct=351, tail=.597, stress_correct=349)
            actual = comparison.compare(baseline, candidate, 20)
            self.assertTrue(actual["candidate_accepted"])
            self.assertAlmostEqual(actual["macro_gain_pp"], .2)
            self.assertAlmostEqual(actual["micro_gain_pp"], .2)
            self.assertEqual(actual["bootstrap_role"], "diagnostic_only_not_a_selection_gate")
            self.assertEqual(actual["conditions"]["native"]["paired_class_bootstrap"]["repeats"], 20)
            summary_path = candidate / "strict_eval.json"
            original = json.loads(summary_path.read_text())
            bad = deepcopy(original); bad["conditions"]["native"]["outer_metrics"]["tail_accuracy"] = .5969
            summary_path.write_text(json.dumps(bad))
            self.assertFalse(comparison.compare(baseline, candidate, 20)["candidate_accepted"])

    def test_comparison_rejects_mismatched_rows_labels_folds_precision_stresses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = evaluation_fixture(root, 14)
            candidate = evaluation_fixture(root, 15, correct=353)
            summary_path = candidate / "strict_eval.json"; original = json.loads(summary_path.read_text())
            for key, value in (("outer_fold_ids", [0]), ("precision", "bf16"),
                               ("stress_version", "other"), ("image_size", 224)):
                bad = {**original, key: value}; summary_path.write_text(json.dumps(bad))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    comparison.compare(baseline, candidate, 10)
            summary_path.write_text(json.dumps(original))
            rows_path = candidate / "val_row_keys.json"; rows = json.loads(rows_path.read_text())
            rows_path.write_text(json.dumps(list(reversed(rows))))
            with self.assertRaisesRegex(ValueError, "rows/labels"):
                comparison.compare(baseline, candidate, 10)
            rows_path.write_text(json.dumps(rows))
            labels_path = candidate / "val_labels.npy"; labels = np.load(labels_path); labels[0] = 1
            np.save(labels_path, labels)
            with self.assertRaisesRegex(ValueError, "rows/labels"):
                comparison.compare(baseline, candidate, 10)

    def test_selection_uses_both_possible_v14_winners_and_rejects_fallback_refit(self):
        for version in (13, 14):
            for count, accepted in ((353, True), (350, False)):
                with self.subTest(version=version, accepted=accepted), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    baseline_selection, baseline = baseline_fixture(root, version)
                    candidate = evaluation_fixture(root, 15, correct=count)
                    output = root / "v15_selection.json"
                    argv = ["select_v15.py", "--baseline-selection", str(baseline_selection),
                            "--candidate", str(candidate), "--model-dir", "official",
                            "--output", str(output), "--bootstrap", "10"]
                    with patch.object(sys, "argv", argv), patch.object(selection, "model_identity", return_value=BASE_IDENTITY), redirect_stdout(io.StringIO()):
                        selection.main()
                    decision = json.loads(output.read_text())
                    self.assertEqual(decision["candidate_accepted"], accepted)
                    self.assertEqual(selection.read_decision(output)["candidate_accepted"], accepted)
                    self.assertEqual(decision["comparison"]["baseline"], str(baseline.resolve()))
                    self.assertEqual(decision["fallback"]["source_checkpoint"], str((baseline / "model.pt").resolve()))
                    if not accepted:
                        with self.assertRaisesRegex(ValueError, "rejected"):
                            selection.load_selection(output)
                        bad = deepcopy(decision); bad["candidate_accepted"] = True
                        output.write_text(json.dumps(bad))
                        with self.assertRaisesRegex(ValueError, "frozen evaluation gate"):
                            selection.load_selection(output)
                        continue
                    selection.load_selection(output, dataset_signature="dataset", class_names=["0000", "0001"], base_identity=BASE_IDENTITY)
                    for field in ("schedule", "recipe", "bias", "mlp", "calibration"):
                        bad = deepcopy(decision)
                        if field == "schedule": bad["schedule_epochs"] = 18
                        elif field == "recipe": bad["recipe"] = "expanded_robust"
                        elif field == "bias": bad["class_bias"][0] = 1.
                        elif field == "mlp": bad["training_config"]["mlp_lora_rank"] = 8
                        else: bad["calibration"]["alpha"] = .5
                        output.write_text(json.dumps(bad))
                        with self.subTest(field=field), self.assertRaises(ValueError):
                            selection.load_selection(output)
                    output.write_text(json.dumps(decision))
                    Path(decision["source_checkpoint"]).write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "checkpoint missing or changed"):
                        selection.load_selection(output)

    def test_baseline_frozen_source_cannot_be_redirected_to_other_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = baseline_fixture(root, 14)
            other = evaluation_fixture(root, 13)
            payload = json.loads(path.read_text()); payload["comparison"]["candidate"] = str(other)
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "frozen selection source"):
                selection.read_baseline(path)

    def test_candidate_incomplete_mlp_state_is_rejected_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_selection, _ = baseline_fixture(root, 14)
            candidate = evaluation_fixture(root, 15, correct=353)
            summary_path = candidate / "strict_eval.json"
            summary = json.loads(summary_path.read_text())
            source = Path(summary["source_epoch_checkpoint"])
            ckpt = torch.load(source, weights_only=False)
            del ckpt["model"]["clip.vision_model.encoder.layers.11.mlp.fc2.lora_b"]
            torch.save(ckpt, source)
            torch.save(ckpt, candidate / "model.pt")
            summary["checkpoint_sha256"] = file_sha256(source)
            summary["calibrated_checkpoint_sha256"] = file_sha256(candidate / "model.pt")
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "complete trainable state"):
                selection.make_selection(baseline_selection, candidate, BASE_IDENTITY, bootstrap_repeats=10)

    def test_complete_epoch_contract_binds_mlp_configuration_and_trainable_names(self):
        manifest = SimpleNamespace(signature="dataset", validation_indices=[0, 1],
            rows=[("a", 0, "0000"), ("b", 1, "0001")], class_names=["0000", "0001"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); epochs = root / "validation_epochs"; epochs.mkdir()
            for epoch in range(1, 25):
                torch.save(checkpoint(15, epoch), epochs / f"epoch_{epoch:02d}.pt")
            paths, _ = evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))
            self.assertEqual(len(paths), 24)
            for field in ("format", "mlp", "metadata"):
                payload = checkpoint(15, 24)
                if field == "format": payload["format_version"] = 14
                elif field == "mlp": payload["config"]["mlp_lora_rank"] = 8
                else: payload["trainable_parameter_names"] = ["other"]
                torch.save(payload, paths[24])
                with self.subTest(field=field), self.assertRaises(ValueError):
                    evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))
            paths[24].unlink()
            with self.assertRaises(FileNotFoundError):
                evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))

    def test_cached_training_logits_require_format_hash_precision_and_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "epoch_01.pt"; source.write_bytes(b"checkpoint")
            logits = np.eye(2, dtype=np.float32)
            metadata = {"center": logits, "hflip": logits, "labels": np.array([0, 1]),
                "row_indices": np.array([0, 1]), "precision": np.asarray("fp32"),
                "format_version": np.asarray(15), "checkpoint_sha256": np.asarray(file_sha256(source))}
            np.savez(root / "epoch_01_logits.npz", **metadata)
            manifest = SimpleNamespace(signature="dataset", class_names=["0000", "0001"])
            def cache():
                return evaluation.LogitCache(output_dir=root / "evaluation", paths={1: source},
                    checkpoints={1: checkpoint(15, 1)}, manifest=manifest, indices=[0, 1],
                    labels=np.array([0, 1]), model_dir="official", device=torch.device("cpu"),
                    batch_size=2, workers=0, force=False)
            with patch.object(evaluation, "build_classifier_from_checkpoint", side_effect=RuntimeError("recompute")):
                np.testing.assert_array_equal(cache().get_views(1, "native", False)["center"], logits)
                for key, value in (("checkpoint_sha256", np.asarray("changed")),
                    ("precision", np.asarray("bf16_autocast_fp32_stats")),
                    ("format_version", np.asarray(14)), ("row_indices", np.array([1, 0]))):
                    np.savez(root / "epoch_01_logits.npz", **{**metadata, key: value})
                    with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "recompute"):
                        cache().get_views(1, "native", False)

    def test_prediction_supports_all_three_formats_and_validates_zip(self):
        prep = resolution_processor(SimpleNamespace(image_processor=SimpleNamespace(
            size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224},
            image_mean=(.5, .5, .5), image_std=(.25, .25, .25))))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); images = root / "images"; images.mkdir()
            for index in range(3):
                Image.new("RGB", (360, 420), (index * 70, 90, 120)).save(images / f"{index}.png")
            for version in (13, 14, 15):
                source = root / f"model_{version}.pt"; torch.save(checkpoint(version), source)
                out = root / str(version)
                argv = ["predict_v15.py", "--checkpoint", str(source), "--model-dir", "official",
                    "--test-dir", str(images), "--output", str(out / "pred_results.csv"),
                    "--zip-output", str(out / "pred_results.zip"), "--device", "cpu",
                    "--expected-rows", "3", "--workers", "0", "--batch-size", "2"]
                with patch.object(sys, "argv", argv), patch.object(prediction, "build_classifier_from_checkpoint", return_value=(TinyPredictor(), prep)), redirect_stdout(io.StringIO()):
                    prediction.main()
                prediction.validate_submission(out / "pred_results.csv", out / "pred_results.zip", 3, [f"{n}.png" for n in range(3)])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
