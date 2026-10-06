"""V18 evaluation provenance, immutable epoch weights and cache recovery."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate_tta_v18 as evaluation
from select_v14 import read_evaluation
from test_v18_delivery import checkpoint
from v10_evaluation import StrictEvaluation


class V18EvaluationTests(unittest.TestCase):
    def test_calibration_preserves_epoch_state_and_records_all_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "epoch_18.pt"
            original = checkpoint(); torch.save(original, source)
            digest = evaluation.sha256_file(source)
            logits = np.array([[2., 0.], [0., 2.]], dtype=np.float32)
            bias = np.array([.125, -.125], dtype=np.float32)
            summary = {"method": "v10_strict_outer5_epoch_inner4_fixed_tta_bias",
                       "selected": {"epoch": 18, "family": "four_view", "alpha": .5,
                                    "view_weights": {name: .25 for name in evaluation.VIEW_NAMES}},
                       "conditions": {name: {"outer_metrics": {"macro_accuracy": 1., "macro_nll": .1,
                                                               "tail_accuracy": 1., "micro_accuracy": 1.}}
                                      for name in evaluation.CONDITIONS}}
            result = StrictEvaluation(summary, {name: logits for name in evaluation.CONDITIONS}, logits + bias, bias)
            output = root / "evaluation"; target = output / "model.pt"
            cache = SimpleNamespace(device=torch.device("cpu"), manifest=SimpleNamespace(signature="dataset"),
                                    hashes={18: digest})
            value = evaluation.write_result(result, output_dir=output, output_checkpoint=target,
                paths={18: source}, checkpoints={18: original}, cache=cache,
                row_keys=["0000/a.jpg", "0001/b.jpg"], labels=np.array([0, 1]))
            loaded = read_evaluation(output)["checkpoint"]
            self.assertEqual(value["format_version"], 18)
            self.assertEqual(value["recipe"], "resolution384")
            self.assertEqual(value["source_epoch_checkpoint"], str(source.resolve()))
            self.assertEqual(value["epoch_checkpoint_sha256"], {"18": digest})
            self.assertEqual(value["calibrated_checkpoint_sha256"], evaluation.sha256_file(target))
            self.assertEqual(value["class_names"], original["class_names"])
            self.assertEqual(value["base_model_identity"], original["config"]["base_model_identity"])
            self.assertEqual(loaded["calibration"]["selected_epoch"], 18)
            self.assertEqual(loaded["calibration"]["image_size"], 384)
            self.assertEqual(loaded["calibration"]["view_weights"], {name: .25 for name in evaluation.VIEW_NAMES})
            self.assertEqual(evaluation.sha256_file(source), digest)
            for key, tensor in original["model"].items():
                self.assertTrue(torch.equal(tensor, loaded["model"][key]), key)
            np.testing.assert_array_equal(loaded["class_bias"].numpy(), bias)
            np.testing.assert_array_equal(original["class_bias"].numpy(), np.zeros(2))
            for condition in evaluation.CONDITIONS:
                np.testing.assert_array_equal(np.load(output / f"val_oof_{condition}.npy"), logits)

    def test_corrupted_or_incomplete_training_logits_recompute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "epoch_01.pt"; source.write_bytes(b"checkpoint")
            path = root / "epoch_01_logits.npz"
            metadata = {"center": np.eye(2, dtype=np.float32), "hflip": np.eye(2, dtype=np.float32),
                        "labels": np.array([0, 1]), "row_indices": np.array([0, 1]),
                        "precision": np.asarray("fp32"), "format_version": np.asarray(18),
                        "checkpoint_sha256": np.asarray(evaluation.sha256_file(source)),
                        "image_size": np.asarray(384), "zoom_shortest_edge": np.asarray(439), "lora_rank": np.asarray(16)}
            def cache():
                return evaluation.LogitCache(output_dir=root / "evaluation", paths={1: source},
                    checkpoints={1: checkpoint(epoch=1)},
                    manifest=SimpleNamespace(signature="dataset", class_names=["0000", "0001"]),
                    indices=[0, 1], labels=np.array([0, 1]), model_dir="official", device=torch.device("cpu"),
                    batch_size=2, workers=0, force=False)
            for name in ("missing", "nonfinite", "shape", "corrupt"):
                value = deepcopy(metadata)
                if name == "missing": del value["labels"]
                if name == "nonfinite": value["center"][0, 0] = np.nan
                if name == "shape": value["hflip"] = np.ones((1, 2), dtype=np.float32)
                np.savez(path, **value)
                if name == "corrupt": path.write_bytes(b"unfinished cache")
                with self.subTest(name=name), patch.object(evaluation, "build_classifier_from_checkpoint",
                        side_effect=RuntimeError("checkpoint inference required")), self.assertRaisesRegex(RuntimeError, "checkpoint inference"):
                    cache().get_views(1, "native", False)

    def test_training_cache_requires_its_own_size_rank_and_source_identity(self):
        for recipe in ("resolution384", "rank32", "matched_control"):
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root / "epoch_01.pt"; source.write_bytes(b"checkpoint")
                current = checkpoint(recipe, 1)
                labels = np.array([0, 1]); logits = np.eye(2, dtype=np.float32)
                metadata = {"center": logits, "hflip": logits, "labels": labels,
                    "row_indices": labels, "precision": np.asarray("fp32"), "format_version": np.asarray(18),
                    "checkpoint_sha256": np.asarray(evaluation.sha256_file(source)),
                    **{field: np.asarray(current["config"][field]) for field in ("image_size", "zoom_shortest_edge", "lora_rank")}}
                def cache():
                    return evaluation.LogitCache(output_dir=root / "evaluation", paths={1: source},
                        checkpoints={1: current}, manifest=SimpleNamespace(signature="dataset", class_names=["0000", "0001"]),
                        indices=[0, 1], labels=labels, model_dir="official", device=torch.device("cpu"),
                        batch_size=2, workers=0, force=False)
                target = root / "epoch_01_logits.npz"; np.savez(target, **metadata)
                with patch.object(evaluation, "build_classifier_from_checkpoint", side_effect=RuntimeError("recompute")):
                    np.testing.assert_array_equal(cache().get_views(1, "native", False)["center"], logits)
                    for field in ("image_size", "zoom_shortest_edge", "lora_rank", "checkpoint_sha256", "precision"):
                        bad = dict(metadata); bad[field] = np.asarray(1 if field not in ("checkpoint_sha256", "precision") else "changed")
                        np.savez(target, **bad)
                        with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "recompute"):
                            cache().get_views(1, "native", False)
                    bad = dict(metadata); del bad["image_size"]; np.savez(target, **bad)
                    with self.assertRaisesRegex(RuntimeError, "recompute"): cache().get_views(1, "native", False)

    def test_complete_24_epochs_for_all_recipes_keep_actual_microbatch_frozen(self):
        manifest = SimpleNamespace(signature="dataset", validation_indices=[0, 1],
            rows=[("a", 0, "0000"), ("b", 1, "0001")], class_names=["0000", "0001"])
        for recipe, branch in (("resolution384", "standard"), ("rank32", "standard"),
                               ("resolution384", "matched_control"), ("matched_control", "matched_control")):
            with self.subTest(recipe=recipe, branch=branch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); epochs = root / "validation_epochs"; epochs.mkdir()
                for epoch in range(1, 25):
                    torch.save(checkpoint(recipe, epoch, branch=branch), epochs / f"epoch_{epoch:02d}.pt")
                paths, _ = evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))
                self.assertEqual(len(paths), 24)
                changed = checkpoint(recipe, 24, branch=branch)
                changed["config"].update(batch_size=128 if branch == "standard" else 256,
                    gradient_accumulation=2 if branch == "standard" else 1)
                torch.save(changed, paths[24])
                with self.assertRaisesRegex(ValueError, "configurations"):
                    evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))

    def test_evaluation_does_not_overwrite_epoch_or_result_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); epochs = root / "validation_epochs"; epochs.mkdir()
            torch.save(checkpoint(epoch=1), epochs / "epoch_01.pt")
            base = ["evaluate_tta_v18.py", "--run-dir", str(root), "--train-dir", "train",
                    "--data-manifest", "manifest.json", "--device", "cpu"]
            cases = [["--output-dir", str(epochs)],
                     ["--output-dir", str(root / "eval"), "--output-checkpoint", str(epochs / "epoch_01.pt")],
                     ["--output-dir", str(root / "eval"), "--output-checkpoint", str(root / "eval" / "strict_eval.json")]]
            for args in cases:
                with self.subTest(args=args), patch.object(sys, "argv", base + args), \
                        patch.object(evaluation, "load_manifest_dataset") as load, self.assertRaises(ValueError):
                    evaluation.main()
                load.assert_not_called()

    def test_epoch_validation_rejects_changed_resolution_state_and_dataset(self):
        manifest = SimpleNamespace(signature="dataset", validation_indices=[0, 1],
            rows=[("a", 0, "0000"), ("b", 1, "0001")], class_names=["0000", "0001"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); epochs = root / "validation_epochs"; epochs.mkdir()
            for epoch in range(1, 25): torch.save(checkpoint(epoch=epoch), epochs / f"epoch_{epoch:02d}.pt")
            target = epochs / "epoch_24.pt"
            for field in ("dataset", "mlp", "resolution", "stage", "schedule"):
                value = checkpoint(epoch=24)
                if field == "dataset": value["dataset_signature"] = "changed"
                elif field == "mlp": del value["model"]["clip.vision_model.encoder.layers.11.mlp.fc2.lora_b"]
                elif field == "resolution": value["config"]["image_size"] = 224
                elif field == "stage": value["training_stage"] = "refit"
                else: value["config"]["schedule_epochs"] = 18
                torch.save(value, target)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
