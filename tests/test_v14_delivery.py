"""V14's independent evaluation, frozen selection and submission contracts."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

import evaluate_tta_v14 as evaluation
import predict_v14 as prediction
import select_v14 as selection
from v13_core import file_sha256
from v14_core import recipe_config
from v11_resolution import resolution_processor, FourViewTestDataset


BASE_IDENTITY = {"config.json": "official", "model.safetensors": "official-weights"}


def processor():
    return resolution_processor(SimpleNamespace(image_processor=SimpleNamespace(
        size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224},
        image_mean=(.5, .5, .5), image_std=(.25, .25, .25))))


def config(recipe):
    return {**recipe_config(recipe), "stage": "validate", "image_size": 320,
            "zoom_shortest_edge": 366, "interpolate_pos_encoding": True,
            "conflict_policy": "partial", "sampler": "repeat-factor",
            "base_model_identity": BASE_IDENTITY, "feature_cache": "frozen.npy",
            "workers": 16, "prefetch_factor": 2, "eval_batch_size": 256,
            "batch_size": 256, "gradient_accumulation": 1, "precision": "fp32"}


def checkpoint(version, epoch=18):
    recipe = "expanded" if version == 13 else "expanded_robust"
    return {"format_version": version, "training_stage": "validation", "selected_epoch": epoch,
            "dataset_signature": "dataset", "class_names": ["0000", "0001"],
            "config": config(recipe), "validation": {"indices": [0, 1]},
            "model": {"weight": torch.tensor([1., 2.])}, "class_bias": torch.zeros(2),
            "calibration": {"view_weights": {"center": .5, "hflip": .5}, "alpha": 0., "precision": "fp32"}}


def conditions(gain=0., tail=0., stress=0.):
    return {name: {"outer_metrics": {"macro_accuracy": .7 + (gain if name == "native" else stress),
                                    "tail_accuracy": .6 + tail, "macro_nll": 1.2}}
            for name in selection.CONDITIONS}


def evaluation_fixture(root, version, gain=0., tail=0., stress=0.):
    name = "baseline" if version == 13 else "candidate"
    source = root / f"{name}_training" / "validation_epochs" / "epoch_18.pt"
    source.parent.mkdir(parents=True)
    ckpt = checkpoint(version)
    torch.save(ckpt, source)
    out = root / f"{name}_evaluation"
    out.mkdir()
    calibrated = out / "model.pt"
    torch.save(ckpt, calibrated)
    summary = {"dataset_signature": "dataset", "calibrated_checkpoint": str(calibrated),
               "calibrated_checkpoint_sha256": file_sha256(calibrated),
               "checkpoint_sha256": file_sha256(source), "source_epoch_checkpoint": str(source),
               "selected": {"epoch": 18, "view_weights": ckpt["calibration"]["view_weights"], "alpha": 0.},
               "precision": "fp32", "stress_version": evaluation.STRESS_VERSION, "image_size": 320,
               "outer_fold_ids": [0, 1], "conditions": conditions(gain, tail, stress)}
    (out / "strict_eval.json").write_text(json.dumps(summary))
    return out


class TinyPredictor(torch.nn.Module):
    def forward(self, pixels, _=None):
        values = pixels.mean(dim=(1, 2, 3))
        return torch.stack((values, -values), dim=1), None


class V14DeliveryTests(unittest.TestCase):
    def test_v13_evaluation_refuses_to_overwrite_original_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); epochs = root / "validation_epochs"; epochs.mkdir()
            source = epochs / "epoch_01.pt"; torch.save(checkpoint(13, 1), source)
            before = source.read_bytes()
            argv = ["evaluate_tta_v14.py", "--run-dir", str(root), "--output-dir", str(root),
                    "--train-dir", "unused", "--data-manifest", "unused"]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "separate output"):
                evaluation.main()
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse((root / "model.pt").exists())

    def test_complete_epoch_files_allow_runtime_changes_but_bind_science_and_format(self):
        manifest = SimpleNamespace(signature="dataset", validation_indices=[0, 1],
                                   rows=[("a", 0, "0000"), ("b", 1, "0001")], class_names=["0000", "0001"])
        for version in (13, 14):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); epochs = root / "validation_epochs"; epochs.mkdir()
                for epoch in range(1, 25):
                    payload = checkpoint(version, epoch)
                    if epoch > 1:
                        payload["config"].update(workers=4, prefetch_factor=1, eval_batch_size=64,
                                                 batch_size=64, gradient_accumulation=4, pin_memory=False)
                    torch.save(payload, epochs / f"epoch_{epoch:02d}.pt")
                paths, _ = evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))
                self.assertEqual(len(paths), 24)
                changed = torch.load(paths[24], weights_only=False)
                original = deepcopy(changed)
                changed["config"]["lora_lr"] *= 2
                torch.save(changed, paths[24])
                with self.assertRaisesRegex(ValueError, "training configurations"):
                    evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))
                changed = deepcopy(original); changed["format_version"] = 14 if version == 13 else 13
                torch.save(changed, paths[24])
                with self.assertRaises(ValueError):
                    evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))
                paths[24].unlink()
                with self.assertRaises(FileNotFoundError):
                    evaluation.validate_epoch_checkpoints(root, manifest, np.array([0, 1]))

    def test_v13_and_v14_training_logits_reuse_requires_hash_precision_format_and_rows(self):
        for version in (13, 14):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root / "epoch_01.pt"; source.write_bytes(b"checkpoint")
                logits = np.array([[3., 0.], [0., 3.]], dtype=np.float32)
                metadata = {"center": logits, "hflip": logits, "labels": np.array([0, 1]),
                            "row_indices": np.array([0, 1]), "precision": np.asarray("fp32"),
                            "format_version": np.asarray(version),
                            "checkpoint_sha256": np.asarray(file_sha256(source))}
                np.savez(root / "epoch_01_logits.npz", **metadata)
                manifest = SimpleNamespace(signature="dataset", class_names=["0000", "0001"])
                def cache():
                    return evaluation.LogitCache(output_dir=root / "evaluation", paths={1: source},
                        checkpoints={1: checkpoint(version, 1)}, manifest=manifest, indices=[0, 1],
                        labels=np.array([0, 1]), model_dir="official", device=torch.device("cpu"),
                        batch_size=2, workers=0, force=False)
                with patch.object(evaluation, "build_classifier_from_checkpoint", side_effect=RuntimeError("recompute")):
                    valid = cache()
                    np.testing.assert_array_equal(valid.get_views(1, "native", False)["center"], logits)
                    self.assertIsNone(valid.classifier)
                    for key, value in (("checkpoint_sha256", np.asarray("changed")),
                                       ("precision", np.asarray("bf16_autocast_fp32_stats")),
                                       ("format_version", np.asarray(14 if version == 13 else 13)),
                                       ("row_indices", np.array([1, 0]))):
                        np.savez(root / "epoch_01_logits.npz", **{**metadata, key: value})
                        with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "recompute"):
                            cache().get_views(1, "native", False)

    def test_eval_loader_defaults_to_unpinned_memory_on_cuda(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(evaluation, "DataLoader", side_effect=RuntimeError("captured")) as loader:
            with self.assertRaisesRegex(RuntimeError, "captured"):
                evaluation.collect_view_logits(None, processor(), [("a", 0, "0000")], [0], "native",
                    four=True, device=torch.device("cuda"), batch_size=1, workers=4, prefetch_factor=1)
            self.assertFalse(loader.call_args.kwargs["pin_memory"])
            self.assertEqual(loader.call_args.kwargs["prefetch_factor"], 1)

    def test_gate_boundaries_and_each_regression(self):
        baseline = {"root": "baseline", "summary": {"dataset_signature": "same", "conditions": conditions()}}
        def candidate(gain=.002, tail=-.003, stress=-.003):
            return {"root": "candidate", "summary": {"dataset_signature": "same", "conditions": conditions(gain, tail, stress)}}
        accepted = candidate()
        self.assertIs(selection.choose_candidate(baseline, accepted)[0], accepted)
        for rejected in (candidate(gain=.001999), candidate(tail=-.003001), candidate(stress=-.003001)):
            self.assertIs(selection.choose_candidate(baseline, rejected)[0], baseline)
        for condition in selection.CONDITIONS[1:]:
            rejected = candidate()
            rejected["summary"]["conditions"][condition]["outer_metrics"]["macro_accuracy"] = .69
            self.assertIs(selection.choose_candidate(baseline, rejected)[0], baseline)
        bad = candidate(); bad["summary"]["dataset_signature"] = "other"
        with self.assertRaisesRegex(ValueError, "different validation"):
            selection.choose_candidate(baseline, bad)

    def test_selection_cli_binds_separate_evaluation_source_and_freezes_candidate_or_fallback(self):
        for gain, winner in ((.003, "expanded_robust"), (.001, "expanded")):
            with self.subTest(winner=winner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                baseline = evaluation_fixture(root, 13)
                candidate = evaluation_fixture(root, 14, gain=gain)
                output = root / "selection.json"
                argv = ["select_v14.py", "--baseline", str(baseline), "--candidate", str(candidate),
                        "--model-dir", "official", "--output", str(output)]
                with patch.object(sys, "argv", argv), patch.object(selection, "model_identity", return_value=BASE_IDENTITY), redirect_stdout(io.StringIO()):
                    selection.main()
                selected = selection.load_selection(output, dataset_signature="dataset",
                    class_names=["0000", "0001"], base_identity=BASE_IDENTITY)
                self.assertEqual(selected["recipe"], winner)
                self.assertEqual(selected["schedule_epochs"], 24)
                self.assertEqual(selected["augmentation"], recipe_config(winner)["augmentation"])
                self.assertFalse((baseline / "validation_epochs").exists())
                for change in ("learning_rate", "augmentation", "bias", "calibration", "schedule", "format", "source_recipe"):
                    invalid = deepcopy(selected)
                    if change == "learning_rate": invalid["training_config"]["lora_lr"] *= 2
                    elif change == "augmentation": invalid["augmentation"] = "wrong"
                    elif change == "bias": invalid["class_bias"][0] = .5
                    elif change == "calibration": invalid["calibration"]["alpha"] = .5
                    elif change == "schedule": invalid["schedule_epochs"] = 18
                    elif change == "format": invalid["format_version"] = 13
                    else: invalid["recipe"] = "expanded" if winner == "expanded_robust" else "expanded_robust"
                    output.write_text(json.dumps(invalid))
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        selection.load_selection(output)
                output.write_text(json.dumps(selected))
                with self.assertRaisesRegex(ValueError, "base_model_identity"):
                    selection.load_selection(output, base_identity={"different": "weights"})
                Path(selected["source_checkpoint"]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checkpoint missing or changed"):
                    selection.load_selection(output)

    def test_prediction_view_subsets_match_legacy_logits_and_both_formats_make_valid_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); images = root / "images"; images.mkdir()
            for number in range(3):
                Image.new("RGB", (360, 420), (number * 70, 90, 120)).save(images / f"{number}.png")
            paths = sorted(images.glob("*.png")); prep = processor(); model = TinyPredictor()
            legacy = next(iter(DataLoader(FourViewTestDataset([str(p) for p in paths], prep), batch_size=3)))
            for weights, required in (({"center": .5, "hflip": .5}, {"center"}),
                                      ({"zoom256": .5, "zoom256_hflip": .5}, {"zoom256"}),
                                      ({name: .25 for name in prediction.VIEW_NAMES}, {"center", "zoom256"})):
                subset = prediction.RequiredViewTestDataset(paths, prep, weights)
                batch = next(iter(DataLoader(subset, batch_size=3)))
                self.assertEqual(set(batch) - {"path"}, required)
                actual = prediction._weighted_logits(model, batch, weights, torch.device("cpu"))
                expected = prediction._weighted_logits(model, legacy, weights, torch.device("cpu"))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for version in (13, 14):
                source = root / f"model_{version}.pt"; torch.save(checkpoint(version), source)
                out = root / str(version)
                argv = ["predict_v14.py", "--checkpoint", str(source), "--model-dir", "official",
                        "--test-dir", str(images), "--output", str(out / "pred_results.csv"),
                        "--zip-output", str(out / "pred_results.zip"), "--device", "cpu",
                        "--expected-rows", "3", "--workers", "0", "--batch-size", "2"]
                with patch.object(sys, "argv", argv), patch.object(prediction, "build_classifier_from_checkpoint", return_value=(model, prep)), \
                        patch.object(prediction, "DataLoader", wraps=DataLoader) as loader, redirect_stdout(io.StringIO()):
                    prediction.main()
                self.assertFalse(loader.call_args.kwargs["pin_memory"])
                prediction.validate_submission(out / "pred_results.csv", out / "pred_results.zip", 3, [f"{n}.png" for n in range(3)])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
