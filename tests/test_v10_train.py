"""Focused V10 training checks without running the CLIP model."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from robust_clip import PairedFolderImageDataset
from train_v5 import clone_trainable
from train_v10 import (
    FORMAT_VERSION,
    _restore_rng,
    load_resume,
    load_v9_selection,
    neighbor_cache_path,
    resolve_v9_base,
    save_epoch_logits,
    save_resume,
    validate_neighbor_evidence,
)
from v10_views import RobustPairedFolderImageDataset, degrade_view_b
from v6_core import SelectiveFeatureQueue
from v8_neighbors import load_or_create_neighbor_evidence


class _PredictableRng:
    def __init__(self, decisions: list[float], values: list[int]):
        self.decisions = iter(decisions)
        self.values = iter(values)

    def random(self) -> float:
        return next(self.decisions)

    def randint(self, _low: int, _high: int) -> int:
        return next(self.values)


class V10TrainingTests(unittest.TestCase):
    def test_quality_degradation_is_independent_and_never_upscales(self):
        image = Image.new("RGB", (800, 400), (120, 90, 30))
        reduced = degrade_view_b(image, _PredictableRng([0.0, 1.0], [384]))
        self.assertEqual(reduced.size, (384, 192))
        small = Image.new("RGB", (200, 100), (120, 90, 30))
        unchanged = degrade_view_b(small, _PredictableRng([0.0, 1.0], [320]))
        self.assertEqual(unchanged.size, small.size)
        jpeg = degrade_view_b(image, _PredictableRng([1.0, 0.0], [75]))
        self.assertEqual(jpeg.size, image.size)
        self.assertEqual(jpeg.mode, "RGB")

    def test_view_a_matches_v9_light_and_pair_repeats_under_same_seed(self):
        processor = SimpleNamespace(image_processor=SimpleNamespace(
            image_mean=(0.5, 0.5, 0.5), image_std=(0.25, 0.25, 0.25),
            size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224},
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            yy, xx = np.indices((480, 800))
            pixels = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
            Image.fromarray(pixels).save(path)
            rows = [(str(path), 0, "class")]
            original = PairedFolderImageDataset(rows, [0], processor, augment="light")
            robust = RobustPairedFolderImageDataset(rows, [0], processor)
            random.seed(19)
            torch.manual_seed(19)
            old = original[0]
            random.seed(19)
            torch.manual_seed(19)
            first = robust[0]
            random.seed(19)
            torch.manual_seed(19)
            second = robust[0]
            self.assertTrue(torch.equal(first["pixel_values_a"], old["pixel_values_a"]))
            self.assertTrue(torch.equal(first["pixel_values_a"], second["pixel_values_a"]))
            self.assertTrue(torch.equal(first["pixel_values_b"], second["pixel_values_b"]))

    def test_selection_requires_finished_supported_v9_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            with self.assertRaisesRegex(FileNotFoundError, "not ready"):
                load_v9_selection(path)
            path.write_text(json.dumps({
                "selected_run": "v9_soft_teacher",
                "selected_path": "/data/outputs/v9_soft_teacher",
            }), encoding="utf-8")
            self.assertEqual(load_v9_selection(path), "v9_soft_teacher")
            path.write_text(json.dumps({
                "selected_run": "v9_soft_teacher",
                "selected_path": "/data/outputs/v9_coverage",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_v9_selection(path)
            self.assertEqual(FORMAT_VERSION, 10)

    def test_explicit_finished_v9_base_does_not_need_selection(self):
        self.assertEqual(resolve_v9_base(None, "v9_coverage"), ("v9_coverage", "explicit"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps({
                "selected_run": "v9_soft_teacher",
                "selected_path": "/data/outputs/v9_soft_teacher",
            }), encoding="utf-8")
            self.assertEqual(resolve_v9_base(str(path), None), ("v9_soft_teacher", "selection_json"))
            with self.assertRaisesRegex(ValueError, "either"):
                resolve_v9_base(str(path), "v9_coverage")
        with self.assertRaisesRegex(ValueError, "required"):
            resolve_v9_base(None, None)

    def test_external_neighbor_cache_is_read_only_and_checks_signature(self):
        rng = np.random.default_rng(13)
        features = rng.normal(size=(4, 12, 4)).astype(np.float32)
        labels = np.asarray([0, 1] * 6)
        hashes = [str(index) for index in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "v9_neighbor_evidence.npz"
            output = root / "v10"
            load_or_create_neighbor_evidence(
                external, "dataset-one", features, labels, hashes,
                list(range(10)), [10, 11], 2, torch.device("cpu"),
            )
            before = external.stat().st_mtime_ns
            selected = neighbor_cache_path(str(external), output)
            self.assertEqual(selected, external)
            evidence = load_or_create_neighbor_evidence(
                selected, "dataset-one", features, labels, hashes,
                list(range(10)), [10, 11], 2, torch.device("cpu"),
            )
            validate_neighbor_evidence(evidence, len(labels))
            self.assertEqual(before, external.stat().st_mtime_ns)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_or_create_neighbor_evidence(
                    selected, "dataset-two", features, labels, hashes,
                    list(range(10)), [10, 11], 2, torch.device("cpu"),
                )
            self.assertEqual(neighbor_cache_path(str(root / "missing.npz"), output), output / "neighbor_evidence.npz")

    def test_epoch_logits_and_resume_round_trip(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Parameter(torch.eye(2))

        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        model.classifier.square().sum().backward()
        optimizer.step()
        scheduler.step()
        queue = SelectiveFeatureQueue(2, capacity=2, dim=2, device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            epoch_dir = root / "validation_epochs"
            epoch_dir.mkdir()
            logits = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
            save_epoch_logits(epoch_dir, 2, logits, logits + 0.1, np.asarray([0, 1]), [7, 9])
            with np.load(epoch_dir / "epoch_02_logits.npz") as saved:
                np.testing.assert_array_equal(saved["row_indices"], [7, 9])
                np.testing.assert_allclose(saved["center"], logits)
            random.seed(77)
            np.random.seed(77)
            torch.manual_seed(77)
            resume_file = root / "resume.pt"
            config = {"version": "v10", "soft_teacher": True}
            save_resume(
                resume_file, stage="validation", epoch=2, classifier=model,
                optimizer=optimizer, scheduler=scheduler, ema=clone_trainable(model),
                queue=queue, arrays={"quality": np.asarray([0.7, 0.25], dtype=np.float32)},
                optimizer_step=1, history=[{"epoch": 1}],
                best_state=clone_trainable(model), best_record={"epoch": 1},
                selected_epoch=None, config=config, dataset_digest="signature",
            )
            expected = (random.random(), float(np.random.random()), torch.rand(1))
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            state = load_resume(str(resume_file), config, "signature", torch.device("cpu"))
            _restore_rng(state)
            actual = (random.random(), float(np.random.random()), torch.rand(1))
            self.assertEqual(state["epoch"], 2)
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            self.assertTrue(torch.equal(actual[2], expected[2]))
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_resume(str(resume_file), {"version": "v9"}, "signature", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
