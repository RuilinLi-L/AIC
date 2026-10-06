"""V10 epoch and calibration choices must exclude outer held-out evidence."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from v10_evaluation import fit_fixed_tta, select_epoch, strict_nested_evaluation
from evaluate_tta_v10 import LogitCache, stress_image
from PIL import Image


class V10EvaluationTests(unittest.TestCase):
    def test_v13_accepts_complete_24_epoch_sequence(self):
        labels = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
        native = np.asarray([[3.0, -1.0]] * 5 + [[-1.0, 3.0]] * 5, dtype=np.float32)

        def get_views(epoch: int, condition: str, four: bool):
            views = {"center": native, "hflip": native}
            if four:
                views.update({"zoom256": native, "zoom256_hflip": native})
            return views

        result = strict_nested_evaluation(
            get_views, list(range(1, 25)), labels, np.asarray([5, 5]),
            device=torch.device("cpu"),
        )
        self.assertEqual(result.summary["selected"]["epoch"], 1)
        with self.assertRaisesRegex(ValueError, "12 or 24 consecutive"):
            strict_nested_evaluation(
                get_views, list(range(1, 24)), labels, np.asarray([5, 5]),
                device=torch.device("cpu"),
            )

    def test_epoch_and_tta_choice_ignore_rows_outside_fit_indices(self):
        labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
        fit_indices = np.asarray([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int64)
        base = np.asarray([[4.0, -1.0]] * 5 + [[-1.0, 4.0]] * 5, dtype=np.float32)
        weaker = base.copy()
        weaker[fit_indices[:2]] = weaker[fit_indices[:2], ::-1]
        epochs = {1: weaker.copy(), 2: base.copy()}
        before_epoch, _ = select_epoch(epochs, labels, fit_indices)
        views = {name: base.copy() for name in ("center", "hflip", "zoom256", "zoom256_hflip")}
        before_tta = fit_fixed_tta(views, labels, fit_indices, folds=4, device=torch.device("cpu"))

        changed_labels = labels.copy()
        changed_labels[[4, 9]] = 1 - changed_labels[[4, 9]]
        for array in epochs.values():
            array[[4, 9]] = 1000.0 * array[[4, 9], ::-1]
        for array in views.values():
            array[[4, 9]] = 1000.0 * array[[4, 9], ::-1]
        after_epoch, _ = select_epoch(epochs, changed_labels, fit_indices)
        after_tta = fit_fixed_tta(views, changed_labels, fit_indices, folds=4, device=torch.device("cpu"))
        self.assertEqual(before_epoch, after_epoch)
        self.assertEqual(before_tta["family"], after_tta["family"])
        self.assertEqual(before_tta["alpha"], after_tta["alpha"])

    def test_stress_logits_cannot_select_epoch_or_calibration(self):
        labels = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
        native = np.asarray([[3.0, -1.0]] * 5 + [[-1.0, 3.0]] * 5, dtype=np.float32)

        def run(stress_value: float):
            def get_views(epoch: int, condition: str, four: bool):
                value = native if condition == "native" else np.full_like(native, stress_value)
                result = {"center": value.copy(), "hflip": value.copy()}
                if four:
                    result.update({"zoom256": value.copy(), "zoom256_hflip": value.copy()})
                return result
            return strict_nested_evaluation(
                get_views, list(range(1, 13)), labels, np.asarray([5, 5]),
                device=torch.device("cpu"),
            )

        first = run(-20.0)
        second = run(20.0)
        self.assertEqual(first.summary["selected"], second.summary["selected"])
        self.assertEqual(
            [(fold["selected_epoch"], fold["selected_family"], fold["alpha"])
             for fold in first.summary["outer_folds"]],
            [(fold["selected_epoch"], fold["selected_family"], fold["alpha"])
             for fold in second.summary["outer_folds"]],
        )
        np.testing.assert_array_equal(first.oof_by_condition["native"], second.oof_by_condition["native"])

    def test_fixed_stress_does_not_upscale_and_is_deterministic(self):
        small = Image.new("RGB", (200, 100), (110, 60, 20))
        self.assertEqual(stress_image(small, "resize384").size, (200, 100))
        large = Image.new("RGB", (800, 400), (110, 60, 20))
        self.assertEqual(stress_image(large, "resize384").size, (384, 192))
        first = stress_image(large, "combined")
        second = stress_image(large, "combined")
        self.assertEqual(first.size, (384, 192))
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_epoch_logits_reuse_checks_row_order(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            epoch_path = root / "epoch_01.pt"
            epoch_path.write_bytes(b"checkpoint")
            labels = np.asarray([0, 1], dtype=np.int64)
            center = np.asarray([[3.0, -1.0], [-1.0, 3.0]], dtype=np.float32)
            np.savez_compressed(
                root / "epoch_01_logits.npz", center=center, hflip=center,
                labels=labels, row_indices=np.asarray([7, 9], dtype=np.int64),
            )
            cache = LogitCache(
                output_dir=root / "evaluation", paths={1: epoch_path},
                checkpoints={1: {"format_version": 10}},
                manifest=SimpleNamespace(signature="dataset", class_names=["0", "1"]),
                indices=[7, 9], labels=labels, model_dir=str(root),
                device=torch.device("cpu"), batch_size=2, workers=0, force=False,
            )
            views = cache.get_views(1, "native", False)
            np.testing.assert_array_equal(views["center"], center)
            self.assertIsNone(cache.classifier)
            bad = LogitCache(
                output_dir=root / "other", paths={1: epoch_path},
                checkpoints={1: {"format_version": 10}},
                manifest=SimpleNamespace(signature="dataset", class_names=["0", "1"]),
                indices=[9, 7], labels=labels, model_dir=str(root),
                device=torch.device("cpu"), batch_size=2, workers=0, force=False,
            )
            with patch("evaluate_tta_v10.build_classifier_from_checkpoint", side_effect=ValueError("fallback")):
                with self.assertRaisesRegex(ValueError, "fallback"):
                    bad.get_views(1, "native", False)


if __name__ == "__main__":
    unittest.main()
