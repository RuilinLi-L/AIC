import unittest

import numpy as np
import torch

from evaluate_tta_v6 import _search_mix_macro, strict_stratified_folds


class EvaluateTTAV6Tests(unittest.TestCase):
    def test_five_folds_include_tail_classes_without_rejection(self):
        labels = np.asarray([0] * 20 + [1] * 3 + [2], dtype=np.int64)
        folds = strict_stratified_folds(labels, folds=5, seed=2026)
        self.assertEqual(set(folds.tolist()), set(range(5)))
        self.assertEqual(len(folds), len(labels))

    def test_macro_search_prefers_correct_original_and_tie_breaks(self):
        labels = np.asarray([0, 1, 2] * 5, dtype=np.int64)
        original = np.full((len(labels), 3), -5.0, dtype=np.float32)
        hflip = np.full_like(original, -5.0)
        original[np.arange(len(labels)), labels] = 5.0
        hflip[np.arange(len(labels)), (labels + 1) % 3] = 5.0
        folds = strict_stratified_folds(labels, folds=5, seed=2026)
        result = _search_mix_macro(
            original,
            hflip,
            labels,
            folds,
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.5], dtype=np.float32),
            torch.device("cpu"),
        )
        self.assertEqual(result["original_weight"], 1.0)
        self.assertEqual(result["alpha"], 0.0)
        self.assertGreater(result["cv_macro_accuracy"], 0.99)

    def test_identity_views_choose_higher_original_weight(self):
        labels = np.asarray([0, 1] * 5, dtype=np.int64)
        logits = np.eye(2, dtype=np.float32)[labels] * 8.0
        folds = strict_stratified_folds(labels, folds=5, seed=7)
        result = _search_mix_macro(
            logits,
            logits.copy(),
            labels,
            folds,
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            torch.device("cpu"),
        )
        self.assertEqual(result["original_weight"], 1.0)

    def test_macro_accuracy_wins_over_higher_micro_accuracy(self):
        labels = np.asarray([0] * 20 + [1] * 2, dtype=np.int64)
        original = np.zeros((len(labels), 2), dtype=np.float32)
        original[:, 0] = 5.0  # 20/22 micro, but only 0.50 macro accuracy.
        hflip = np.zeros_like(original)
        hflip[:14, 0] = 5.0
        hflip[:14, 1] = -5.0
        hflip[14:20, 0] = -5.0
        hflip[14:20, 1] = 5.0
        hflip[20:, 0] = -5.0
        hflip[20:, 1] = 5.0  # 16/22 micro, but 0.85 macro accuracy.
        folds = strict_stratified_folds(labels, folds=5, seed=2026)
        result = _search_mix_macro(
            original,
            hflip,
            labels,
            folds,
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            torch.device("cpu"),
        )
        self.assertEqual(result["original_weight"], 0.0)
        self.assertAlmostEqual(result["cv_macro_accuracy"], 0.85, places=6)


if __name__ == "__main__":
    unittest.main()
