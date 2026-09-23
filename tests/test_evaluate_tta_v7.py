import unittest

import numpy as np
import torch

from evaluate_tta_v6 import strict_stratified_folds
from evaluate_tta_v7 import greedy_view_search
from v7_views import VIEW_NAMES, normalize_view_weights


class EvaluateTTAV7Tests(unittest.TestCase):
    def test_weights_are_nonnegative_and_normalized(self):
        normalized = normalize_view_weights({"center": 0.3, "hflip": 0.6, "zoom256": 0.1})
        self.assertAlmostEqual(sum(normalized.values()), 1.0)
        self.assertTrue(all(value >= 0.0 for value in normalized.values()))
        with self.assertRaises(ValueError):
            normalize_view_weights({"center": 0.0})

    def test_greedy_search_only_keeps_improving_view(self):
        labels = np.asarray([0, 1, 2] * 5, dtype=np.int64)
        wrong = np.full((len(labels), 3), -4.0, dtype=np.float32)
        wrong[np.arange(len(labels)), (labels + 1) % 3] = 4.0
        correct = np.full_like(wrong, -4.0)
        correct[np.arange(len(labels)), labels] = 4.0
        views = {
            "center": wrong,
            "hflip": correct,
            "zoom256": wrong.copy(),
            "zoom256_hflip": wrong.copy(),
        }
        folds = strict_stratified_folds(labels, folds=5, seed=2026)
        result = greedy_view_search(views, labels, folds, torch.device("cpu"))
        self.assertGreater(result["cv_macro_accuracy"], 0.99)
        self.assertIn("hflip", result["view_weights"])
        self.assertAlmostEqual(sum(result["view_weights"].values()), 1.0)
        self.assertEqual(tuple(views), VIEW_NAMES)


if __name__ == "__main__":
    unittest.main()
