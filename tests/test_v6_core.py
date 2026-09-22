import unittest

import numpy as np
import torch
import torch.nn.functional as F

from v6_core import (
    SelectiveFeatureQueue,
    balanced_softmax_cross_entropy,
    build_repair_plan,
    capped_stratified_split,
    cross_fitted_visual_signals,
    fused_reliability,
    project_conflicting_gradients,
)


class V6CoreTests(unittest.TestCase):
    def test_tail_safe_split_keeps_training_rows(self):
        counts = [1, 2, 3, 4, 25, 300]
        rows = []
        for label, count in enumerate(counts):
            rows.extend((f"{label}_{index}.jpg", label, str(label)) for index in range(count))
        train, validation = capped_stratified_split(rows, seed=2026)
        train_labels = np.asarray([rows[index][1] for index in train])
        val_labels = np.asarray([rows[index][1] for index in validation])
        expected_validation = [0, 1, 1, 2, 2, 16]
        for label, count in enumerate(counts):
            self.assertEqual(int((val_labels == label).sum()), expected_validation[label])
            self.assertGreaterEqual(int((train_labels == label).sum()), 1)

    def test_cross_fitted_prototype_does_not_score_with_self(self):
        features = np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=np.float32,
        )
        views = np.repeat(features[None, :, :], 4, axis=0)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        signals = cross_fitted_visual_signals(
            views,
            labels,
            list(range(4)),
            num_classes=2,
            folds=5,
            keep_fraction=1.0,
            iterations=1,
        )
        # Each two-row class uses leave-one-out; the only remaining same-label
        # feature is the exact opposite of the held-out row.
        self.assertTrue(np.all(signals.label_similarity < -0.99))

    def test_synthetic_noisy_rows_receive_lower_quality(self):
        clean_per_class = 12
        true_labels = np.repeat(np.arange(2), clean_per_class)
        observed = true_labels.copy()
        observed[[1, clean_per_class + 1]] = observed[[clean_per_class + 1, 1]]
        noisy = observed != true_labels
        base = np.eye(2, dtype=np.float32)[true_labels]
        views = np.repeat(base[None, :, :], 4, axis=0)
        signals = cross_fitted_visual_signals(
            views,
            observed,
            list(range(len(observed))),
            num_classes=2,
            folds=5,
            keep_fraction=0.70,
            iterations=2,
        )
        warmup_loss = np.where(noisy, 3.0, 0.1).astype(np.float32)
        stability = np.ones(len(observed), dtype=np.float32)
        quality = fused_reliability(
            observed,
            list(range(len(observed))),
            signals.label_margin,
            signals.view_agreement,
            warmup_loss,
            stability,
        )
        self.assertLess(float(quality[noisy].mean()), float(quality[~noisy].mean()))
        self.assertGreaterEqual(float(quality.min()), 0.05)

    def test_balanced_softmax_with_uniform_prior_equals_ce(self):
        logits = torch.tensor([[2.0, -1.0], [0.2, 0.8]])
        labels = torch.tensor([0, 1])
        prior = torch.tensor([0.5, 0.5])
        expected = F.cross_entropy(logits, labels, reduction="none")
        actual = balanced_softmax_cross_entropy(logits, labels, prior)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_negative_gradient_is_projected_to_non_conflicting(self):
        trusted = [torch.tensor([1.0, 0.0]), torch.tensor([0.5])]
        uncertain = [torch.tensor([-2.0, 1.0]), torch.tensor([-1.0])]
        projected, changed = project_conflicting_gradients(trusted, uncertain)
        self.assertTrue(changed)
        dot = sum((a * b).sum() for a, b in zip(trusted, projected))
        self.assertGreaterEqual(float(dot), -1e-6)

    def test_selective_queue_filters_and_caps_each_class(self):
        queue = SelectiveFeatureQueue(2, capacity=2, dim=2, device=torch.device("cpu"))
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 0, 1])
        accepted = torch.tensor([True, True, True, False])
        queue.enqueue(features, labels, accepted)
        queued, queued_labels = queue.flattened()
        self.assertEqual(len(queued), 2)
        self.assertTrue(torch.equal(queued_labels, torch.zeros(2, dtype=torch.long)))
        self.assertEqual(int(queue.pointer[0]), 3)
        self.assertEqual(int(queue.pointer[1]), 0)

    def test_repair_requires_consensus_and_obeys_strict_class_cap(self):
        labels = np.zeros(20, dtype=np.int64)
        prototype_label = np.ones(20, dtype=np.int64)
        margins = np.linspace(0.1, 1.0, 20, dtype=np.float32)
        temporal = np.tile(np.asarray([[0.05, 0.95]], dtype=np.float32), (20, 1))
        plan = build_repair_plan(
            labels,
            list(range(20)),
            np.full(20, 0.2, dtype=np.float32),
            prototype_label,
            margins,
            np.asarray([0.0], dtype=np.float32),
            temporal,
            np.ones(20, dtype=np.int64),
            confidence_threshold=0.80,
            per_class_fraction=0.15,
        )
        self.assertEqual(int(plan.sum()), 3)
        no_consensus = build_repair_plan(
            labels,
            list(range(20)),
            np.full(20, 0.2, dtype=np.float32),
            prototype_label,
            margins,
            np.asarray([0.0], dtype=np.float32),
            temporal,
            np.zeros(20, dtype=np.int64),
        )
        self.assertFalse(no_consensus.any())

        tail_plan = build_repair_plan(
            labels[:6],
            list(range(6)),
            np.full(6, 0.2, dtype=np.float32),
            prototype_label[:6],
            margins[:6],
            np.asarray([0.0], dtype=np.float32),
            temporal[:6],
            np.ones(6, dtype=np.int64),
        )
        self.assertEqual(int(tail_plan.sum()), 0)


if __name__ == "__main__":
    unittest.main()
