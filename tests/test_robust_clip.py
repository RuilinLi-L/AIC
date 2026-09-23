from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from robust_clip import (
    LoRALinear,
    PairedFolderImageDataset,
    RobustCLIPClassifier,
    classwise_quality,
    classwise_rank_scores,
    generalized_cross_entropy,
    inject_visual_lora,
    load_classifier_state,
    prototype_pseudo_targets,
    robust_visual_prototypes_and_scores,
    symmetric_kl_loss,
    safe_open_image,
    trainable_state_dict,
)
from calibrate import class_bias_from_logits, stratified_folds
from evaluate_tta_v5 import _search_mix


class MockClip(nn.Module):
    def __init__(self, dim: int = 2):
        super().__init__()
        self.config = SimpleNamespace(projection_dim=dim)
        self.frozen_marker = nn.Parameter(torch.ones(1), requires_grad=False)

    def get_image_features(self, pixel_values):
        return pixel_values


class MockAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)


class MockVisionLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = MockAttention(dim)


class MockLoraClip(MockClip):
    def __init__(self, dim: int = 4, layers: int = 3):
        super().__init__(dim)
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList(
            [MockVisionLayer(dim) for _ in range(layers)]
        )


class RobustClipTests(unittest.TestCase):
    def test_safe_open_retries_pixels_when_exif_transpose_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid_pixels.png"
            Image.new("RGB", (8, 8), color=(12, 34, 56)).save(path)
            with patch("robust_clip.ImageOps.exif_transpose", side_effect=OSError("bad EXIF")):
                recovered = safe_open_image(str(path))
            self.assertEqual(recovered.size, (8, 8))
            self.assertEqual(recovered.getpixel((0, 0)), (12, 34, 56))

    def test_calibration_bias_penalizes_overpredicted_class(self):
        logits = torch.tensor([[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
        bias = class_bias_from_logits(logits)
        self.assertLess(float(bias[0]), float(bias[1]))

    def test_stratified_folds_cover_each_class(self):
        labels = np.repeat(np.arange(3), 10)
        folds = stratified_folds(labels, folds=5, seed=2026)
        for label in range(3):
            self.assertEqual(set(folds[labels == label].tolist()), set(range(5)))

    def test_trimmed_prototypes_reject_outliers(self):
        features = np.array(
            [[1, 0], [0.98, 0.02], [-1, 0], [0, 1], [0.02, 0.98], [0, -1]],
            dtype=np.float32,
        )
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        labels = np.array([0, 0, 0, 1, 1, 1])
        prototypes, _ = robust_visual_prototypes_and_scores(
            features, labels, 2, keep_fraction=2 / 3, iterations=2
        )
        self.assertGreater(prototypes[0, 0], 0.99)
        self.assertGreater(prototypes[1, 1], 0.99)

    def test_holdout_rows_receive_no_training_weight(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.1, 0.9, 0.8, 0.1], dtype=np.float32)
        candidates = [0, 1, 3, 4]
        ranks = classwise_rank_scores(labels, scores, candidates)
        rows = [(str(i), int(labels[i]), str(labels[i])) for i in range(len(labels))]
        quality, clean = classwise_quality(
            rows, scores=ranks, clean_fraction=0.5, weight_floor=0.25, candidate_indices=candidates
        )
        np.testing.assert_array_equal(quality[[2, 5]], np.zeros(2, dtype=np.float32))
        self.assertFalse(clean[2])
        self.assertFalse(clean[5])

    def test_gce_has_stronger_low_probability_gradient_than_mae(self):
        logits = torch.tensor([[0.0, 6.0, 5.0]], requires_grad=True)
        target = torch.tensor([0])
        generalized_cross_entropy(logits, target, 0.7).backward()
        gce_gradient = abs(float(logits.grad[0, 0]))
        logits.grad.zero_()
        (1.0 - logits.softmax(dim=1)[0, 0]).backward()
        mae_gradient = abs(float(logits.grad[0, 0]))
        self.assertGreater(gce_gradient, mae_gradient)

    def test_compact_state_omits_clip_and_reloads(self):
        first = RobustCLIPClassifier(
            MockClip(), classifier_init=torch.eye(2), max_logit_scale=50, learn_logit_scale=False
        )
        state = trainable_state_dict(first)
        self.assertFalse(any(name.startswith("clip.") for name in state))
        second = RobustCLIPClassifier(
            MockClip(), classifier_init=torch.randn(2, 2), max_logit_scale=50, learn_logit_scale=False
        )
        load_classifier_state(second, state)
        self.assertTrue(
            torch.allclose(F.normalize(first.classifier, dim=1), F.normalize(second.classifier, dim=1))
        )

    def test_lora_zero_init_preserves_projection(self):
        base = nn.Linear(4, 3)
        inputs = torch.randn(5, 4)
        expected = base(inputs).detach()
        lora = LoRALinear(base, rank=2, alpha=4)
        self.assertTrue(torch.allclose(lora(inputs), expected))

    def test_lora_is_compact_and_reloads(self):
        first = RobustCLIPClassifier(
            MockLoraClip(), classifier_init=torch.randn(2, 4), max_logit_scale=50
        )
        replaced = inject_visual_lora(first.clip, last_n_layers=2, rank=2, alpha=4)
        self.assertEqual(len(replaced), 4)
        with torch.no_grad():
            first.clip.vision_model.encoder.layers[-1].self_attn.q_proj.lora_b.fill_(0.25)
        state = trainable_state_dict(first)
        lora_keys = [name for name in state if name.startswith("clip.")]
        self.assertTrue(lora_keys)
        self.assertTrue(all("lora_" in name for name in lora_keys))
        self.assertFalse(any("base.weight" in name for name in state))

        second = RobustCLIPClassifier(
            MockLoraClip(), classifier_init=torch.randn(2, 4), max_logit_scale=50
        )
        inject_visual_lora(second.clip, last_n_layers=2, rank=2, alpha=4)
        load_classifier_state(second, state)
        self.assertTrue(
            torch.allclose(
                first.clip.vision_model.encoder.layers[-1].self_attn.q_proj.lora_b,
                second.clip.vision_model.encoder.layers[-1].self_attn.q_proj.lora_b,
            )
        )

    def test_prototype_pseudo_targets_report_margin(self):
        features = np.asarray([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
        prototypes = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        pseudo, confidence, margin = prototype_pseudo_targets(features, prototypes, temperature=0.1)
        np.testing.assert_array_equal(pseudo, np.asarray([0, 0]))
        self.assertGreater(float(confidence[0]), 0.99)
        self.assertGreater(float(margin[0]), float(margin[1]))

    def test_symmetric_kl_is_zero_for_equal_views(self):
        logits = torch.randn(4, 5)
        divergence = symmetric_kl_loss(logits, logits, temperature=2.0)
        self.assertTrue(torch.allclose(divergence, torch.zeros_like(divergence), atol=1e-6))

    def test_paired_dataset_returns_two_clip_views(self):
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
                size={"shortest_edge": 224},
                crop_size={"height": 224, "width": 224},
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/image.jpg"
            Image.new("RGB", (224, 224), color=(128, 64, 32)).save(path)
            dataset = PairedFolderImageDataset([(path, 0, "0000")], [0], processor, augment="light")
            sample = dataset[0]
        self.assertEqual(tuple(sample["pixel_values_a"].shape), (3, 224, 224))
        self.assertEqual(tuple(sample["pixel_values_b"].shape), (3, 224, 224))
        self.assertEqual(int(sample["label"]), 0)
        self.assertEqual(int(sample["row_index"]), 0)

    def test_tta_search_prefers_correct_original_view(self):
        labels = np.asarray([0, 1, 2] * 4, dtype=np.int64)
        original = np.full((len(labels), 3), -4.0, dtype=np.float32)
        hflip = np.full((len(labels), 3), -4.0, dtype=np.float32)
        original[np.arange(len(labels)), labels] = 4.0
        hflip[np.arange(len(labels)), (labels + 1) % 3] = 4.0
        folds = stratified_folds(labels, folds=3, seed=2026)
        weight, alpha, accuracy, _ = _search_mix(
            original,
            hflip,
            labels,
            folds,
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.5], dtype=np.float32),
            torch.device("cpu"),
        )
        self.assertEqual(weight, 1.0)
        self.assertGreaterEqual(accuracy, 0.99)


if __name__ == "__main__":
    unittest.main()
