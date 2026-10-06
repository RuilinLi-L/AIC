"""V14's controlled augmentation, model compatibility and runtime controls."""

from copy import deepcopy
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPConfig, CLIPModel

from robust_clip import trainable_state_dict
from train_v6 import load_or_create_multiview_features
from v10_views import degrade_view_b
from v11_resolution import HighResolutionPairedDataset, resolution_processor
from v13_core import recipe_config as v13_recipe_config
from v14_core import recipe_config
from v14_model import build_classifier, build_classifier_from_checkpoint, checkpoint_resolution
from v14_runtime import pin_memory_enabled, scientific_config
from v14_views import Robust320PairedDataset, training_dataset


def processor():
    return SimpleNamespace(image_processor=SimpleNamespace(
        size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224},
        image_mean=(0.5, 0.5, 0.5), image_std=(0.25, 0.25, 0.25),
    ))


def tiny_clip():
    return CLIPModel(CLIPConfig(
        projection_dim=16,
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 12,
                       "num_attention_heads": 4, "patch_size": 32, "image_size": 224},
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                     "num_attention_heads": 4, "vocab_size": 20},
    ))


class V14CoreTests(unittest.TestCase):
    def test_recipe_changes_only_augmentation_and_name(self):
        reference = v13_recipe_config("expanded")
        for name in ("expanded", "expanded_robust"):
            actual = recipe_config(name)
            augmentation = actual.pop("augmentation")
            actual["recipe"] = "expanded"
            self.assertEqual(actual, reference)
            self.assertEqual(augmentation, "robust_quality_320_view_b" if name.endswith("robust")
                             else "light_320_both_views")
        with self.assertRaisesRegex(ValueError, "unknown V14 recipe"):
            recipe_config("dynamic")

    def test_scientific_config_allows_only_runtime_changes_and_effective_256(self):
        original = {**recipe_config("expanded_robust"), "batch_size": 256, "gradient_accumulation": 1,
                    "workers": 4, "prefetch_factor": 1, "eval_batch_size": 128, "pin_memory": False,
                    "dataset_signature": "same", "precision": "fp32"}
        tuned = {**original, "batch_size": 64, "gradient_accumulation": 4, "workers": 0,
                 "prefetch_factor": 3, "eval_batch_size": 64, "pin_memory": True}
        self.assertEqual(scientific_config(original), scientific_config(tuned))
        self.assertEqual(original["batch_size"], 256)
        for key, value in (("lora_lr", 1e-3), ("precision", "bf16"), ("recipe", "expanded"),
                           ("augmentation", "different"), ("dataset_signature", "changed")):
            self.assertNotEqual(scientific_config(original), scientific_config({**original, key: value}))
        for batch, accumulation in ((128, 1), (0, 256), (256.0, 1), (True, 256), (256, -1)):
            with self.assertRaisesRegex(ValueError, "= 256"):
                scientific_config({**original, "batch_size": batch, "gradient_accumulation": accumulation})

    def test_pinned_memory_defaults_off_and_rejects_bad_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(pin_memory_enabled(torch.device("cuda")))
        with patch.dict(os.environ, {"AIC_PIN_MEMORY": "1"}):
            self.assertTrue(pin_memory_enabled(torch.device("cuda")))
            self.assertFalse(pin_memory_enabled(torch.device("cpu")))
        for value in ("yes", "", "2"):
            with patch.dict(os.environ, {"AIC_PIN_MEMORY": value}), self.assertRaises(ValueError):
                pin_memory_enabled(torch.device("cpu"))

    def test_robust_views_preserve_view_a_and_are_seeded_320(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.png"
            Image.fromarray(np.random.default_rng(7).integers(0, 256, (450, 640, 3), dtype=np.uint8)).save(path)
            rows = [(str(path), 2, "0002")]
            prep = resolution_processor(processor())
            plain = training_dataset(rows, [0], prep, recipe_config("expanded"))
            robust = training_dataset(rows, [0], prep, recipe_config("expanded_robust"))
            self.assertIsInstance(plain, HighResolutionPairedDataset)
            self.assertIsInstance(robust, Robust320PairedDataset)
            torch.manual_seed(28)
            plain_row = plain[0]
            random.seed(28); torch.manual_seed(28)
            first = robust[0]
            random.seed(28); torch.manual_seed(28)
            second = robust[0]
            torch.testing.assert_close(first["pixel_values_a"], plain_row["pixel_values_a"], rtol=0, atol=0)
            for key in ("pixel_values_a", "pixel_values_b"):
                self.assertEqual(tuple(first[key].shape), (3, 320, 320))
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            self.assertEqual((first["label"].item(), first["row_index"].item()), (2, 0))
            with patch("v14_views.degrade_view_b", return_value=Image.new("RGB", (640, 450), "black")) as change:
                black_b = robust[0]
            change.assert_called_once()
            self.assertEqual(change.call_args.args[0].size, (640, 450))
            self.assertTrue(torch.all(black_b["pixel_values_b"] == -2))
            with self.assertRaisesRegex(ValueError, "320-pixel"):
                training_dataset(rows, [0], processor(), recipe_config("expanded_robust"))
            with self.assertRaisesRegex(ValueError, "augmentation"):
                training_dataset(rows, [0], prep, {**recipe_config("expanded"), "augmentation": "wrong"})

    def test_quality_degradation_ranges_and_no_upscale(self):
        image = Image.new("RGB", (800, 400), (60, 90, 120))
        rng = Mock()
        rng.random.side_effect = [0.1, 0.1]
        rng.randint.side_effect = [320, 65]
        result = degrade_view_b(image, rng)
        self.assertEqual(result.size, (320, 160))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(rng.randint.call_args_list, [call(320, 640), call(65, 95)])
        small = Image.new("RGB", (200, 100))
        rng = Mock()
        rng.random.side_effect = [0.1, 0.9]
        rng.randint.return_value = 640
        self.assertIs(degrade_view_b(small, rng), small)
        rng = Mock()
        rng.random.side_effect = [0.9, 0.9]
        self.assertIs(degrade_view_b(image, rng), image)
        rng.randint.assert_not_called()

    def test_feature_loader_legacy_defaults_and_explicit_pin_prefetch(self):
        model = SimpleNamespace(config=SimpleNamespace(projection_dim=2), eval=lambda: None,
                                get_image_features=lambda pixel_values: torch.ones(len(pixel_values), 2))
        batch = {"views": torch.zeros(1, 4, 3, 224, 224), "row_index": torch.tensor([0])}
        rows = [("not_decoded.png", 0, "0000")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (workers, options) in enumerate(((0, {}), (0, {"pin_memory": False, "prefetch_factor": 1}),
                                                       (4, {"pin_memory": False, "prefetch_factor": 1}))):
                with patch("train_v6.DataLoader", return_value=[batch]) as loader:
                    features = load_or_create_multiview_features(model, processor(), rows, root / f"{index}.npy",
                                                                 "signature", torch.device("cpu"), 64, workers, **options)
                self.assertEqual(features.shape, (4, 1, 2))
                self.assertFalse(loader.call_args.kwargs["pin_memory"])
                self.assertEqual(loader.call_args.kwargs["persistent_workers"], workers > 0)
                if workers:
                    self.assertEqual(loader.call_args.kwargs["prefetch_factor"], 1)
                else:
                    self.assertNotIn("prefetch_factor", loader.call_args.kwargs)
            for options in ({"pin_memory": "0"}, {"prefetch_factor": 0}, {"prefetch_factor": True}):
                with self.assertRaises(ValueError):
                    load_or_create_multiview_features(model, processor(), rows, root / "invalid.npy",
                                                      "signature", torch.device("cpu"), 64, 0, **options)

    def test_v14_model_gradients_roundtrip_and_v13_delegation(self):
        torch.manual_seed(13)
        base = tiny_clip()
        pristine = deepcopy(base)
        config = {**recipe_config("expanded_robust"), "image_size": 320, "zoom_shortest_edge": 366,
                  "interpolate_pos_encoding": True, "base_model_identity": {"test": "base"}}
        model = build_classifier(base, np.eye(3, 16), torch.device("cpu"), config)
        pixels = torch.randn(2, 3, 320, 320)
        F.cross_entropy(model(pixels)[0], torch.tensor([0, 1])).backward()
        grads = [p.grad for name, p in model.named_parameters() if name.endswith("lora_b")]
        self.assertEqual(len(grads), 48)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
        self.assertTrue(all(name.endswith(("lora_a", "lora_b")) for name, p in model.clip.named_parameters()
                            if p.requires_grad))
        checkpoint = {"format_version": 14, "config": config, "class_names": ["0000", "0001", "0002"],
                      "model": trainable_state_dict(model)}
        with patch("v14_model.load_clip", return_value=(pristine, processor())), \
             patch("v14_model.model_identity", return_value={"test": "base"}), \
             patch("v14_model.validate_clip_vit_b32"):
            restored, _ = build_classifier_from_checkpoint(checkpoint, "unused", torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(pixels)[0], restored(pixels)[0], rtol=0, atol=0)
        self.assertEqual(checkpoint_resolution(checkpoint), 320)
        legacy = {**checkpoint, "format_version": 13, "config": {**config, **v13_recipe_config("expanded"),
                                                                 "augmentation": "light_320_both_views"}}
        self.assertEqual(checkpoint_resolution(legacy), 320)
        with patch("v14_model.legacy_classifier", return_value=("legacy", "processor")) as delegated:
            self.assertEqual(build_classifier_from_checkpoint(legacy, "unused", torch.device("cpu")),
                             ("legacy", "processor"))
        delegated.assert_called_once()
        for key, value in (("lora_layers", 4), ("image_size", 224), ("augmentation", "wrong")):
            with self.assertRaises(ValueError):
                checkpoint_resolution({**checkpoint, "config": {**config, key: value}})


if __name__ == "__main__":
    unittest.main()
