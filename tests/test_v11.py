"""Focused V11 checks: resolution, trainable gradients, inference and resume."""

from copy import deepcopy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transformers import CLIPConfig, CLIPModel

from evaluate_tta_v11 import StressFourViewDataset, collect_view_logits
from predict_v11 import _weighted_logits
from robust_clip import FolderImageDataset, RobustCLIPClassifier, load_classifier_state, trainable_state_dict
from train_v5 import clone_trainable
from train_v11 import _restore_rng, load_resume, save_resume
from v6_core import SelectiveFeatureQueue
from v11_resolution import (
    FourViewTestDataset, HighResolutionPairedDataset, build_classifier,
    build_classifier_from_checkpoint, checkpoint_resolution, resolution_processor,
)


def processor():
    return SimpleNamespace(image_processor=SimpleNamespace(
        size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224},
        image_mean=(0.5, 0.5, 0.5), image_std=(0.25, 0.25, 0.25),
    ))


def model():
    return CLIPModel(CLIPConfig(
        projection_dim=16,
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                       "num_attention_heads": 4, "patch_size": 32, "image_size": 224},
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                     "num_attention_heads": 4, "vocab_size": 20},
    ))


class V11Tests(unittest.TestCase):
    def test_320_forward_backward_and_224_equivalence(self):
        torch.manual_seed(2026)
        classifier = build_classifier(model(), np.eye(3, 16, dtype=np.float32), torch.device("cpu"))
        pixels = torch.randn(2, 3, 320, 320)
        embedding = classifier.clip.vision_model.embeddings(pixels, interpolate_pos_encoding=True)
        self.assertEqual(embedding.shape[1], 101)
        logits = classifier(pixels)[0]
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        grads = [p.grad for n, p in classifier.named_parameters() if n.endswith("lora_b")]
        self.assertEqual(len(grads), 8)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        self.assertIsNone(classifier.clip.vision_model.embeddings.position_embedding.weight.grad)
        self.assertEqual(classifier.clip.vision_model.embeddings.position_embedding.weight.shape[0], 50)
        with self.assertRaisesRegex(ValueError, "320x320"):
            classifier(torch.randn(1, 3, 224, 224))
        classifier.image_size = 224
        classifier.eval()
        pixels = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            torch.testing.assert_close(classifier(pixels)[0], RobustCLIPClassifier.forward(classifier, pixels)[0], rtol=0, atol=0)

    def test_train_audit_validation_and_prediction_use_320(self):
        original = processor()
        highres = resolution_processor(original)
        self.assertEqual(original.image_processor.size, {"shortest_edge": 224})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            yy, xx = np.indices((480, 800))
            Image.fromarray(np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)).save(path)
            rows = [(str(path), 0, "class")]
            paired = HighResolutionPairedDataset(rows, [0], highres)
            torch.manual_seed(10)
            first = paired[0]
            torch.manual_seed(10)
            repeat = paired[0]
            for key in ("pixel_values_a", "pixel_values_b"):
                self.assertEqual(tuple(first[key].shape), (3, 320, 320))
                torch.testing.assert_close(first[key], repeat[key], rtol=0, atol=0)
            validation = StressFourViewDataset(rows, [0], highres, "native", four=True)[0]
            prediction = FourViewTestDataset([str(path)], highres)
            self.assertEqual(prediction.zoom_transform.transforms[0].size, 366)
            predicted = prediction[0]
            for key in ("center", "zoom256"):
                torch.testing.assert_close(validation[key], predicted[key], rtol=0, atol=0)
                self.assertEqual(tuple(predicted[key].shape), (3, 320, 320))
            audit = FolderImageDataset(rows, [0], highres, augment="none")[0]
            torch.testing.assert_close(audit["pixel_values"], validation["center"], rtol=0, atol=0)
            classifier = build_classifier(model(), np.eye(3, 16, dtype=np.float32), torch.device("cpu"))
            views, _ = collect_view_logits(classifier, highres, rows, [0], "native", four=True,
                                          device=torch.device("cpu"), batch_size=1, workers=0)
            batch = {key: predicted[key].unsqueeze(0) for key in ("center", "zoom256")}
            with torch.no_grad():
                logits = _weighted_logits(classifier, batch, {key: 0.25 for key in views}, torch.device("cpu"))
            np.testing.assert_allclose(logits.numpy(), sum(views.values()) / 4, atol=1e-6)

    def test_checkpoint_recreates_the_same_high_resolution_model(self):
        base = model()
        pristine = deepcopy(base)
        classifier = build_classifier(base, np.eye(3, 16, dtype=np.float32), torch.device("cpu"))
        checkpoint = {"format_version": 11, "class_names": ["0000", "0001", "0002"],
                      "config": {"image_size": 320, "zoom_shortest_edge": 366,
                                 "interpolate_pos_encoding": True, "lora_rank": 8,
                                 "lora_layers": 4, "lora_alpha": 16.0},
                      "model": trainable_state_dict(classifier)}
        with patch("v11_resolution.load_clip", return_value=(pristine, processor())), patch("v11_resolution.validate_clip_vit_b32"):
            restored, prep = build_classifier_from_checkpoint(checkpoint, "unused", torch.device("cpu"))
        self.assertEqual(prep.image_processor.crop_size["height"], 320)
        pixels = torch.randn(1, 3, 320, 320)
        classifier.eval()
        with torch.no_grad():
            torch.testing.assert_close(classifier(pixels)[0], restored(pixels)[0], rtol=0, atol=0)
        checkpoint["config"]["image_size"] = 224
        with self.assertRaisesRegex(ValueError, "320"):
            checkpoint_resolution(checkpoint)

    def test_resumed_optimizer_step_matches_uninterrupted_step(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Parameter(torch.eye(2))

        def setup():
            network = Tiny()
            optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            return network, optimizer, scheduler

        def step(network, optimizer, scheduler):
            optimizer.zero_grad()
            (network.classifier @ torch.randn(2, 2)).square().mean().backward()
            optimizer.step()
            scheduler.step()

        torch.manual_seed(12)
        random.seed(12)
        np.random.seed(12)
        network, optimizer, scheduler = setup()
        step(network, optimizer, scheduler)
        config = {"version": "v11", "image_size": 320, "soft_teacher": True,
                  "batch_size": 64, "gradient_accumulation": 4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            save_resume(path, stage="validation", epoch=1, classifier=network, optimizer=optimizer,
                        scheduler=scheduler, ema=clone_trainable(network),
                        queue=SelectiveFeatureQueue(2, 2, 2, torch.device("cpu")), arrays={},
                        optimizer_step=1, history=[], best_state=clone_trainable(network),
                        best_record=None, selected_epoch=None, config=config, dataset_digest="data")
            step(network, optimizer, scheduler)
            expected = network.classifier.detach().clone()
            state = load_resume(str(path), config, "data", torch.device("cpu"))
            resumed, new_optimizer, new_scheduler = setup()
            load_classifier_state(resumed, state["model"])
            new_optimizer.load_state_dict(state["optimizer"])
            new_scheduler.load_state_dict(state["scheduler"])
            _restore_rng(state)
            step(resumed, new_optimizer, new_scheduler)
            torch.testing.assert_close(resumed.classifier, expected, rtol=0, atol=0)
            load_resume(str(path), {**config, "batch_size": 32, "gradient_accumulation": 8}, "data", torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_resume(str(path), {**config, "image_size": 224}, "data", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
