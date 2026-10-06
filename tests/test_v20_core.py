"""V20 DoRA mathematics, fixed optimizer topology and complete state contract."""

from copy import deepcopy
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from robust_clip import LoRALinear, trainable_state_dict
from test_v14_core import processor, tiny_clip
from train_v5 import clone_trainable, swapped_trainable, update_ema
from v19_core import recipe_config as v19_config
from v19_views import training_dataset as v19_dataset
from v20_core import RECIPES, build_optimizer, optimizer_group_metadata, recipe_config, validate_recipe_config
from v20_model import (
    DoRALinear, _official_trainable_shapes, build_classifier, build_classifier_from_checkpoint,
    checkpoint_resolution, load_classifier_state, trainable_parameter_names,
    validate_checkpoint_state_metadata, validate_trainable_metadata, validate_trainable_state,
)
from v20_views import FourViewTestDataset, build_zoom_transform, resolution_processor, training_dataset


def config(recipe="dora_rank32"):
    return {**recipe_config(recipe), "base_model_identity": {"test": "base"}}


class V20CoreTests(unittest.TestCase):
    def test_frozen_recipes_and_stage_selection(self):
        for recipe in RECIPES:
            values = recipe_config(recipe)
            validate_recipe_config(values)
            self.assertEqual((values["image_size"], values["zoom_shortest_edge"], values["lora_rank"],
                              values["lora_alpha"], values["lora_dropout"], values["mlp_lora_dropout"]),
                             (320, 366, 32, 64.0, 0.0, 0.0))
            self.assertEqual((values["epochs"], values["schedule_epochs"], values["batch_size"], values["seed"]),
                             (24, 24, 256, 2026))
            for key, value in values.items():
                if key == "recipe": continue
                if isinstance(value, bool): wrong = not value
                elif isinstance(value, (int, float)): wrong = value + 1
                elif isinstance(value, str): wrong = value + "_changed"
                else: wrong = list(value) + ["changed"]
                with self.subTest(recipe=recipe, key=key), self.assertRaises(ValueError):
                    validate_recipe_config({**values, key: wrong})
            for stage, selection in (("validate", "joint_epoch_tta_bias_nested_cv"),
                                     ("refit", "frozen_selection")):
                validate_recipe_config({**values, "stage": stage, "epoch_selection": selection})
                with self.assertRaisesRegex(ValueError, "epoch_selection"):
                    validate_recipe_config({**values, "stage": stage, "epoch_selection": "fixed_equal_center_hflip_macro"})
        unchanged = v19_config("rank32_control")
        changed = recipe_config("loraplus_rank32")
        self.assertEqual({key for key in unchanged if unchanged[key] != changed[key]}, {"recipe"})
        with self.assertRaisesRegex(ValueError, "unknown V20"):
            recipe_config("rank32_control")
        changed["neighbor_quality_mix"][0] = .1
        self.assertEqual(recipe_config("loraplus_rank32")["neighbor_quality_mix"], [.75, .25])

    def test_adapter_initialization_preserves_v19_rng_order(self):
        from v19_model import build_classifier as old_builder
        torch.manual_seed(92)
        original = tiny_clip()
        torch.manual_seed(93)
        old = old_builder(deepcopy(original), np.eye(3, 16), "cpu", v19_config("rank32_control"))
        old_state = old.state_dict()
        for recipe in RECIPES:
            torch.manual_seed(93)
            new = build_classifier(deepcopy(original), np.eye(3, 16), "cpu", config(recipe))
            for name, value in old_state.items():
                torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)

    def test_dora_initialization_and_explicit_nonzero_formula_bias_and_gradients(self):
        for bias in (True, False):
            torch.manual_seed(23)
            base = nn.Linear(5, 7, bias=bias)
            layer = DoRALinear(deepcopy(base), rank=3, alpha=6)
            inputs = torch.randn(2, 4, 5)
            torch.testing.assert_close(layer(inputs), base(inputs), rtol=0, atol=0)
            self.assertEqual(layer.lora_m.dtype, torch.float32)
            self.assertTrue(all(not parameter.requires_grad for parameter in layer.base.parameters()))
            with torch.no_grad():
                layer.lora_b.normal_(std=.2)
                layer.lora_m.mul_(torch.linspace(.5, 1.8, 7))
            direction = layer.base.weight + layer.scaling * (layer.lora_b @ layer.lora_a)
            norm = direction.float().norm(dim=1).clamp_min(1e-6).detach()
            effective = direction * (layer.lora_m / norm)[:, None]
            expected = F.linear(inputs, effective, layer.base.bias)
            actual = layer(inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            parameters = (layer.lora_a, layer.lora_b, layer.lora_m)
            expected_grad = torch.autograd.grad(expected.square().sum(), parameters, retain_graph=True)
            actual_grad = torch.autograd.grad(actual.square().sum(), parameters)
            for left, right in zip(actual_grad, expected_grad):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
            self.assertTrue(all(parameter.grad is None for parameter in layer.base.parameters()))
        with self.assertRaisesRegex(ValueError, "dropout"):
            DoRALinear(nn.Linear(3, 4), 2, 4, .05)

    def test_dora_fp32_norm_under_autocast_and_zero_norm_guard(self):
        torch.manual_seed(2)
        layer = DoRALinear(nn.Linear(4, 6), 2, 4)
        with torch.no_grad():
            layer.lora_b.normal_(std=.1)
            layer.lora_m.mul_(1.2)
        inputs = torch.randn(3, 4)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = layer(inputs)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual.dtype, torch.bfloat16)
        with torch.no_grad():
            layer.base.weight.zero_()
            layer.lora_b.zero_()
        self.assertTrue(torch.isfinite(layer(inputs)).all())

    def test_optimizer_all_parameters_rates_decay_order_and_stable_metadata(self):
        for recipe in RECIPES:
            values = config(recipe)
            model = build_classifier(tiny_clip(), np.eye(3, 16), "cpu", values)
            optimizer, visual, head = build_optimizer(model, values)
            is_dora = values["method"] == "dora"
            self.assertEqual(len(visual), 216 if is_dora else 144)
            self.assertEqual(len(head), 6)
            self.assertEqual(len(optimizer.param_groups), 37 if is_dora else 25)
            metadata = optimizer_group_metadata(optimizer)
            flattened_names = [name for group in metadata for name in group["param_names"]]
            self.assertEqual(sorted(flattened_names), trainable_parameter_names(model))
            self.assertEqual(len(flattened_names), len(set(flattened_names)))
            params = [p for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(params), len({id(p) for p in params}))
            for group in metadata:
                if group["kind"] == "head":
                    self.assertEqual(group["initial_lr"], 5e-4)
                    continue
                kind, layer = group["kind"], group["layer"]
                rate = values["magnitude_lr"] if kind == "m" else values[f"lora_lr_{kind}"]
                self.assertAlmostEqual(group["initial_lr"], rate * .8 ** (11-layer))
                self.assertEqual(group["weight_decay"], 0.0 if kind == "m" else 1e-4)
            for group in optimizer.param_groups: group["lr"] *= .17
            self.assertEqual(metadata, optimizer_group_metadata(optimizer))
            again, _, _ = build_optimizer(model, values)
            self.assertEqual(metadata, optimizer_group_metadata(again))
            first_visual = next(name for name, p in model.named_parameters() if name.endswith("lora_a"))
            dict(model.named_parameters())[first_visual].requires_grad_(False)
            with self.assertRaisesRegex(ValueError, "complete"):
                build_optimizer(model, values)

    def test_all_modules_state_save_restore_and_magnitude_ema(self):
        for recipe in RECIPES:
            torch.manual_seed(11)
            base = tiny_clip()
            pristine = deepcopy(base)
            values = config(recipe)
            model = build_classifier(base, np.eye(3, 16), "cpu", values)
            cls = DoRALinear if values["method"] == "dora" else LoRALinear
            modules = [m for m in model.clip.modules() if isinstance(m, cls)]
            self.assertEqual(len(modules), 72)
            self.assertTrue(all(module.scaling == 2 for module in modules))
            ema = clone_trainable(model)
            with torch.no_grad():
                for module in modules:
                    module.lora_b.normal_(std=.01)
                    if isinstance(module, DoRALinear): module.lora_m.mul_(1.1)
            update_ema(ema, model, decay=.999, step=8)
            validate_trainable_state(model, ema, exact=True)
            if values["method"] == "dora":
                names = [name for name in ema if name.endswith("lora_m")]
                self.assertEqual(len(names), 72)
                with swapped_trainable(model, ema):
                    for name in names:
                        torch.testing.assert_close(dict(model.named_parameters())[name], ema[name], rtol=0, atol=0)
            checkpoint = {"format_version": 20, "config": values, "class_names": ["0000", "0001", "0002"],
                          "model": trainable_state_dict(model),
                          "trainable_parameter_names": trainable_parameter_names(model)}
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=False)
            with patch("v20_model.load_clip", return_value=(pristine, processor())), \
                 patch("v20_model.model_identity", return_value={"test": "base"}), \
                 patch("v20_model.validate_clip_vit_b32"):
                restored, prep = build_classifier_from_checkpoint(saved, "unused", "cpu")
            model.eval()
            pixels = torch.randn(1, 3, 320, 320)
            with torch.no_grad():
                torch.testing.assert_close(model(pixels)[0], restored(pixels)[0], rtol=0, atol=0)
            self.assertEqual(checkpoint_resolution(saved), 320)
            self.assertEqual(prep.image_processor.crop_size["height"], 320)

    def test_missing_magnitude_and_invalid_states_reject_before_mutation(self):
        model = build_classifier(tiny_clip(), np.eye(3, 16), "cpu", config())
        state = trainable_state_dict(model)
        name = next(name for name in state if name.endswith("lora_m"))
        missing = dict(state); del missing[name]
        invalid = [missing, {**state, name: torch.zeros(1)}, {**state, name: state[name].long()},
                   {**state, name: torch.full_like(state[name], float("nan"))},
                   {**state, "clip.unexpected": torch.zeros(1)}]
        for broken in invalid:
            with self.assertRaises(RuntimeError): load_classifier_state(model, broken)
            for key, value in trainable_state_dict(model).items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "trainable_parameter_names"):
            validate_trainable_metadata(model, {"trainable_parameter_names": [n for n in trainable_parameter_names(model) if n != name]})
        with self.assertRaises(RuntimeError): validate_trainable_state(model, state, exact=True)

    def test_official_shapes_complete_metadata_and_ema(self):
        for recipe in RECIPES:
            values = config(recipe)
            shapes = _official_trainable_shapes(750, values)
            m_shapes = [shape for name, shape in shapes.items() if name.endswith("lora_m")]
            self.assertEqual(len(m_shapes), 72 if values["method"] == "dora" else 0)
            self.assertEqual(sum(int(np.prod(shape)) for shape in m_shapes), 82944 if m_shapes else 0)
            visual_count = sum(int(np.prod(shape)) for name, shape in shapes.items() if name.startswith("clip."))
            self.assertEqual(visual_count, 5308416 + (82944 if m_shapes else 0))
            model = {name: torch.zeros(shape) for name, shape in {**shapes, "logit_scale": ()}.items()}
            checkpoint = {"format_version": 20, "config": values, "class_names": [str(i) for i in range(750)],
                          "model": model, "trainable_parameter_names": sorted(shapes),
                          "ema": {name: model[name] for name in shapes}}
            self.assertEqual(validate_checkpoint_state_metadata(checkpoint), sorted(shapes))
            name = next(name for name in shapes if name.endswith("lora_m" if m_shapes else "lora_b"))
            for key in ("model", "ema"):
                missing = dict(checkpoint[key]); del missing[name]
                with self.assertRaisesRegex(ValueError, "keys"):
                    validate_checkpoint_state_metadata({**checkpoint, key: missing})
                with self.assertRaisesRegex(ValueError, "shape"):
                    validate_checkpoint_state_metadata({**checkpoint, key: {**checkpoint[key], name: torch.zeros(1)}})
            with self.assertRaisesRegex(ValueError, "trainable_parameter_names"):
                validate_checkpoint_state_metadata({**checkpoint, "trainable_parameter_names": sorted(shapes)[1:]})

    def test_v19_v18_checkpoint_delegation(self):
        for version in (18, 19):
            checkpoint = {"format_version": version}
            with patch("v20_model.legacy_classifier", return_value=("legacy", "processor")) as old:
                self.assertEqual(build_classifier_from_checkpoint(checkpoint, "unused", "cpu"), ("legacy", "processor"))
                old.assert_called_once_with(checkpoint, "unused", "cpu")
            with patch("v20_model.legacy_resolution", return_value=384):
                self.assertEqual(checkpoint_resolution(checkpoint), 384)
            with patch(f"v{version}_model.validate_checkpoint_state_metadata", return_value=["legacy"]):
                self.assertEqual(validate_checkpoint_state_metadata(checkpoint), ["legacy"])

    def test_320_views_keep_v19_geometry_quality_and_rng(self):
        original = processor()
        prep = resolution_processor(original)
        self.assertEqual(original.image_processor.crop_size["height"], 224)
        transform = build_zoom_transform(prep)
        self.assertEqual(transform.transforms[0].size, 366)
        self.assertEqual(transform.transforms[1].size, (320, 320))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.png"
            Image.fromarray(np.random.default_rng(9).integers(0, 256, (450, 640, 3), dtype=np.uint8)).save(path)
            rows = [(str(path), 0, "0000")]
            old = v19_dataset(rows, [0], prep, v19_config("rank32_control"))
            for recipe in RECIPES:
                new = training_dataset(rows, [0], prep, config(recipe))
                random.seed(12); torch.manual_seed(12); expected = old[0]
                random.seed(12); torch.manual_seed(12); actual = new[0]
                for key in ("pixel_values_a", "pixel_values_b"):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                views = FourViewTestDataset([str(path)], prep)[0]
                self.assertEqual(views["zoom256"].shape, (3, 320, 320))
            with self.assertRaisesRegex(ValueError, "resolution"):
                training_dataset(rows, [0], resolution_processor(original, 384), config())


if __name__ == "__main__":
    unittest.main()
