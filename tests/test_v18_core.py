"""V18 controlled recipes, actual input geometry, LoRA state and robust views."""

from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from robust_clip import LoRALinear, _feature_tensor, trainable_state_dict
from test_v14_core import processor, tiny_clip
from v14_views import Robust320PairedDataset
from v15_core import recipe_config as baseline_recipe
from v18_core import RECIPES, V18_CONFIG_KEYS, build_optimizer, recipe_config, validate_recipe_config
from v18_model import (
    _official_trainable_shapes, build_classifier, build_classifier_from_checkpoint,
    checkpoint_resolution, load_classifier_state, trainable_parameter_names,
    validate_checkpoint_state_metadata, validate_trainable_state,
)
from v18_views import (
    FourViewTestDataset, RobustPairedDataset, build_zoom_transform,
    degrade_view_b, resolution_processor, training_dataset,
)


def config(recipe="resolution384"):
    return {**recipe_config(recipe), "base_model_identity": {"test": "base"}}


class V18CoreTests(unittest.TestCase):
    def test_matched_control_initialization_and_optimizer_match_v15_exactly(self):
        from v15_model import build_classifier as old_model
        from v15_core import build_optimizer as old_optimizer
        torch.manual_seed(103)
        original = tiny_clip()
        models = []
        for builder, values in ((old_model, baseline_recipe('expanded_mlp')),
                                (build_classifier, recipe_config('matched_control'))):
            torch.manual_seed(104)
            models.append(builder(deepcopy(original), np.eye(3, 16), torch.device('cpu'), values))
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
        a, _, _ = old_optimizer(models[0], baseline_recipe('expanded_mlp'))
        b, _, _ = build_optimizer(models[1], recipe_config('matched_control'))
        self.assertEqual(a.state_dict()['param_groups'], b.state_dict()['param_groups'])
        pixels = torch.randn(2, 3, 320, 320)
        for model in models: model.eval()
        with torch.no_grad():
            torch.testing.assert_close(models[0](pixels)[0], models[1](pixels)[0], rtol=0, atol=0)

    def test_recipes_change_only_resolution_or_rank_from_v15(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                values = recipe_config(recipe)
                comparable = {key: value for key, value in values.items() if key not in V18_CONFIG_KEYS}
                comparable["recipe"] = "expanded_mlp"
                for key in ("lora_rank", "mlp_lora_rank"):
                    comparable[key] = 16
                for key in ("lora_alpha", "mlp_lora_alpha"):
                    comparable[key] = 32.0
                self.assertEqual(comparable, baseline_recipe("expanded_mlp"))
                validate_recipe_config(values)
                validate_recipe_config({**values, "lora_targets": list(values["lora_targets"]),
                                        "mlp_lora_targets": list(values["mlp_lora_targets"])})
        a, b, control = [recipe_config(recipe) for recipe in RECIPES]
        self.assertEqual((a["image_size"], a["zoom_shortest_edge"], a["lora_rank"], a["lora_alpha"]),
                         (384, 439, 16, 32.0))
        self.assertEqual((b["image_size"], b["zoom_shortest_edge"], b["lora_rank"], b["lora_alpha"]),
                         (320, 366, 32, 64.0))
        self.assertEqual({key for key in a if a[key] != control[key]},
                         {"recipe", "image_size", "zoom_shortest_edge"})
        self.assertEqual({key for key in b if b[key] != control[key]},
                         {"recipe", "lora_rank", "lora_alpha", "mlp_lora_rank", "mlp_lora_alpha"})
        for key, value in (("image_size", 320), ("zoom_shortest_edge", 440),
                           ("lora_rank", 32), ("mlp_lora_alpha", 64.0),
                           ("augmentation", "light_320_both_views"), ("layernorm_layers", 4),
                           ("agreement_recovery", True), ("dynamic_prototype", True),
                           ("activation_checkpointing", True), ("frozen_feature_image_size", 384),
                           ("interpolate_pos_encoding", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe_config({**a, key: value})
        with self.assertRaisesRegex(ValueError, "unknown V18"):
            recipe_config("rank64")

    def test_recipe_dimensions_processor_copy_and_zoom(self):
        original = processor()
        for size, zoom in ((224, 256), (320, 366), (384, 439)):
            prep = resolution_processor(original, size)
            self.assertEqual(prep.image_processor.size, {"shortest_edge": size})
            self.assertEqual(prep.image_processor.crop_size, {"height": size, "width": size})
            transform = build_zoom_transform(prep)
            self.assertEqual(transform.transforms[0].size, zoom)
            self.assertEqual(transform.transforms[1].size, (size, size))
        self.assertEqual(original.image_processor.crop_size, {"height": 224, "width": 224})
        for bad in (256, 448, True):
            with self.assertRaises(ValueError):
                resolution_processor(original, bad)
        prep.image_processor.crop_size["width"] = 320
        with self.assertRaises(ValueError):
            build_zoom_transform(prep)

    def test_all_recipes_keep_v15_quality_views_and_rng_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.png"
            Image.fromarray(np.random.default_rng(7).integers(0, 256, (450, 640, 3), dtype=np.uint8)).save(path)
            rows = [(str(path), 2, "0002")]
            prep320 = resolution_processor(processor(), 320)
            old = Robust320PairedDataset(rows, [0], prep320)
            for recipe in ("rank32", "matched_control"):
                new = training_dataset(rows, [0], prep320, config(recipe))
                self.assertIsInstance(new, RobustPairedDataset)
                random.seed(28); torch.manual_seed(28)
                expected = old[0]
                random.seed(28); torch.manual_seed(28)
                actual = new[0]
                for key in ("pixel_values_a", "pixel_values_b"):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            prep384 = resolution_processor(processor(), 384)
            new = training_dataset(rows, [0], prep384, config())
            with patch("v18_views.degrade_view_b", wraps=degrade_view_b) as change:
                actual = new[0]
            change.assert_called_once()
            self.assertEqual(change.call_args.args[0].size, (640, 450))
            self.assertEqual(actual["pixel_values_a"].shape, (3, 384, 384))
            self.assertEqual(actual["pixel_values_b"].shape, (3, 384, 384))
            views = FourViewTestDataset([str(path)], prep384)[0]
            self.assertEqual(set(views), {"center", "zoom256", "path"})
            for key in ("center", "zoom256"):
                self.assertEqual(views[key].shape, (3, 384, 384))
            with self.assertRaisesRegex(ValueError, "resolution"):
                training_dataset(rows, [0], prep320, config())

    def test_quality_degradation_has_original_unscaled_ranges(self):
        rng = Mock()
        rng.random.side_effect = [.1, .1]
        rng.randint.side_effect = [320, 65]
        image = Image.new("RGB", (800, 400), (60, 90, 120))
        self.assertEqual(degrade_view_b(image, rng).size, (320, 160))
        self.assertEqual(rng.randint.call_args_list, [call(320, 640), call(65, 95)])
        small = Image.new("RGB", (200, 100))
        rng = Mock(); rng.random.side_effect = [.1, .9]; rng.randint.return_value = 640
        self.assertIs(degrade_view_b(small, rng), small)

    def test_all_modules_init_gradients_frozen_weights_and_roundtrip(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                torch.manual_seed(17)
                base = tiny_clip()
                pristine = deepcopy(base)
                values = config(recipe)
                model = build_classifier(base, np.eye(3, 16), torch.device("cpu"), values)
                modules = [module for module in model.clip.modules() if isinstance(module, LoRALinear)]
                self.assertEqual(len(modules), 72)
                self.assertTrue(all(module.lora_a.shape[0] == values["lora_rank"] for module in modules))
                self.assertTrue(all(module.scaling == 2 and module.lora_b.count_nonzero() == 0 for module in modules))
                self.assertTrue(all(name.endswith(("lora_a", "lora_b"))
                                    for name, p in model.clip.named_parameters() if p.requires_grad))
                before = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
                optimizer, visual, head = build_optimizer(model, values)
                self.assertEqual(len(optimizer.param_groups), 13)
                self.assertEqual(len(visual), 144)
                self.assertEqual(len({id(p) for group in optimizer.param_groups for p in group["params"]}),
                                 len(visual) + len(head))
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4 * .8 ** 11)
                self.assertEqual(optimizer.param_groups[-2]["lr"], 1e-4)
                self.assertEqual(optimizer.param_groups[-1]["lr"], 5e-4)
                pixels = torch.randn(2, 3, values["image_size"], values["image_size"])
                with torch.no_grad():
                    embeddings = model.clip.vision_model.embeddings(pixels, interpolate_pos_encoding=True)
                    self.assertEqual(embeddings.shape[1], 145 if recipe == "resolution384" else 101)
                    original_base = _feature_tensor(pristine.get_image_features(
                        pixel_values=pixels, interpolate_pos_encoding=True))
                    base_features, adapted = model.encode_image(pixels)
                    torch.testing.assert_close(base_features, F.normalize(original_base.float(), dim=-1))
                    torch.testing.assert_close(base_features, adapted)
                for step in range(2):
                    optimizer.zero_grad()
                    F.cross_entropy(model(pixels)[0], torch.tensor([0, 1])).backward()
                    grads = [p.grad for name, p in model.named_parameters() if name.endswith("lora_b") or
                             (step == 1 and name.endswith("lora_a"))]
                    self.assertEqual(len(grads), 72 if step == 0 else 144)
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
                    optimizer.step()
                for name, p in model.named_parameters():
                    if name in before:
                        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
                checkpoint = {"format_version": 18, "config": values,
                              "class_names": ["0000", "0001", "0002"],
                              "model": trainable_state_dict(model),
                              "trainable_parameter_names": trainable_parameter_names(model)}
                with patch("v18_model.load_clip", return_value=(pristine, processor())), \
                     patch("v18_model.model_identity", return_value={"test": "base"}), \
                     patch("v18_model.validate_clip_vit_b32"):
                    restored, prep = build_classifier_from_checkpoint(checkpoint, "unused", torch.device("cpu"))
                self.assertEqual(prep.image_processor.crop_size["height"], values["image_size"])
                self.assertEqual(checkpoint_resolution(checkpoint), values["image_size"])
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(pixels)[0], restored(pixels)[0], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "expected"):
                    restored(torch.zeros(1, 3, 224, 224))

    def test_strict_state_validation_rejects_before_mutation(self):
        model = build_classifier(tiny_clip(), np.eye(3, 16), torch.device("cpu"), config("rank32"))
        state = trainable_state_dict(model)
        first = next(name for name in state if name.endswith("lora_b"))
        invalid_states = []
        missing = dict(state); del missing[first]; invalid_states.append(missing)
        invalid_states.append({**state, "clip.surprise": torch.ones(1)})
        invalid_states.append({**state, first: state[first].to(torch.int64)})
        invalid_states.append({**state, first: state[first][:, :16]})
        invalid_states.append({**state, "logit_scale": torch.ones(1)})
        for invalid in invalid_states:
            with self.assertRaises(RuntimeError):
                load_classifier_state(model, invalid)
            for name, value in trainable_state_dict(model).items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        exact = {name: state[name] for name in trainable_parameter_names(model)}
        validate_trainable_state(model, exact, exact=True)
        with self.assertRaises(RuntimeError):
            validate_trainable_state(model, state, exact=True)

    def test_official_metadata_shapes_counts_and_bad_rank(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                values = config(recipe)
                shapes = _official_trainable_shapes(750, values)
                count = sum(int(np.prod(shape)) for name, shape in shapes.items() if name.startswith("clip."))
                self.assertEqual(count, 5308416 if recipe == "rank32" else 2654208)
                checkpoint = {"format_version": 18, "config": values,
                              "class_names": [f"{i:04d}" for i in range(750)],
                              "trainable_parameter_names": sorted(shapes),
                              "model": {name: torch.empty(shape) for name, shape in
                                        {**shapes, "logit_scale": ()}.items()}}
                self.assertEqual(validate_checkpoint_state_metadata(checkpoint), sorted(shapes))
                bad = {**checkpoint, "config": {**values, "mlp_lora_rank": 64}}
                with self.assertRaisesRegex(ValueError, "mlp_lora_rank"):
                    validate_checkpoint_state_metadata(bad)
                bad = {**checkpoint, "model": {**checkpoint["model"], "unexpected": torch.ones(1)}}
                with self.assertRaisesRegex(ValueError, "keys"):
                    validate_checkpoint_state_metadata(bad)
                bad = {**checkpoint, "trainable_parameter_names": sorted(shapes)[1:]}
                with self.assertRaisesRegex(ValueError, "trainable_parameter_names"):
                    validate_checkpoint_state_metadata(bad)
                name = next(name for name in shapes if name.endswith("lora_a"))
                bad = {**checkpoint, "model": {**checkpoint["model"], name: torch.zeros(1)}}
                with self.assertRaisesRegex(ValueError, "shape"):
                    validate_checkpoint_state_metadata(bad)

    def test_old_checkpoint_loading_resolution_and_metadata_delegate(self):
        for version in (9, 11, 13, 14, 15, 16, 17):
            checkpoint = {"format_version": version}
            with patch("v18_model.legacy_classifier", return_value=("legacy", "processor")) as old:
                self.assertEqual(build_classifier_from_checkpoint(checkpoint, "unused", "cpu"),
                                 ("legacy", "processor"))
                old.assert_called_once_with(checkpoint, "unused", "cpu")
            with patch("v18_model.legacy_resolution", return_value=320) as old:
                self.assertEqual(checkpoint_resolution(checkpoint), 320)
                old.assert_called_once_with(checkpoint)
            if version in (15, 16, 17):
                with patch(f"v{version}_model.validate_checkpoint_state_metadata", return_value=["legacy"]) as old:
                    self.assertEqual(validate_checkpoint_state_metadata(checkpoint), ["legacy"])
                    old.assert_called_once_with(checkpoint)


if __name__ == "__main__":
    unittest.main()
