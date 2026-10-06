"""Exact V16 ablations, optimizer coverage, views and complete state contracts."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from robust_clip import LoRALinear, trainable_state_dict
from test_v14_core import processor, tiny_clip
from v11_resolution import HighResolutionPairedDataset, resolution_processor
from v15_core import recipe_config as baseline_recipe
from v16_core import (RECIPES, LAYER_NORM_CONFIG_KEYS, recipe_config, validate_recipe_config,
                      layernorm_parameter_names, build_optimizer)
from v16_model import (build_classifier, build_classifier_from_checkpoint, checkpoint_resolution,
                       load_classifier_state, trainable_parameter_names, validate_trainable_metadata,
                       validate_trainable_state, validate_checkpoint_state_metadata, _official_trainable_shapes)
from v16_runtime import scientific_config
from v16_views import training_dataset


def model_config(recipe="mlp_light"):
    return {**recipe_config(recipe), "image_size": 320, "zoom_shortest_edge": 366,
            "interpolate_pos_encoding": True, "base_model_identity": {"test": "base"}}


class V16CoreTests(unittest.TestCase):
    def test_only_planned_scientific_changes(self):
        a, b = (recipe_config(recipe) for recipe in RECIPES)
        unchanged = {key: value for key, value in a.items() if key not in LAYER_NORM_CONFIG_KEYS}
        unchanged.update(recipe="expanded_mlp", augmentation="robust_quality_320_view_b")
        self.assertEqual(unchanged, baseline_recipe("expanded_mlp"))
        self.assertEqual({key for key in a if a[key] != b[key]}, {"recipe", "layernorm_layers"})
        for config in (a, b):
            self.assertEqual(config["augmentation"], "light_320_both_views")
            self.assertEqual(config["layernorm_lr"], 1e-5)
            self.assertEqual(config["layernorm_weight_decay"], 0.0)
            self.assertEqual(config["schedule_epochs"], 24)
            validate_recipe_config({**config, "lora_targets": list(config["lora_targets"]),
                                    "mlp_lora_targets": list(config["mlp_lora_targets"])})
        for key, value in (("layernorm_layers", 12), ("layernorm_lr", 1e-4),
                           ("layernorm_weight_decay", 1e-4), ("augmentation", "robust_quality_320_view_b"),
                           ("mlp_lora_rank", 8), ("repair_confidence", .8)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "conflicts"):
                validate_recipe_config({**b, key: value})
        with self.assertRaisesRegex(ValueError, "unknown V16 recipe"):
            recipe_config("expanded_mlp")

    def test_same_light_views_for_both_recipes(self):
        for recipe in RECIPES:
            self.assertIs(type(training_dataset([], [], resolution_processor(processor()), model_config(recipe))),
                          HighResolutionPairedDataset)
        with self.assertRaisesRegex(ValueError, "320-pixel"):
            training_dataset([], [], processor(), model_config())

    def test_exact_ln_scope_and_optimizer_coverage(self):
        for recipe in RECIPES:
            config = model_config(recipe)
            model = build_classifier(tiny_clip(), np.eye(3, 16), torch.device("cpu"), config)
            self.assertEqual(sum(isinstance(module, LoRALinear) for module in model.modules()), 72)
            norms = layernorm_parameter_names(config)
            actual = {name for name, p in model.named_parameters()
                      if p.requires_grad and name.startswith("clip.") and not name.endswith(("lora_a", "lora_b"))}
            self.assertEqual(actual, norms)
            self.assertEqual(len(norms), 16 if recipe.endswith("_ln") else 0)
            opt, visual, head = build_optimizer(model, config)
            flat = [p for group in opt.param_groups for p in group["params"]]
            self.assertEqual(len(flat), len({id(p) for p in flat}))
            self.assertEqual({id(p) for p in flat}, {id(p) for p in model.parameters() if p.requires_grad})
            self.assertEqual(len(opt.param_groups), 14 if norms else 13)
            self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-4 * .8**11)
            self.assertEqual(opt.param_groups[11]["lr"], 1e-4)
            self.assertEqual(opt.param_groups[-1]["lr"], 5e-4)
            if norms:
                group = opt.param_groups[-2]
                self.assertEqual((group["lr"], group["weight_decay"]), (1e-5, 0.0))
                self.assertEqual({id(p) for p in group["params"]},
                                 {id(p) for name, p in model.named_parameters() if name in norms})
            self.assertEqual(len(visual) + len(head), len(flat))
            model.clip.vision_model.post_layernorm.weight.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "unexpected unfrozen"):
                build_optimizer(model, config)

    def test_metadata_and_loading_reject_missing_ln_before_any_mutation(self):
        config = model_config("mlp_light_ln")
        model = build_classifier(tiny_clip(), np.eye(3, 16), torch.device("cpu"), config)
        state = trainable_state_dict(model)
        names = trainable_parameter_names(model)
        validate_trainable_metadata(model, {"trainable_parameter_names": names})
        validate_trainable_state(model, {name: state[name] for name in names}, exact=True)
        for missing in sorted(layernorm_parameter_names(config)):
            damaged = {name: value + 1 for name, value in state.items() if name != missing}
            with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, "trainable state mismatch"):
                load_classifier_state(model, damaged)
            for name, value in trainable_state_dict(model).items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            load_classifier_state(model, {**state, "head_unknown": torch.ones(1)})
        with self.assertRaisesRegex(RuntimeError, "missing"):
            load_classifier_state(model, {name: value for name, value in state.items() if name != "logit_scale"})
        a = build_classifier(tiny_clip(), np.eye(3, 16), torch.device("cpu"), model_config())
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            load_classifier_state(a, state)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            load_classifier_state(model, trainable_state_dict(a))

    def test_official_shape_metadata_includes_exact_ln_tensors(self):
        for recipe in RECIPES:
            config = model_config(recipe)
            shapes = _official_trainable_shapes(3, config)
            norms = layernorm_parameter_names(config)
            self.assertTrue(all(shapes[name] == (768,) for name in norms))
            self.assertEqual(sum(np.prod(shapes[name]) for name in norms), 12288 if norms else 0)
            state = {name: torch.zeros(shape) for name, shape in {**shapes, "logit_scale": ()}.items()}
            payload = {"format_version": 16, "config": config, "class_names": ["0", "1", "2"],
                       "model": state, "trainable_parameter_names": sorted(shapes)}
            self.assertEqual(validate_checkpoint_state_metadata(payload), sorted(shapes))
            for name in norms or {"clip.vision_model.encoder.layers.11.mlp.fc2.lora_b"}:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "model keys"):
                    validate_checkpoint_state_metadata({**payload, "model": {k: v for k, v in state.items() if k != name}})
            with self.assertRaisesRegex(ValueError, "trainable_parameter_names"):
                other = "mlp_light" if norms else "mlp_light_ln"
                validate_checkpoint_state_metadata({**payload, "config": model_config(other)})

    def test_runtime_settings_do_not_allow_cross_recipe_resume(self):
        a = {**model_config(), "batch_size": 256, "gradient_accumulation": 1}
        adjusted = {**a, "batch_size": 128, "gradient_accumulation": 2, "workers": 0}
        self.assertEqual(scientific_config(a), scientific_config(adjusted))
        b = {**a, **recipe_config("mlp_light_ln")}
        self.assertNotEqual(scientific_config(a), scientific_config(b))

    def test_checkpoint_builder_preserves_old_version_fallback(self):
        with patch("v16_model.legacy_classifier", return_value=("old", "processor")) as legacy:
            self.assertEqual(build_classifier_from_checkpoint({"format_version": 15}, "unused", "cpu"),
                             ("old", "processor"))
            legacy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
