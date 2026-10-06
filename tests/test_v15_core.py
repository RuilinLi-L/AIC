"""V15 keeps the scientific recipe and rejects incomplete MLP LoRA states."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from robust_clip import LoRALinear, trainable_state_dict
from test_v14_core import processor, tiny_clip
from v14_core import recipe_config as v14_recipe_config
from v14_views import Robust320PairedDataset
from v15_core import recipe_config, validate_recipe_config, MLP_CONFIG_KEYS
from v15_model import (
    build_classifier, build_classifier_from_checkpoint, checkpoint_resolution,
    load_classifier_state, trainable_parameter_names, validate_trainable_metadata,
    validate_trainable_state, validate_checkpoint_state_metadata, _official_trainable_shapes,
)
from v15_runtime import scientific_config, pipeline_eta, training_history_seconds
from v15_views import training_dataset
from v11_resolution import resolution_processor


def model_config():
    return {**recipe_config("expanded_mlp"), "image_size": 320, "zoom_shortest_edge": 366,
            "interpolate_pos_encoding": True, "base_model_identity": {"test": "base"}}


class V15CoreTests(unittest.TestCase):
    def test_single_recipe_changes_only_name_and_mlp_capacity(self):
        config = recipe_config("expanded_mlp")
        original = {key: value for key, value in config.items() if key not in MLP_CONFIG_KEYS}
        original["recipe"] = "expanded_robust"
        self.assertEqual(original, v14_recipe_config("expanded_robust"))
        self.assertEqual(config["mlp_lora_targets"], ("fc1", "fc2"))
        for key in ("mlp_lora_layers", "mlp_lora_rank"):
            self.assertEqual(config[key], 12 if key.endswith("layers") else 16)
        for recipe in ("expanded", "expanded_robust", "dynamic"):
            with self.assertRaisesRegex(ValueError, "unknown V15 recipe"):
                recipe_config(recipe)
        for key, value in (("mlp_lora_layers", 4), ("mlp_lora_rank", 8), ("mlp_lora_alpha", 16),
                           ("mlp_lora_dropout", .1), ("mlp_lora_targets", ["fc1"]),
                           ("layer_lr_decay", .9), ("augmentation", "light_320_both_views")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "conflicts"):
                validate_recipe_config({**config, key: value})
        json_config = {**config, "lora_targets": list(config["lora_targets"]),
                       "mlp_lora_targets": list(config["mlp_lora_targets"])}
        self.assertEqual(validate_recipe_config(json_config), config)

    def test_runtime_changes_do_not_hide_mlp_or_recipe_changes(self):
        config = {**model_config(), "batch_size": 256, "gradient_accumulation": 1}
        adjusted = {**config, "batch_size": 64, "gradient_accumulation": 4,
                    "workers": 0, "eval_batch_size": 32, "pin_memory": False,
                    "eta_final_rows": 150000, "eta_prior_elapsed_seconds": 1000,
                    "eta_prior_elapsed_known": True}
        self.assertEqual(scientific_config(config), scientific_config(adjusted))
        self.assertNotEqual(scientific_config(config), scientific_config({**config, "mlp_lora_rank": 8}))
        with self.assertRaisesRegex(ValueError, "= 256"):
            scientific_config({**config, "batch_size": 128})

    def test_pipeline_eta_includes_refit_audits_reserve_and_budget(self):
        history = [{"epoch": 1, "train_seconds": 100, "validation_seconds": 10, "audit_seconds": 20},
                   {"epoch": 2, "train_seconds": 120, "validation_seconds": 14, "audit_seconds": 30}]
        eta = pipeline_eta(history, stage="validate", stop_epoch=24, train_rows=100, final_rows=110)
        self.assertEqual(training_history_seconds(history), 294)
        self.assertEqual(eta["remaining_stage_seconds"], 2959)
        np.testing.assert_allclose(eta["refit_seconds_range"], [2453, 3261.5])
        np.testing.assert_allclose(eta["remaining_seconds_range"], [16212, 17020.5])
        np.testing.assert_allclose(eta["estimated_total_seconds_range"], [16506, 17314.5])
        self.assertEqual(eta["refit_epochs_assumption"], [18, 24])
        self.assertFalse(eta["external_dependency_wait_included"])
        self.assertFalse(eta["budget_exceeded"])
        exceeded = pipeline_eta(history, stage="validate", stop_epoch=24, train_rows=100, final_rows=110,
                                prior_elapsed_seconds=48 * 3600)
        self.assertTrue(exceeded["budget_exceeded"])
        self.assertTrue(exceeded["budget_exceeded_at_low_estimate"])
        refit_history = [{key: value for key, value in record.items() if key != "validation_seconds"}
                         for record in history]
        refit = pipeline_eta(refit_history, stage="refit", stop_epoch=18, train_rows=110, final_rows=110,
                             prior_elapsed_seconds=1000, prior_elapsed_known=False)
        self.assertEqual(refit["remaining_seconds_range"], [5560, 5560])
        self.assertEqual(refit["measured_history_seconds"], 1270)
        self.assertFalse(refit["prior_elapsed_known"])

    def test_exact_v14_view_class_with_v15_validation(self):
        prep = resolution_processor(processor())
        self.assertIsInstance(training_dataset([], [], prep, model_config()), Robust320PairedDataset)
        with self.assertRaisesRegex(ValueError, "augmentation"):
            training_dataset([], [], prep, {**model_config(), "augmentation": "wrong"})
        with self.assertRaisesRegex(ValueError, "320-pixel"):
            training_dataset([], [], processor(), model_config())

    def test_72_modules_and_strict_trainable_loading_before_mutation(self):
        torch.manual_seed(15)
        model = build_classifier(tiny_clip(), np.eye(3, 16), torch.device("cpu"), model_config())
        names = trainable_parameter_names(model)
        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]
        self.assertEqual(len(modules), 72)
        self.assertEqual(sum(".mlp." in name for name, _ in modules), 24)
        state = trainable_state_dict(model)
        before = {name: value.clone() for name, value in state.items()}
        for suffix in ("mlp.fc1.lora_a", "mlp.fc2.lora_b", "self_attn.q_proj.lora_b"):
            missing = next(name for name in state if name.endswith(suffix))
            damaged = {name: value + 1 for name, value in state.items() if name != missing}
            with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, "trainable state mismatch"):
                load_classifier_state(model, damaged)
            for name, value in trainable_state_dict(model).items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "trainable_parameter_names"):
            validate_trainable_metadata(model, {})
        with self.assertRaisesRegex(ValueError, "trainable_parameter_names"):
            validate_trainable_metadata(model, {"trainable_parameter_names": names[:-1]})
        validate_trainable_metadata(model, {"trainable_parameter_names": names})
        validate_trainable_state(model, {name: state[name] for name in names}, exact=True)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            validate_trainable_state(model, state, exact=True)
        frozen_name, frozen_parameter = next((name, parameter) for name, parameter in model.named_parameters()
                                             if name.startswith("clip.") and not parameter.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            load_classifier_state(model, {**state, frozen_name: frozen_parameter.clone()})
        load_classifier_state(model, state)

    def test_static_official_metadata_rejects_incomplete_or_wrong_shape_states(self):
        shapes = _official_trainable_shapes(3)
        state = {name: torch.zeros(shape) for name, shape in {**shapes, "logit_scale": ()}.items()}
        payload = {"format_version": 15, "config": model_config(), "class_names": ["0000", "0001", "0002"],
                   "model": state, "trainable_parameter_names": sorted(shapes)}
        self.assertEqual(validate_checkpoint_state_metadata(payload), sorted(shapes))
        mlp_name = "clip.vision_model.encoder.layers.11.mlp.fc2.lora_b"
        damaged = {name: value for name, value in state.items() if name != mlp_name}
        with self.assertRaisesRegex(ValueError, "model keys"):
            validate_checkpoint_state_metadata({**payload, "model": damaged})
        with self.assertRaisesRegex(ValueError, "invalid shape"):
            validate_checkpoint_state_metadata({**payload, "model": {**state, mlp_name: torch.zeros(1)}})
        with self.assertRaisesRegex(ValueError, "trainable_parameter_names"):
            validate_checkpoint_state_metadata({**payload, "trainable_parameter_names": sorted(shapes)[::-1]})
        with self.assertRaisesRegex(ValueError, "model keys"):
            validate_checkpoint_state_metadata({**payload, "model": {**state, "clip.frozen.weight": torch.zeros(1)}})

    def test_checkpoint_requires_metadata_and_official_identity(self):
        base = tiny_clip()
        model = build_classifier(deepcopy(base), np.eye(3, 16), torch.device("cpu"), model_config())
        payload = {"format_version": 15, "config": model_config(), "class_names": ["0000", "0001", "0002"],
                   "model": trainable_state_dict(model), "trainable_parameter_names": trainable_parameter_names(model)}
        self.assertEqual(checkpoint_resolution(payload), 320)
        with patch("v15_model.load_clip", side_effect=lambda *args: (deepcopy(base), processor())), \
             patch("v15_model.model_identity", return_value={"test": "base"}), \
             patch("v15_model.validate_clip_vit_b32"):
            restored, prep = build_classifier_from_checkpoint(payload, "unused", torch.device("cpu"))
            self.assertEqual(prep.image_processor.crop_size, {"height": 320, "width": 320})
            for name, value in trainable_state_dict(restored).items():
                torch.testing.assert_close(value, payload["model"][name], rtol=0, atol=0)
            broken = {key: value for key, value in payload.items() if key != "trainable_parameter_names"}
            with self.assertRaisesRegex(ValueError, "trainable_parameter_names"):
                build_classifier_from_checkpoint(broken, "unused", torch.device("cpu"))
        with patch("v15_model.model_identity", return_value={"wrong": "base"}), \
             self.assertRaisesRegex(ValueError, "official base weights"):
            build_classifier_from_checkpoint(payload, "unused", torch.device("cpu"))
        with patch("v15_model.legacy_classifier", return_value=("legacy", "processor")) as old:
            self.assertEqual(build_classifier_from_checkpoint({"format_version": 14}, "unused", torch.device("cpu")),
                             ("legacy", "processor"))
            old.assert_called_once()


if __name__ == "__main__":
    unittest.main()
