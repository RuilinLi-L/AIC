"""V17 evidence gates, independent ablations, fold isolation and exact model state."""

from copy import deepcopy
import unittest

import numpy as np
import torch

from robust_clip import LoRALinear, trainable_state_dict
from test_v14_core import processor, tiny_clip
from v11_resolution import resolution_processor
from v14_views import Robust320PairedDataset
from v15_core import recipe_config as baseline_recipe
from v17_core import (
    RECIPES, V17_CONFIG_KEYS, recipe_config, validate_recipe_config, build_optimizer,
    agreement_recovery_quality, cross_fitted_dynamic_prototypes,
    dynamic_prototype_repair_plan, dynamic_repair_plan,
)
from v17_model import (
    build_classifier, load_classifier_state, trainable_parameter_names,
    validate_checkpoint_state_metadata, _official_trainable_shapes,
)
from v17_views import training_dataset
from v8_neighbors import NeighborEvidence


def neighbor(n, labels=None):
    return NeighborEvidence(np.zeros(n, np.float32), np.full(n, 2) if labels is None else labels.copy(),
                            np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.int16))


def model_config(recipe=RECIPES[0]):
    return {**recipe_config(recipe), "image_size": 320, "zoom_shortest_edge": 366,
            "interpolate_pos_encoding": True, "base_model_identity": {"test": "base"}}


class V17CoreTests(unittest.TestCase):
    def test_only_independent_noise_changes_and_robust_views(self):
        a, b = [recipe_config(recipe) for recipe in RECIPES]
        for config in (a, b):
            comparable = {key: value for key, value in config.items() if key not in V17_CONFIG_KEYS}
            comparable["recipe"] = "expanded_mlp"
            self.assertEqual(comparable, baseline_recipe("expanded_mlp"))
            validate_recipe_config(config)
            self.assertIs(type(training_dataset([], [], resolution_processor(processor()), config)), Robust320PairedDataset)
        self.assertEqual({key for key in a if a[key] != b[key]}, {"recipe", "agreement_recovery", "dynamic_prototype"})
        for key, value in (("recovery_confidence", .8), ("dynamic_min_references", 2),
                           ("dynamic_repair_confidence", .85), ("augmentation", "light_320_both_views"),
                           ("layernorm_layers", 4)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe_config({**a, key: value})

    def test_recovery_static_gate_schedule_exclusions_and_withdrawal(self):
        n = 8
        labels = np.zeros(n, np.int64)
        current = np.tile([.95, .04, .01], (n, 1)).astype(np.float32)
        previous = current.copy(); previous[4] = [.89, .10, .01]
        consensus = labels.copy(); consensus[5] = 1
        proto = labels.copy(); proto[[1, 2]] = 1
        evidence = neighbor(n); evidence.top_label[2] = 0; evidence.label_support[2] = .5
        repairs = np.zeros(n, bool); repairs[3] = True
        quality = np.full(n, .2, np.float32)
        common = (labels, np.arange(6), quality, repairs, proto, evidence, previous, current, consensus)
        for epoch, value in ((4, .2), (5, .3), (6, .4), (7, .5), (8, .6), (24, .6)):
            result, mask = agreement_recovery_quality(*common, epoch=epoch)
            np.testing.assert_array_equal(np.where(mask)[0], [0, 2] if epoch >= 5 else [])
            np.testing.assert_allclose(result[[0, 2]], value)
            np.testing.assert_allclose(result[[1, 3, 4, 5, 6, 7]], .2)
        np.testing.assert_allclose(quality, .2)
        current[0] = [.1, .85, .05]
        result, mask = agreement_recovery_quality(*common, epoch=9)
        self.assertFalse(mask[0]); self.assertEqual(result[0], quality[0])

    def test_dynamic_prototypes_exclude_query_fold_validation_and_untrusted(self):
        labels = np.repeat(np.arange(2), 7)
        ids = np.array([*range(6), *range(7, 13)])
        folds = np.full(len(labels), -1, np.int16)
        folds[ids] = np.tile(np.arange(6) % 5, 2)
        features = np.eye(2, 4, dtype=np.float32)[labels]
        probabilities = np.full((len(labels), 2), .05, np.float32)
        probabilities[np.arange(len(labels)), labels] = .95
        trusted = np.ones(len(labels), bool)
        features[0] = features[7]
        probabilities[0] = [.05, .95]
        common = (labels, ids, folds, trusted, probabilities, probabilities)
        result = cross_fitted_dynamic_prototypes(features, *common, num_classes=2)
        self.assertEqual(result['dynamic_prototype_label'][0], 1)
        self.assertGreater(result['dynamic_prototype_margin'][0], .03)
        self.assertFalse(result['dynamic_reference_mask'][0])
        self.assertFalse(result['dynamic_reference_mask'][[6, 13]].any())
        self.assertFalse(result['dynamic_prototype_valid'][[6, 13]].any())
        altered = features.copy(); altered[[5, 6, 12, 13]] = [0, 0, 1, 0]
        other = cross_fitted_dynamic_prototypes(altered, *common, num_classes=2)
        np.testing.assert_array_equal(result['dynamic_prototypes'][0], other['dynamic_prototypes'][0])
        trusted[labels == 1] = False
        sparse = cross_fitted_dynamic_prototypes(features, *common, num_classes=2)
        self.assertFalse(sparse['dynamic_prototype_valid'].any())
        self.assertTrue((sparse['dynamic_prototype_label'] == -1).all())

    def test_dynamic_repair_union_uses_one_cap_static_tie_priority_and_thresholds(self):
        n = 20; labels = np.zeros(n, np.int64); ids = np.arange(n)
        current = np.tile([.9, .05, .05], (n, 1)).astype(np.float32)
        current[:6] = [.15, .8, .05]
        previous = current.copy(); previous[4] = [.26, .69, .05]
        consensus = current.argmax(1)
        proto = labels.copy(); proto[[1, 3]] = 1
        evidence = neighbor(n)
        dyn_labels = np.full(n, -1); dyn_labels[:6] = 1
        valid = np.zeros(n, bool); valid[:6] = True
        margins = np.zeros(n, np.float32); margins[:6] = .03; margins[5] = .029
        plan, diagnostics = dynamic_prototype_repair_plan(
            labels, ids, proto, evidence, previous, current, consensus, dyn_labels, margins, valid)
        # Cap is three total: old evidence at 1/3 wins ties, followed by row 0.
        np.testing.assert_array_equal(np.where(plan)[0], [0, 1, 3])
        self.assertEqual(diagnostics['dynamic_only_selected'], 1)
        self.assertEqual(diagnostics['dynamic_only_selected_by_class'], [1, 0, 0])
        self.assertEqual(diagnostics['dynamic_only_selected_by_target_class'], [0, 1, 0])
        self.assertFalse(plan[4]); self.assertFalse(plan[5])
        disabled, _ = dynamic_prototype_repair_plan(
            labels, ids, proto, evidence, previous, current, consensus,
            dyn_labels, margins, np.zeros(n, bool))
        legacy = dynamic_repair_plan(labels, ids, proto, evidence, previous, current, consensus)
        np.testing.assert_array_equal(disabled, legacy)
        with self.assertRaisesRegex(ValueError, 'evidence'):
            dynamic_prototype_repair_plan(labels, ids, proto, evidence, previous, current, consensus,
                                          dyn_labels[:1], margins, valid)

    def test_exact_lora_optimizer_state_and_frozen_layernorms(self):
        for recipe in RECIPES:
            config = model_config(recipe)
            model = build_classifier(tiny_clip(), np.eye(3, 16), torch.device('cpu'), config)
            self.assertEqual(sum(isinstance(m, LoRALinear) for m in model.modules()), 72)
            self.assertFalse(any(p.requires_grad for n, p in model.named_parameters() if 'layer_norm' in n))
            optimizer, visual, head = build_optimizer(model, config)
            flattened = [p for group in optimizer.param_groups for p in group['params']]
            self.assertEqual(len(flattened), len({id(p) for p in flattened}))
            self.assertEqual({id(p) for p in flattened}, {id(p) for p in model.parameters() if p.requires_grad})
            self.assertEqual(len(optimizer.param_groups), 13)
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-4 * .8**11)
            state = trainable_state_dict(model)
            missing = trainable_parameter_names(model)[0]
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                load_classifier_state(model, {n: v for n, v in state.items() if n != missing})
            with self.assertRaisesRegex(RuntimeError, 'unexpected'):
                load_classifier_state(model, {**state, 'clip.vision_model.post_layernorm.weight': torch.ones(32)})
            shapes = _official_trainable_shapes(3, config)
            self.assertEqual(sum(np.prod(shape) for name, shape in shapes.items() if name.startswith('clip.')), 2654208)
            payload = {'format_version': 17, 'config': config, 'class_names': ['0', '1', '2'],
                       'trainable_parameter_names': sorted(shapes),
                       'model': {n: torch.zeros(shape) for n, shape in {**shapes, 'logit_scale': ()}.items()}}
            validate_checkpoint_state_metadata(payload)
            model.clip.vision_model.post_layernorm.weight.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, 'unexpected unfrozen'):
                build_optimizer(model, config)

    def test_b_strict_floor_cap_for_tiny_classes_preserves_a_legacy_policy(self):
        for n, expected in ((3, 0), (6, 0), (7, 1)):
            labels = np.zeros(n, np.int64)
            probability = np.tile([.05, .90, .05], (n, 1)).astype(np.float32)
            prototype = np.ones(n, np.int64)
            evidence = neighbor(n)
            plan, _ = dynamic_prototype_repair_plan(
                labels, np.arange(n), prototype, evidence, probability, probability, prototype,
                prototype, np.full(n, .1, np.float32), np.ones(n, bool))
            self.assertEqual(int(plan.sum()), expected)
            # A delegates unchanged to V15, whose established policy allows one.
            legacy = dynamic_repair_plan(labels, np.arange(n), prototype, evidence,
                                         probability, probability, prototype)
            self.assertEqual(int(legacy.sum()), 1)


if __name__ == '__main__':
    unittest.main()
