"""V17 behavioral checks: routing, noise correction, LoRA and selection isolation."""
from copy import deepcopy
import json
import random
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPConfig, CLIPModel

from robust_clip import FolderImageDataset, trainable_state_dict
from train_v5 import clone_trainable
from train_v6 import build_optimizer as legacy_optimizer, make_v6_scheduler
from train_v17 import (fit, run_epoch, load_resume, _restore_rng, audit_ema_coverage,
                       prepare_epoch_supervision, _class_prior)
from v6_core import PrototypeSignals
from v8_neighbors import NeighborEvidence
from v12_runtime import BatchedFeatureQueue
from v17_core import (recipe_config, build_optimizer, stage_indices, pool_signature, should_audit,
                      dynamic_repair_plan, repaired_soft_loss, repair_weight, file_sha256)
from v17_model import (build_classifier, build_classifier_from_checkpoint, resolution_processor,
                       load_classifier_state, trainable_parameter_names)
from test_v11 import processor
from test_v12_runtime import Tiny


def clip():
    return CLIPModel(CLIPConfig(projection_dim=16,
        vision_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 12,
                       'num_attention_heads': 4, 'patch_size': 32, 'image_size': 224},
        text_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1,
                     'num_attention_heads': 4, 'vocab_size': 20}))


class V17TrainingTests(unittest.TestCase):
    def test_audit_reuses_two_forwards_and_collects_only_training_features(self):
        network = Tiny().eval()
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for i in range(4):
                path = Path(directory) / f'{i}.png'
                Image.new('RGB', (330, 340), (i * 50, 80, 20)).save(path)
                rows.append((str(path), i % 3, str(i % 3)))
            with patch.object(network, 'forward', wraps=network.forward) as forward:
                result = audit_ema_coverage(network, clone_trainable(network),
                    resolution_processor(processor()), rows, [0, 2], torch.device('cpu'),
                    batch_size=2, workers=0, collect_features=True)
            self.assertEqual(forward.call_count, 2)
            self.assertEqual(len(result), 6)
            self.assertEqual(result[5].shape, (4, 8))
            self.assertTrue((result[5][[1, 3]] == 0).all())
            np.testing.assert_allclose(np.linalg.norm(result[5][[0, 2]].astype(np.float32), axis=1), 1, atol=.001)
            self.assertFalse(network.training)

    def test_epoch_recovery_ramp_prior_and_repair_precedence(self):
        n = 8; labels = np.repeat(np.arange(2), 4); q_base = np.full(n, .2, np.float32)
        prob = np.eye(2, dtype=np.float32)[labels] * .9 + .05
        prob[0] = [.05, .95]
        proto_labels = labels.copy(); proto_labels[0] = 1
        signals = SimpleNamespace(prototype_label=proto_labels)
        neighbor = NeighborEvidence(np.zeros(n), np.full(n, -1), np.zeros(n), np.zeros(n), np.zeros(n))
        arrays = {'audit_previous_prob': prob.copy(), 'audit_current_prob': prob.copy(),
                  'audit_view_consensus': prob.argmax(1), 'audit_previous_epoch': np.asarray(2, np.int64),
                  'audit_epoch': np.asarray(4, np.int64), 'dynamic_prototype_epoch': np.asarray(0, np.int64),
                  'recovery_mask': np.zeros(n, bool), 'dynamic_reference_mask': np.zeros(n, bool),
                  'dynamic_prototype_valid': np.zeros(n, bool), 'dynamic_reference_counts': np.zeros((0, 2), np.int64)}
        config = recipe_config('agreement_recovery')
        plan5, q5, diagnostics = prepare_epoch_supervision(labels, np.arange(n), signals, neighbor,
            q_base, np.zeros(n, bool), arrays, 5, config)
        self.assertTrue(plan5[0]); self.assertFalse(arrays['recovery_mask'][0])
        self.assertEqual(q5[0], q_base[0]); np.testing.assert_allclose(q5[1:], .3)
        plan6, q6, _ = prepare_epoch_supervision(labels, np.arange(n), signals, neighbor,
            q_base, np.zeros(n, bool), arrays, 6, config)
        np.testing.assert_array_equal(plan5, plan6)
        np.testing.assert_allclose(q6[1:], .4)
        prior5 = _class_prior(labels, q5, np.arange(n), np.ones(n), 2)
        prior6 = _class_prior(labels, q6, np.arange(n), np.ones(n), 2)
        self.assertFalse(np.array_equal(prior5, prior6))
        np.testing.assert_allclose(q_base, .2)
        arrays['audit_previous_epoch'] = arrays['audit_epoch'].copy()
        with self.assertRaisesRegex(ValueError, 'distinct completed prior audits'):
            prepare_epoch_supervision(labels, np.arange(n), signals, neighbor,
                q_base, np.zeros(n, bool), arrays, 7, config)

    def test_q_train_does_not_reach_regularizers_or_trust_queue(self):
        import train_v17
        n = 3; labels = np.arange(3); q_base = np.array([.3, .4, .5], np.float32)
        torch.manual_seed(17)
        pixels = torch.randn(n, 3, 8, 8)
        initial = Tiny()
        teacher = np.eye(3, dtype=np.float32) * .9 + .1 / 3
        batch = {'pixel_values_a': pixels, 'pixel_values_b': pixels * .8 + .1,
                 'label': torch.arange(n), 'row_index': torch.arange(n)}
        records = []
        add = train_v17._add_gradients
        kl = train_v17.symmetric_kl_loss
        for supervision in (q_base, np.ones(n, np.float32)):
            network = deepcopy(initial); optimizer, _, _ = legacy_optimizer(network)
            queue = BatchedFeatureQueue(3, 2, 8, torch.device('cpu'))
            record = {}
            def capture_gradients(parameters, trusted_grads, uncertain_grads, regularizer_grads, count):
                record['regularizer_grads'] = [None if g is None else g.detach().clone() for g in regularizer_grads]
                return add(parameters, trusted_grads, uncertain_grads, regularizer_grads, count)
            def capture_consistency(*args):
                value = kl(*args)
                record['consistency'] = value.detach().clone()
                return value
            with patch('train_v17.soft_teacher_regularizer', wraps=train_v17.soft_teacher_regularizer) as soft, \
                 patch('train_v17.selective_supervised_contrastive_loss', wraps=train_v17.selective_supervised_contrastive_loss) as contrastive, \
                 patch('train_v17.symmetric_kl_loss', side_effect=capture_consistency), \
                 patch('train_v17._add_gradients', side_effect=capture_gradients):
                stats, _ = run_epoch(network, [batch], optimizer, make_v6_scheduler(optimizer, 24),
                    clone_trainable(network), queue, torch.device('cpu'), 8, labels, q_base, np.zeros(n, bool),
                    torch.ones(3) / 3, np.eye(n, 8, dtype=np.float32), torch.eye(3, 8),
                    np.zeros(n, bool), teacher.copy(), np.ones(n, bool), np.zeros(n, np.int64),
                    np.zeros(n), np.zeros(n, np.int16), np.zeros(n), np.zeros(n, np.int16), 0,
                    np.zeros(n, bool), [(i,) for i in labels], 1, True, dynamic_noise=True,
                    audit_probabilities=teacher, supervision_quality=supervision)
            np.testing.assert_array_equal(soft.call_args.args[2].numpy(), q_base)
            self.assertFalse(contrastive.call_args.args[3].any())
            self.assertFalse(queue.valid.any())
            self.assertEqual(stats['soft_teacher_rows'], 3)
            record['loss'] = stats['loss']
            records.append(record)
        self.assertNotEqual(records[0]['loss'], records[1]['loss'])
        torch.testing.assert_close(records[0]['consistency'], records[1]['consistency'], rtol=0, atol=0)
        for left, right in zip(records[0]['regularizer_grads'], records[1]['regularizer_grads']):
            if left is None:
                self.assertIsNone(right)
            else:
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_expanded_gradients_frozen_weights_and_roundtrip(self):
        torch.manual_seed(17)
        base = clip()
        pristine = deepcopy(base)
        config = {**recipe_config('agreement_recovery'), 'image_size': 320, 'zoom_shortest_edge': 366,
                  'interpolate_pos_encoding': True, 'base_model_identity': {'test': 'base'}}
        model = build_classifier(base, np.eye(3, 16), torch.device('cpu'), config)
        before = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
        optimizer, visual, head = build_optimizer(model, config)
        self.assertEqual(len(optimizer.param_groups), 13)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-4 * .8**11)
        self.assertEqual(optimizer.param_groups[-2]['lr'], 1e-4)
        self.assertEqual(optimizer.param_groups[-1]['lr'], 5e-4)
        pixels = torch.randn(2, 3, 320, 320)
        for _ in range(2):
            optimizer.zero_grad()
            F.cross_entropy(model(pixels)[0], torch.tensor([0, 1])).backward()
            grads = [p.grad for n, p in model.named_parameters() if n.endswith('lora_b')]
            self.assertEqual(len(grads), 72)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
            optimizer.step()
        for n, p in model.named_parameters():
            if n in before:
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        checkpoint = {'format_version': 17, 'config': config, 'class_names': ['0000', '0001', '0002'],
                      'model': trainable_state_dict(model), 'trainable_parameter_names': trainable_parameter_names(model)}
        with patch('v17_model.load_clip', return_value=(pristine, processor())), \
             patch('v17_model.model_identity', return_value={'test': 'base'}), \
             patch('v17_model.validate_clip_vit_b32'):
            restored, _ = build_classifier_from_checkpoint(checkpoint, 'unused', torch.device('cpu'))
        self.assertEqual(sum('.mlp.' in name for name in trainable_parameter_names(model)), 48)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(pixels)[0], restored(pixels)[0], rtol=0, atol=0)

    def test_repaired_rows_ignore_old_label_and_detach_teacher(self):
        n, dim, classes = 3, 8, 3
        torch.manual_seed(5)
        initial = Tiny()
        pixels = torch.randn(n, 3, 8, 8)
        teacher = torch.tensor([[.05, .90, .05]] * n, requires_grad=True)
        a = torch.randn(n, classes, requires_grad=True)
        b = torch.randn(n, classes, requires_grad=True)
        repaired_soft_loss(a, b, teacher, torch.ones(n, dtype=torch.bool), .5).sum().backward()
        self.assertIsNone(teacher.grad)
        results = []
        for old_label in (0, 2):
            network = deepcopy(initial)
            optimizer, _, _ = legacy_optimizer(network)
            scheduler = make_v6_scheduler(optimizer, 24)
            queue = BatchedFeatureQueue(classes, 2, dim, torch.device('cpu'))
            batch = {'pixel_values_a': pixels, 'pixel_values_b': pixels,
                     'label': torch.full((n,), old_label), 'row_index': torch.arange(n)}
            stats, _ = run_epoch(network, [batch], optimizer, scheduler, clone_trainable(network), queue,
                torch.device('cpu'), 8, np.full(n, old_label), np.full(n, .3), np.ones(n, bool),
                torch.ones(classes) / classes, np.eye(n, dim, dtype=np.float32), torch.eye(classes, dim),
                np.ones(n, bool), teacher.detach().numpy().copy(), np.ones(n, bool), np.zeros(n, dtype=np.int64),
                np.zeros(n), np.zeros(n, dtype=np.int16), np.zeros(n), np.zeros(n, dtype=np.int16), 0,
                np.zeros(n, bool), [(old_label,)] * n, 1, True, dynamic_noise=True,
                audit_probabilities=teacher.detach().numpy(), schedule_epochs=24,
                supervision_quality=np.full(n, .2 if old_label == 0 else .95, np.float32))
            self.assertEqual(stats['repaired_rows'], n)
            self.assertEqual(stats['soft_teacher_rows'], 0)
            self.assertFalse(queue.valid.any())
            results.append(network.state_dict())
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)

    def test_periodic_audit_and_exact_epoch_resume(self):
        self._check_periodic_resume('agreement_recovery')

    def test_dynamic_features_prototypes_and_exact_epoch_resume(self):
        self._check_periodic_resume('dynamic_prototype')

    def _check_periodic_resume(self, recipe):
        cpu, n = torch.device('cpu'), 9
        labels = np.arange(n) % 3
        ones = np.ones(n, np.float32)
        proto = np.eye(3, 8, dtype=np.float32)
        signals = PrototypeSignals(proto, np.zeros(n), ones, ones, ones, labels.copy(), ones, ones)
        neighbor = NeighborEvidence(ones, labels.copy(), ones, ones, np.zeros(n))
        config = {**recipe_config(recipe), 'version': 'v17', 'stage': 'validate', 'soft_teacher': True,
                  'precision': 'fp32', 'workers': 0, 'batch_size': 256, 'gradient_accumulation': 1,
                  'prefetch_factor': 2, 'eval_batch_size': 3}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = []
            for i in range(n):
                path = root / f'{i}.png'; Image.new('RGB', (330, 350), (i*20, 90, 30)).save(path)
                rows.append((str(path), int(labels[i]), f'{labels[i]:04d}'))
            prep = resolution_processor(processor())
            common = dict(processor=prep, rows=rows, train_indices=list(range(6)), supervised_indices=list(range(6)),
                labels=labels, view_features=np.stack([proto[labels]] * 4), signals=signals, quality=ones.copy()*.8,
                validation_loader=DataLoader(FolderImageDataset(rows, [6, 7, 8], prep), batch_size=3),
                trusted_validation=np.ones(n, bool), class_counts=np.array([2, 2, 2]),
                class_names=['0000', '0001', '0002'], config=config, dataset_digest='tiny', device=cpu,
                batch_size=256, workers=0, stage='validate', validation_indices=[6, 7, 8],
                ambiguous_mask=np.zeros(n, bool), candidate_labels=[(int(x),) for x in labels],
                sampler_mode='shuffle', row_repeat_factors=np.ones(n), gradient_accumulation=1, neighbor_evidence=neighbor)
            torch.manual_seed(77); initial = Tiny()
            with patch('train_v17.build_optimizer', side_effect=lambda model, cfg: legacy_optimizer(model)):
                full = root / 'full'; full.mkdir()
                torch.manual_seed(123); random.seed(123)
                result = fit(classifier=deepcopy(initial), output_dir=full, resume=None, epochs=6, **deepcopy(common))
                split = root / 'split'; split.mkdir()
                torch.manual_seed(123); random.seed(123)
                fit(classifier=deepcopy(initial), output_dir=split, resume=None, epochs=4, **deepcopy(common))
                state = load_resume(str(split / 'resume_latest.pt'), config, 'tiny', cpu)
                resumed = fit(classifier=deepcopy(initial), output_dir=split, resume=state, epochs=6, **deepcopy(common))
            for name, value in result[0].items():
                torch.testing.assert_close(value, resumed[0][name], rtol=0, atol=0)
            latest = torch.load(split / 'resume_latest.pt', map_location='cpu', weights_only=False)
            self.assertTrue(latest['audit_epoch2_seen'][:6].all())
            self.assertFalse(latest['audit_epoch2_seen'][6:].any())
            self.assertTrue((latest['audit_current_prob'][6:] == 0).all())
            self.assertTrue((latest['audit_previous_prob'][6:] == 0).all())
            self.assertEqual([h['epoch'] for h in resumed[-1] if h['quality_refreshed']], [3, 5])
            self.assertEqual(latest['trainable_parameter_names'], trainable_parameter_names(initial))
            self.assertEqual(int(latest['audit_previous_epoch']), 4)
            self.assertEqual(int(latest['audit_epoch']), 6)
            self.assertEqual(latest['format_version'], 17)
            epoch5 = torch.load(split / 'validation_epochs' / 'epoch_05.pt', map_location='cpu', weights_only=False)
            self.assertEqual(epoch5['noise_provenance']['audit_previous_epoch'], 2)
            self.assertEqual(epoch5['noise_provenance']['audit_epoch'], 4)
            self.assertEqual(epoch5['quality_semantics'], 'base_reliability_not_recovered_supervision')
            if recipe == 'dynamic_prototype':
                self.assertTrue(np.isfinite(latest['audit_features']).all())
                self.assertTrue(np.allclose(np.linalg.norm(latest['audit_features'][:6].astype(np.float32), axis=1), 1., atol=.001))
                self.assertTrue((latest['audit_features'][6:] == 0).all())
            for missing in ('supervision_quality', 'recovery_mask', 'audit_features', 'dynamic_prototypes', 'audit_previous_epoch'):
                damaged = deepcopy(latest); del damaged[missing]
                with self.subTest(missing=missing), \
                     patch('train_v17.build_optimizer', side_effect=lambda model, cfg: legacy_optimizer(model)), \
                     self.assertRaisesRegex(ValueError, 'missing noise state'):
                    fit(classifier=deepcopy(initial), output_dir=split, resume=damaged, epochs=7, **deepcopy(common))
            for field, value in (('audit_previous_epoch', 6), ('audit_epoch', 7)):
                damaged = deepcopy(latest); damaged[field] = np.asarray(value, np.int64)
                with self.subTest(field=field), \
                     patch('train_v17.build_optimizer', side_effect=lambda model, cfg: legacy_optimizer(model)), \
                     self.assertRaisesRegex(ValueError, 'audit epochs'):
                    fit(classifier=deepcopy(initial), output_dir=split, resume=damaged, epochs=7, **deepcopy(common))
            damaged = deepcopy(latest); damaged['recovery_mask'][6] = True
            with patch('train_v17.build_optimizer', side_effect=lambda model, cfg: legacy_optimizer(model)), \
                 self.assertRaisesRegex(ValueError, 'outside supervised'):
                fit(classifier=deepcopy(initial), output_dir=split, resume=damaged, epochs=7, **deepcopy(common))
            for state_key in ('model', 'ema', 'best_state'):
                damaged = deepcopy(latest)
                del damaged[state_key][latest['trainable_parameter_names'][0]]
                with self.subTest(state=state_key), \
                     patch('train_v17.build_optimizer', side_effect=lambda model, cfg: legacy_optimizer(model)), \
                     self.assertRaisesRegex(RuntimeError, 'trainable state mismatch'):
                    fit(classifier=deepcopy(initial), output_dir=split, resume=damaged, epochs=7, **deepcopy(common))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_resume(str(split / 'resume_latest.pt'), {**config, 'stage': 'refit'}, 'tiny', cpu)

    def test_refit_rejects_a_fallback_before_loading_data(self):
        import train_v17
        args = SimpleNamespace(batch_size=256, workers=0, gradient_accumulation=1, prefetch_factor=1,
                               eval_batch_size=256, stage='refit', selection_json='selection.json',
                               neighbor_cache=None, train_dir='data', output_dir='output')
        with patch('train_v17.parse_args', return_value=args), \
             patch('select_v17.load_selection', return_value={'recipe': 'expanded_robust'}), \
             patch('train_v17.load_manifest_dataset') as manifest, \
             self.assertRaisesRegex(ValueError, 'winning V17 recipe'):
            train_v17.main()
        manifest.assert_not_called()

    def test_refit_main_rebuilds_pool_without_validation_and_freezes_calibration(self):
        import train_v17
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / 'images'; data.mkdir()
            rows, records = [], []
            for i in range(18):
                path = data / f'{i:02d}.png'
                Image.new('RGB', (330, 340), (i*10, 80, 20)).save(path)
                rows.append((str(path), i % 3, f'{i % 3:04d}'))
                records.append({'relative_path': path.name, 'role': 'clean', 'sha256': str(i)})
            labels = np.arange(18) % 3
            manifest = SimpleNamespace(rows=rows, class_names=['0000', '0001', '0002'],
                train_indices=list(range(15)), final_indices=list(range(18)), validation_indices=[15, 16, 17],
                clean_mask=np.ones(18, bool), ambiguous_mask=np.zeros(18, bool),
                candidate_labels=[(int(v),) for v in labels], signature='tiny', summary={})
            source = root / 'selected.pt'; source.write_bytes(b'validation-only-source')
            calibration = {'view_weights': {'center': .5, 'hflip': .5}, 'precision': 'fp32', 'alpha': .2,
                           'selected_epoch': 2, 'image_size': 320}
            selection = {'kind': 'v17_refit_selection', 'format_version': 17, 'recipe': 'agreement_recovery',
                'selected_epoch': 2, 'schedule_epochs': 24, 'dataset_signature': 'tiny',
                'class_names': manifest.class_names, 'base_model_identity': {'base': 'official'},
                'class_bias': [.1, 0., -.1], 'calibration': calibration,
                'source_checkpoint': str(source), 'source_checkpoint_sha256': file_sha256(source),
                'training_config': {'conflict_policy': 'partial', 'sampler': 'shuffle',
                                    'feature_cache': str(root / 'features.npy')}}
            selected = root / 'selection.json'; selected.write_text(json.dumps(selection))
            out = root / 'refit'
            argv = ['train_v17.py', '--stage', 'refit', '--selection-json', str(selected),
                    '--train-dir', str(data), '--data-manifest', str(root / 'manifest.json'),
                    '--model-dir', 'official', '--output-dir', str(out), '--device', 'cpu',
                    '--workers', '0', '--sampler', 'shuffle', '--eval-batch-size', '18']
            features = np.stack([np.eye(3, 16, dtype=np.float32)[labels]] * 4)
            real_neighbors = train_v17.load_or_create_neighbor_evidence
            with patch.object(sys, 'argv', argv), \
                 patch('train_v17.load_manifest_dataset', return_value=manifest), \
                 patch('select_v17.load_selection', return_value=selection), \
                 patch('train_v17.read_manifest', return_value={'rows': records}), \
                 patch('train_v17.model_identity', return_value={'base': 'official'}), \
                 patch('train_v17.load_clip', side_effect=lambda *a: (clip(), processor())), \
                 patch('train_v17.validate_clip_vit_b32'), \
                 patch('train_v17.load_or_create_multiview_features', return_value=features), \
                 patch('train_v17.load_or_create_neighbor_evidence', wraps=real_neighbors) as neighbors, \
                 patch('train_v17.evaluate_two_view_cv', side_effect=AssertionError('refit used validation')):
                train_v17.main()
            self.assertEqual(neighbors.call_args.args[5], list(range(18)))
            self.assertEqual(neighbors.call_args.args[6], [])
            checkpoint = torch.load(out / 'model.pt', map_location='cpu', weights_only=False)
            self.assertTrue(checkpoint['trainable_parameter_names'])
            self.assertEqual(checkpoint['format_version'], 17)
            self.assertEqual(checkpoint['training_stage'], 'refit')
            self.assertEqual(checkpoint['validation']['indices'], [])
            self.assertEqual(checkpoint['config']['schedule_epochs'], 24)
            self.assertEqual(checkpoint['selected_epoch'], 2)
            self.assertEqual(checkpoint['calibration']['view_weights'], calibration['view_weights'])
            self.assertTrue(checkpoint['calibration']['frozen_before_refit'])
            np.testing.assert_allclose(checkpoint['class_bias'], selection['class_bias'])
            self.assertFalse((out / 'validation_epochs').exists())
            # Exercise the real CSV/ZIP entry point without using any training labels.
            import predict_v17
            predicted = root / 'prediction'
            argv = ['predict_v17.py', '--checkpoint', str(out / 'model.pt'), '--model-dir', 'official',
                    '--test-dir', str(data), '--output', str(predicted / 'pred_results.csv'),
                    '--zip-output', str(predicted / 'pred_results.zip'), '--expected-rows', '18',
                    '--device', 'cpu', '--workers', '0', '--batch-size', '18']
            with patch.object(sys, 'argv', argv), \
                 patch('predict_v17.build_classifier_from_checkpoint', return_value=(Tiny().eval(), resolution_processor(processor()))):
                predict_v17.main()
            predict_v17.validate_submission(predicted / 'pred_results.csv', predicted / 'pred_results.zip', 18,
                                            [f'{i:02d}.png' for i in range(18)])


    def test_best_epoch_matches_exact_macro_priority(self):
        from train_v17 import _is_better_cv
        best = {"selection_macro_accuracy": .74, "selection_macro_nll": 1.0, "epoch": 1}
        candidate = {"selection_macro_accuracy": .7404, "selection_macro_nll": 2.0, "epoch": 2}
        self.assertTrue(_is_better_cv(candidate, best))
        self.assertFalse(_is_better_cv(best, candidate))
