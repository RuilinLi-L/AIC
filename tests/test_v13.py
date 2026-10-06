"""V13 behavioral checks: routing, noise correction, LoRA and selection isolation."""
from copy import deepcopy
import json
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

from robust_clip import FolderImageDataset, trainable_state_dict, load_classifier_state
from train_v5 import clone_trainable
from train_v6 import build_optimizer as legacy_optimizer, make_v6_scheduler
from train_v13 import fit, run_epoch, load_resume, _restore_rng
from v6_core import PrototypeSignals
from v8_neighbors import NeighborEvidence
from v12_runtime import BatchedFeatureQueue
from v13_core import (recipe_config, build_optimizer, stage_indices, pool_signature, should_audit,
                      dynamic_repair_plan, repaired_soft_loss, repair_weight, file_sha256)
from v13_model import build_classifier, build_classifier_from_checkpoint, resolution_processor
from select_v13 import choose_candidate, load_selection, CONDITIONS
from test_v11 import processor
from test_v12_runtime import Tiny


def clip():
    return CLIPModel(CLIPConfig(projection_dim=16,
        vision_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 12,
                       'num_attention_heads': 4, 'patch_size': 32, 'image_size': 224},
        text_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1,
                     'num_attention_heads': 4, 'vocab_size': 20}))


class V13Tests(unittest.TestCase):
    def test_expanded_gradients_frozen_weights_and_roundtrip(self):
        torch.manual_seed(17)
        base = clip()
        pristine = deepcopy(base)
        config = {**recipe_config('expanded'), 'image_size': 320, 'zoom_shortest_edge': 366,
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
            self.assertEqual(len(grads), 48)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
            optimizer.step()
        for n, p in model.named_parameters():
            if n in before:
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        checkpoint = {'format_version': 13, 'config': config, 'class_names': ['0000', '0001', '0002'],
                      'model': trainable_state_dict(model)}
        with patch('v13_model.load_clip', return_value=(pristine, processor())), \
             patch('v13_model.model_identity', return_value={'test': 'base'}), \
             patch('v13_model.validate_clip_vit_b32'):
            restored, _ = build_classifier_from_checkpoint(checkpoint, 'unused', torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(pixels)[0], restored(pixels)[0], rtol=0, atol=0)

    def test_repair_requires_two_audits_and_support_with_small_class_cap(self):
        n = 12
        labels = np.zeros(n, dtype=np.int64)
        labels[10:] = 1
        current = np.tile([.1, .9], (n, 1)).astype(np.float32)
        previous = current.copy()
        evidence = NeighborEvidence(np.zeros(n), np.ones(n, dtype=np.int64), np.ones(n), np.ones(n), np.zeros(n))
        proto = np.ones(n, dtype=np.int64)
        consensus = np.ones(n, dtype=np.int64)
        plan = dynamic_repair_plan(labels, list(range(10)), proto, evidence, previous, current, consensus)
        self.assertEqual(int(plan.sum()), 1)  # 15% of ten, stable tie chooses row zero.
        self.assertTrue(plan[0])
        self.assertFalse(plan[10:].any())  # holdout cannot enter repair
        previous[0] = [.9, .1]
        consensus[1] = -1
        current[2] = [.4, .6]
        proto[3] = 0
        evidence.top_support[3] = .49
        plan = dynamic_repair_plan(labels, [0, 1, 2, 3, 4], proto, evidence, previous, current, consensus)
        self.assertEqual(np.flatnonzero(plan).tolist(), [4])
        self.assertEqual([repair_weight(e) for e in (4, 5, 6, 7, 8, 24)], [0, .125, .25, .375, .5, .5])

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
                audit_probabilities=teacher.detach().numpy(), schedule_epochs=24)
            self.assertEqual(stats['repaired_rows'], n)
            self.assertEqual(stats['soft_teacher_rows'], 0)
            self.assertFalse(queue.valid.any())
            results.append(network.state_dict())
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)

    def test_stage_pool_and_candidate_gates(self):
        manifest = SimpleNamespace(train_indices=[0, 1], validation_indices=[2], final_indices=[0, 1, 2],
                                   clean_mask=np.ones(3, bool))
        self.assertEqual(stage_indices(manifest, 'refit'), ([0, 1, 2], [0, 1, 2], []))
        self.assertNotEqual(pool_signature('d', [0, 1], [2]), pool_signature('d', [0, 1, 2], []))
        self.assertEqual([e for e in range(1, 9) if should_audit(e, True)], [1, 2, 4, 6, 8])
        self.assertEqual([e for e in range(1, 9) if should_audit(e, False)], [1, 2])
        def item(name, gain, tail=0, stress=0, params=1):
            conditions = {c: {'outer_metrics': {'macro_accuracy': .65 + (gain if c == 'native' else stress),
                          'tail_accuracy': .60 + tail, 'macro_nll': 1.7}} for c in CONDITIONS}
            return {'root': name, 'summary': {'dataset_signature': 'data', 'conditions': conditions},
                    'checkpoint': {'model': {'p': torch.zeros(params)}, 'config': {'recipe': name}}}
        baseline = item('v12', 0)
        a, b, c = item('strength', .006), item('dynamic', .01, tail=-.006), item('expanded', .02, stress=-.006)
        self.assertIs(choose_candidate(baseline, [a, b, c])[0], a)
        self.assertIs(choose_candidate(baseline, [item('weak', .004)])[0], baseline)
        c = item('expanded', .006, params=2)
        self.assertIs(choose_candidate(baseline, [a, c])[0], a)
        bad = deepcopy(a); bad['summary']['dataset_signature'] = 'other'
        with self.assertRaisesRegex(ValueError, 'different validation'):
            choose_candidate(baseline, [bad])

    def test_refit_selection_binds_schedule_source_and_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'model.pt'; source.write_bytes(b'weights')
            data = {'kind': 'v13_refit_selection', 'format_version': 13, 'recipe': 'dynamic',
                    'selected_epoch': 17, 'schedule_epochs': 24, 'dataset_signature': 'split',
                    'class_names': ['0000', '0001'], 'class_bias': [0., 0.],
                    'base_model_identity': {'weights': 'official'}, 'source_checkpoint': str(source),
                    'source_checkpoint_sha256': file_sha256(source)}
            path = root / 'selection.json'; path.write_text(json.dumps(data))
            self.assertEqual(load_selection(path)['selected_epoch'], 17)
            with self.assertRaisesRegex(ValueError, 'base_model_identity'):
                load_selection(path, base_identity={'weights': 'other'})
            data['schedule_epochs'] = 17; path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'schedule'):
                load_selection(path)
            data['schedule_epochs'] = 24; path.write_text(json.dumps(data)); source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                load_selection(path)

    def test_periodic_audit_and_exact_epoch_resume(self):
        cpu, n = torch.device('cpu'), 9
        labels = np.arange(n) % 3
        ones = np.ones(n, np.float32)
        proto = np.eye(3, 8, dtype=np.float32)
        signals = PrototypeSignals(proto, np.zeros(n), ones, ones, ones, labels.copy(), ones, ones)
        neighbor = NeighborEvidence(ones, labels.copy(), ones, ones, np.zeros(n))
        config = {**recipe_config('dynamic'), 'version': 'v13', 'stage': 'validate', 'soft_teacher': True,
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
            with patch('train_v13.build_optimizer', side_effect=lambda model, cfg: legacy_optimizer(model)):
                full = root / 'full'; full.mkdir()
                torch.manual_seed(123)
                result = fit(classifier=deepcopy(initial), output_dir=full, resume=None, epochs=6, **deepcopy(common))
                split = root / 'split'; split.mkdir()
                torch.manual_seed(123)
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
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_resume(str(split / 'resume_latest.pt'), {**config, 'stage': 'refit'}, 'tiny', cpu)

    def test_refit_main_rebuilds_pool_without_validation_and_freezes_calibration(self):
        import train_v13
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
            calibration = {'view_weights': {'center': .5, 'hflip': .5}, 'precision': 'fp32', 'alpha': .2}
            selection = {'kind': 'v13_refit_selection', 'format_version': 13, 'recipe': 'dynamic',
                'selected_epoch': 2, 'schedule_epochs': 24, 'dataset_signature': 'tiny',
                'class_names': manifest.class_names, 'base_model_identity': {'base': 'official'},
                'class_bias': [.1, 0., -.1], 'calibration': calibration,
                'source_checkpoint': str(source), 'source_checkpoint_sha256': file_sha256(source),
                'training_config': {'conflict_policy': 'partial', 'sampler': 'shuffle',
                                    'feature_cache': str(root / 'features.npy')}}
            selected = root / 'selection.json'; selected.write_text(json.dumps(selection))
            out = root / 'refit'
            argv = ['train_v13.py', '--stage', 'refit', '--selection-json', str(selected),
                    '--train-dir', str(data), '--data-manifest', str(root / 'manifest.json'),
                    '--model-dir', 'official', '--output-dir', str(out), '--device', 'cpu',
                    '--workers', '0', '--sampler', 'shuffle', '--eval-batch-size', '18']
            features = np.stack([np.eye(3, 16, dtype=np.float32)[labels]] * 4)
            real_neighbors = train_v13.load_or_create_neighbor_evidence
            with patch.object(sys, 'argv', argv), \
                 patch('train_v13.load_manifest_dataset', return_value=manifest), \
                 patch('train_v13.read_manifest', return_value={'rows': records}), \
                 patch('train_v13.model_identity', return_value={'base': 'official'}), \
                 patch('train_v13.load_clip', side_effect=lambda *a: (clip(), processor())), \
                 patch('train_v13.validate_clip_vit_b32'), \
                 patch('train_v13.load_or_create_multiview_features', return_value=features), \
                 patch('train_v13.load_or_create_neighbor_evidence', wraps=real_neighbors) as neighbors, \
                 patch('train_v13.evaluate_two_view_cv', side_effect=AssertionError('refit used validation')):
                train_v13.main()
            self.assertEqual(neighbors.call_args.args[5], list(range(18)))
            self.assertEqual(neighbors.call_args.args[6], [])
            checkpoint = torch.load(out / 'model.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['training_stage'], 'refit')
            self.assertEqual(checkpoint['validation']['indices'], [])
            self.assertEqual(checkpoint['config']['schedule_epochs'], 24)
            self.assertEqual(checkpoint['selected_epoch'], 2)
            self.assertEqual(checkpoint['calibration']['view_weights'], calibration['view_weights'])
            self.assertTrue(checkpoint['calibration']['frozen_before_refit'])
            np.testing.assert_allclose(checkpoint['class_bias'], selection['class_bias'])
            self.assertFalse((out / 'validation_epochs').exists())
            # Exercise the real CSV/ZIP entry point without using any training labels.
            import predict_v13
            predicted = root / 'prediction'
            argv = ['predict_v13.py', '--checkpoint', str(out / 'model.pt'), '--model-dir', 'official',
                    '--test-dir', str(data), '--output', str(predicted / 'pred_results.csv'),
                    '--zip-output', str(predicted / 'pred_results.zip'), '--expected-rows', '18',
                    '--device', 'cpu', '--workers', '0', '--batch-size', '18']
            with patch.object(sys, 'argv', argv), \
                 patch('predict_v13.build_classifier_from_checkpoint', return_value=(Tiny().eval(), resolution_processor(processor()))):
                predict_v13.main()
            predict_v13.validate_submission(predicted / 'pred_results.csv', predicted / 'pred_results.zip', 18,
                                            [f'{i:02d}.png' for i in range(18)])

    def test_v13_cache_identity_changes_with_checkpoint_precision_and_rows(self):
        from evaluate_tta_v13 import LogitCache
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'epoch_01.pt'; source.write_bytes(b'weights')
            config = {**recipe_config('strength'), 'image_size': 320, 'zoom_shortest_edge': 366,
                      'interpolate_pos_encoding': True}
            checkpoint = {'format_version': 13, 'config': config}
            manifest = SimpleNamespace(signature='data', class_names=['0000', '0001'])
            cache = LogitCache(output_dir=root, paths={1: source}, checkpoints={1: checkpoint},
                manifest=manifest, indices=[1, 3], labels=np.array([0, 1]), model_dir='official',
                device=torch.device('cpu'), batch_size=2, workers=0, force=False)
            identity = cache._identity(1, 'native', False)
            with patch('evaluate_tta_v13.precision_name', return_value='bf16_autocast_fp32_stats'):
                self.assertNotEqual(cache._identity(1, 'native', False), identity)
            cache.hashes[1] = 'new-weights'
            self.assertNotEqual(cache._identity(1, 'native', False), identity)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
