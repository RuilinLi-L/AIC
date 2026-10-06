"""Behavioral V12 checks: batched queue, late losses, epoch loop, cache and delivery."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from benchmark_v12 import isolated_state
from evaluate_tta_v12 import LogitCache
from predict_v12 import validate_submission
from robust_clip import FolderImageDataset
from train_v5 import clone_trainable
from train_v6 import build_optimizer, make_v6_scheduler
from train_v12 import fit, run_epoch, save_epoch_logits
from v6_core import SelectiveFeatureQueue, PrototypeSignals
from v8_neighbors import NeighborEvidence
from v12_model import LORA_CONFIG
from v12_runtime import BatchedFeatureQueue, loader_options
from test_v11 import model, processor
from v12_model import build_classifier, resolution_processor


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = torch.nn.Linear(3, 8)
        self.adapter = torch.nn.Linear(8, 8)
        self.classifier = torch.nn.Parameter(torch.eye(3, 8))
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, pixels, unused=None):
        base = F.normalize(self.clip(pixels.mean(dim=(2, 3))).float(), dim=1)
        adapted = F.normalize(self.adapter(base).float(), dim=1)
        return 10 * adapted @ F.normalize(self.classifier, dim=1).T, None, base, adapted


class RuntimeTests(unittest.TestCase):
    def test_queue_matches_sequential_including_wrap_and_empty(self):
        torch.manual_seed(2)
        old = SelectiveFeatureQueue(5, 3, 8, torch.device('cpu'))
        new = BatchedFeatureQueue(5, 3, 8, torch.device('cpu'))
        for count in (0, 2, 100, 17, 1):
            values = torch.randn(count, 8)
            labels = torch.randint(0, 5, (count,))
            mask = torch.rand(count) > .2
            old.enqueue(values, labels, mask)
            new.enqueue(values, labels, mask)
            for key in old.state_dict():
                torch.testing.assert_close(old.state_dict()[key], new.state_dict()[key], rtol=0, atol=0)
        restored = BatchedFeatureQueue(5, 3, 8, torch.device('cpu'))
        restored.load_state_dict(new.state_dict())
        torch.testing.assert_close(restored.pointer, old.pointer)

    def test_loader_zero_workers_omits_prefetch(self):
        self.assertEqual(loader_options(0, 2), {})
        self.assertEqual(loader_options(16, 2), {'prefetch_factor': 2})
        with self.assertRaises(ValueError):
            loader_options(4, 0)

    def test_cpu_bf16_lora_gradients_are_finite(self):
        classifier = build_classifier(model(), np.eye(3, 16, dtype=np.float32), torch.device('cpu'))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = classifier(torch.randn(2, 3, 320, 320))[0]
            loss = F.cross_entropy(logits.float(), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        grads = [p.grad for n, p in classifier.named_parameters() if n.endswith('lora_b')]
        self.assertEqual(len(grads), 16)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

    def test_late_epoch_matches_original_math_on_cpu(self):
        import train_v11
        n = 18
        torch.manual_seed(7)
        initial = Tiny()
        labels = np.arange(n) % 3
        ambiguous = np.arange(n) == 4
        candidates = [(0, 1) if v else (int(labels[i]),) for i, v in enumerate(ambiguous)]
        centers = np.random.default_rng(6).normal(size=(n, 8)).astype(np.float32)
        prototypes = torch.eye(3, 8)
        batch = {'pixel_values_a': torch.randn(n, 3, 8, 8), 'pixel_values_b': torch.randn(n, 3, 8, 8),
                 'label': torch.tensor(labels), 'row_index': torch.arange(n)}
        outputs = []
        for epoch_fn, queue_type in ((train_v11.run_epoch, SelectiveFeatureQueue), (run_epoch, BatchedFeatureQueue)):
            network = deepcopy(initial)
            optimizer, _, _ = build_optimizer(network)
            scheduler = make_v6_scheduler(optimizer, 10)
            state = isolated_state(labels, ambiguous, 3)
            queue = queue_type(3, 2, 8, torch.device('cpu'))
            queue.enqueue(prototypes, torch.arange(3), torch.ones(3, dtype=torch.bool))
            stats, step = epoch_fn(network, [batch], optimizer, scheduler, clone_trainable(network),
                queue, torch.device('cpu'), 5, labels, state['quality'], state['trusted'], torch.ones(3) / 3,
                centers, prototypes, state['repair_plan'], state['temporal'], state['temporal_seen'],
                state['previous_view_prediction'], state['stability_sum'], state['stability_count'],
                state['loss_sum'], state['loss_count'], 0, ambiguous, candidates, 1, True)
            self.assertEqual(step, 1)
            self.assertGreater(stats['repaired_rows'], 0)
            self.assertGreater(stats['soft_teacher_rows'], 0)
            outputs.append((network, state, stats))
        for a, b in zip(outputs[0][0].parameters(), outputs[1][0].parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key in outputs[0][1]:
            np.testing.assert_array_equal(outputs[0][1][key], outputs[1][1][key])
        self.assertEqual(outputs[0][2], outputs[1][2])

    def test_three_epoch_loop_and_completed_resume(self):
        cpu = torch.device('cpu')
        n = 9
        labels = np.arange(n) % 3
        ambiguous = np.arange(n) == 5
        ones = np.ones(n, np.float32)
        proto = np.eye(3, 8, dtype=np.float32)
        signals = PrototypeSignals(proto, np.zeros(n), ones, ones, ones, labels.copy(), ones, ones)
        neighbors = NeighborEvidence(ones, labels.copy(), ones, ones, np.zeros(n))
        config = {'version': 'v12', 'soft_teacher': True, 'precision': 'fp32', 'workers': 0,
                  'batch_size': 256, 'gradient_accumulation': 1, 'prefetch_factor': 2, 'eval_batch_size': 3,
                  'image_size': 320, 'zoom_shortest_edge': 366, 'interpolate_pos_encoding': True, **LORA_CONFIG}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(n):
                path = root / f'{i}.png'
                Image.new('RGB', (330, 350), (i * 20, 90, 30)).save(path)
                rows.append((str(path), int(labels[i]), f'{labels[i]:04d}'))
            prep = resolution_processor(processor())
            network = Tiny()
            common = dict(classifier=network, processor=prep, rows=rows, train_indices=list(range(6)),
                supervised_indices=list(range(5)), labels=labels,
                view_features=np.stack([proto[labels]] * 4), signals=signals, quality=ones.copy() * .8,
                validation_loader=DataLoader(FolderImageDataset(rows, [6, 7, 8], prep), batch_size=3),
                trusted_validation=np.ones(n, dtype=bool), class_counts=np.array([2, 2, 1]),
                class_names=['0000', '0001', '0002'], config=config, dataset_digest='tiny', output_dir=root,
                device=cpu, batch_size=256, workers=0, epochs=3, stage='validation', validation_indices=[6, 7, 8],
                ambiguous_mask=ambiguous, candidate_labels=[(0, 2) if i == 5 else (int(labels[i]),) for i in range(n)],
                sampler_mode='shuffle', row_repeat_factors=np.ones(n), gradient_accumulation=1, neighbor_evidence=neighbors)
            result = fit(resume=None, **common)
            self.assertEqual(len(result[-1]), 3)
            self.assertTrue(all('audit_seconds' in h and 'validation_seconds' in h for h in result[-1]))
            from train_v12 import load_resume
            resume = load_resume(str(root / 'resume_latest.pt'), config, 'tiny', cpu)
            resumed = fit(resume=resume, **common)
            self.assertEqual(resumed[1], result[1])
            self.assertEqual(resumed[-1], result[-1])

    def test_native_cache_rejects_other_precision_and_checkpoint(self):
        cpu = torch.device('cpu')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'epoch_01.pt'
            checkpoint.write_bytes(b'checkpoint')
            values = np.ones((2, 3), np.float32)
            config = dict(image_size=320, zoom_shortest_edge=366, interpolate_pos_encoding=True, **LORA_CONFIG)
            manifest = SimpleNamespace(signature='data', class_names=['a', 'b', 'c'], rows=[])
            kwargs = dict(output_dir=root, paths={1: checkpoint}, checkpoints={1: {'format_version': 12, 'config': config}},
                          manifest=manifest, indices=[0, 1], labels=np.array([0, 1]), model_dir='unused', device=cpu,
                          batch_size=2, workers=0, force=False)
            save_epoch_logits(root, 1, values, values, np.array([0, 1]), [0, 1], 'fp32')
            cache = LogitCache(**kwargs)
            np.testing.assert_array_equal(cache.get_views(1, 'native', False)['center'], values)
            save_epoch_logits(root, 1, values, values, np.array([0, 1]), [0, 1], 'bf16_autocast_fp32_stats')
            with patch('evaluate_tta_v12.build_classifier_from_checkpoint', side_effect=RuntimeError('recompute')):
                with self.assertRaisesRegex(RuntimeError, 'recompute'):
                    LogitCache(**kwargs).get_views(1, 'native', False)
            save_epoch_logits(root, 1, values, values, np.array([0, 1]), [0, 1], 'fp32')
            checkpoint.write_bytes(b'changed')
            with patch('evaluate_tta_v12.build_classifier_from_checkpoint', side_effect=RuntimeError('recompute')):
                with self.assertRaisesRegex(RuntimeError, 'recompute'):
                    LogitCache(**kwargs).get_views(1, 'native', False)

    def test_submission_exact_row_count_and_contents(self):
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / 'pred_results.csv'
            archive = root / 'pred_results.zip'
            names = [f'{i}.jpg' for i in range(37444)]
            csv.write_text(''.join(f'{name}, 0001\n' for name in names))
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.write(csv, 'pred_results.csv')
            validate_submission(csv, archive, 37444, names)
            with self.assertRaisesRegex(ValueError, 'rows'):
                validate_submission(csv, archive, 37443, names)

    def test_benchmark_resets_state_and_does_not_create_training_outputs(self):
        from benchmark_v12 import benchmark
        network = Tiny()
        before = clone_trainable(network)
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output_dir=str(root / 'formal'), benchmark_output=str(root / 'bench.json'),
                                   benchmark_steps=1, benchmark_warmup_steps=1, workers=0, prefetch_factor=2)
            manifest = SimpleNamespace(train_indices=[0, 1, 2], rows=[('unused', i, f'{i:04d}') for i in range(3)],
                                       ambiguous_mask=np.zeros(3, bool), candidate_labels=[(i,) for i in range(3)],
                                       class_names=['0000', '0001', '0002'])
            def fake_epoch(**kw):
                if kw['optimizer_step'] == 0:
                    for key, value in clone_trainable(kw['classifier']).items():
                        torch.testing.assert_close(value, before[key])
                    self.assertTrue(kw['queue'].valid.all())
                    self.assertTrue(kw['repair_plan'].any())
                with torch.no_grad():
                    kw['classifier'].classifier.add_(1)
                calls.append(kw['gradient_accumulation'])
                return {'loss': 1.0}, kw['optimizer_step'] + len(kw['loader']) // kw['gradient_accumulation']
            with patch('train_v12.run_epoch', side_effect=fake_epoch):
                benchmark(args, network, resolution_processor(processor()), manifest,
                          np.zeros((4, 3, 8), np.float32), SimpleNamespace(prototypes=np.eye(3, 8, dtype=np.float32)),
                          torch.device('cpu'))
            self.assertEqual(calls, [4, 4, 2, 2, 1, 1])
            self.assertFalse((root / 'formal').exists())
            self.assertEqual(len(json.loads((root / 'bench.json').read_text())['configurations']), 3)
            for key, value in clone_trainable(network).items():
                torch.testing.assert_close(value, before[key])

    def test_prediction_entrypoint_creates_valid_zip(self):
        from predict_v12 import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            for i in range(2):
                Image.new('RGB', (330, 340), (100, i * 90, 20)).save(images / f'{i}.jpg')
            checkpoint = root / 'model.pt'
            torch.save({'format_version': 12, 'class_names': ['0000', '0001', '0002'],
                        'class_bias': torch.zeros(3),
                        'calibration': {'precision': 'fp32', 'view_weights': {'center': .5, 'hflip': .5}}}, checkpoint)
            argv = ['predict_v12.py', '--checkpoint', str(checkpoint), '--test-dir', str(images),
                    '--output', str(root / 'pred_results.csv'), '--zip-output', str(root / 'pred_results.zip'),
                    '--device', 'cpu', '--workers', '0', '--batch-size', '2', '--expected-rows', '2']
            with patch('sys.argv', argv), patch('predict_v12.build_classifier_from_checkpoint',
                    return_value=(Tiny(), resolution_processor(processor()))):
                main()
            validate_submission(root / 'pred_results.csv', root / 'pred_results.zip', 2, ['0.jpg', '1.jpg'])
