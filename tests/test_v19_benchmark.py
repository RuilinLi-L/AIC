"""V19 real-training probe isolation and strict resource-error classification."""

from contextlib import ExitStack
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from benchmark_v19 import DeterministicImages, benchmark, cuda_oom_report, validate_probe_args
from robust_clip import trainable_state_dict


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = torch.nn.Linear(3, 8)
        self.adapter = torch.nn.Linear(8, 8)
        self.classifier = torch.nn.Parameter(torch.eye(3, 8))
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, pixels, unused=None):
        base = F.normalize(self.clip(pixels.mean(dim=(2, 3))).float(), dim=1)
        adapted = F.normalize(self.adapter(base).float(), dim=1)
        return 10 * adapted @ F.normalize(self.classifier, dim=1).T, None, base, adapted


class TinyPairs(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {'pixel_values_a': torch.randn(3, 4, 4),
                'pixel_values_b': torch.randn(3, 4, 4) * random.random() + np.random.random(),
                'label': torch.tensor(self.rows[index][1]), 'row_index': torch.tensor(index)}


class BenchmarkTests(unittest.TestCase):
    def execute_probe(self, root, fail_256=False, recipe="resolution384_rank32"):
        from train_v19 import run_epoch

        network = TinyClassifier()
        initial = trainable_state_dict(network)
        observed_loaders, observed_epochs = [], []
        manifest = SimpleNamespace(
            rows=[('train0', 0, '0000'), ('train1', 1, '0001'), ('train2', 2, '0002'),
                  ('VALIDATION', 0, '0000')],
            train_indices=[0, 1, 2], validation_indices=[3], clean_mask=np.array([False, True, True, True]),
            ambiguous_mask=np.array([True, False, False, False]),
            candidate_labels=[(0, 1), (1,), (2,), (0,)],
            class_names=['0000', '0001', '0002'], signature='unit-test',
        )
        args = SimpleNamespace(steps=1, warmup_steps=1, output=str(root / 'benchmark.json'),
                               recipe=recipe, batch_size=256, gradient_accumulation=1,
                               base_model_identity={'test': 'official'})

        def loader(dataset, **kwargs):
            observed_loaders.append(kwargs.copy())
            kwargs.pop('prefetch_factor')
            kwargs['num_workers'] = 0
            return DataLoader(dataset, **kwargs)

        def dataset(rows, indices, processor, config):
            self.assertNotIn('VALIDATION', [row[0] for row in rows])
            self.assertEqual(config['recipe'], recipe)
            return TinyPairs(rows)

        def epoch(**kwargs):
            if kwargs['optimizer_step'] == 0:
                for key, value in trainable_state_dict(network).items():
                    torch.testing.assert_close(value, initial[key])
                self.assertTrue(kwargs['queue'].valid.all())
                self.assertTrue(kwargs['dynamic_noise'])
                self.assertTrue(kwargs['repair_plan'].any())
            observed_epochs.append(kwargs['gradient_accumulation'])
            if fail_256 == 'data':
                raise ValueError('bad training image')
            if fail_256 and kwargs['gradient_accumulation'] == 1:
                raise torch.OutOfMemoryError('test OOM')
            # Run the actual V19 projection, repair, partial-label and EMA code.
            return run_epoch(**kwargs)

        def optimizer(classifier, config):
            return torch.optim.AdamW(classifier.parameters(), lr=.0001), [], []

        with ExitStack() as stack:
            stack.enter_context(patch('benchmark_v19.DataLoader', side_effect=loader))
            stack.enter_context(patch('benchmark_v19.training_dataset', side_effect=dataset))
            stack.enter_context(patch('benchmark_v19.build_optimizer', side_effect=optimizer))
            stack.enter_context(patch('train_v19.run_epoch', side_effect=epoch))
            report = benchmark(args, network, None, manifest,
                               np.ones((4, 4, 8), np.float32), np.eye(3, 8), torch.device('cpu'))
        for key, value in trainable_state_dict(network).items():
            torch.testing.assert_close(value, initial[key])
        self.assertTrue(network.training)
        self.assertEqual(list(root.iterdir()), [root / 'benchmark.json'])
        return report, observed_loaders, observed_epochs

    def test_real_training_probe_resets_state_and_does_not_write_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            report, loaders, epochs = self.execute_probe(Path(directory))
        self.assertTrue(report['success'])
        self.assertEqual(report['chosen_workers'], 8)
        self.assertEqual(report['batch_size'], 256)
        self.assertEqual([row['num_workers'] for row in loaders], [8])
        self.assertTrue(all(row['prefetch_factor'] == 1 and row['pin_memory'] is False for row in loaders))
        self.assertEqual(epochs, [1, 1])
        for row in report['configurations']:
            self.assertGreater(row['stats']['repaired_rows'], 0)
            self.assertGreater(row['stats']['ambiguous_rows_seen'], 0)
            self.assertGreater(row['stats']['soft_teacher_rows'], 0)
            self.assertGreater(row['images_per_s'], 0)

    def test_cpu_oom_is_not_mislabeled_as_cuda_resource_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(torch.OutOfMemoryError):
                self.execute_probe(root, fail_256=True)
            self.assertFalse((root / 'benchmark.json').exists())

    def test_typed_cuda_oom_returns_structured_failure_for_both_recipes(self):
        for recipe in ('resolution384_rank32', 'rank32_dropout'):
            with tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(recipe=recipe, output=str(Path(directory) / 'benchmark.json'),
                                       base_model_identity={'base': 'hash'}, steps=8, warmup_steps=2)
                result = cuda_oom_report(args, torch.device('cuda'), torch.OutOfMemoryError('CUDA OOM'),
                                         dataset_signature='dataset', phase='model_or_probe_setup')
                self.assertFalse(result['success'])
                self.assertEqual(result['status'], 'cuda_oom')
                self.assertEqual((result['batch_size'], result['gradient_accumulation']), (256, 1))
                self.assertEqual([x['batch_size'] for x in result['configurations']], [256])
                self.assertEqual(result['lora_dropout'], .05 if recipe == 'rank32_dropout' else 0.)
                self.assertEqual(result['lora_dropout'], result['mlp_lora_dropout'])
                self.assertEqual(json.loads(Path(args.output).read_text()), result)
                with self.assertRaisesRegex(RuntimeError, 'other error'):
                    cuda_oom_report(args, torch.device('cuda'), RuntimeError('other error'),
                                    dataset_signature='dataset', phase='probe')

    def test_per_item_augmentations_are_independent_of_worker_rng(self):
        dataset = DeterministicImages(TinyPairs([('unused', 0, '0000')]))
        first = dataset[0]
        random.seed(99)
        np.random.seed(77)
        torch.manual_seed(22)
        second = dataset[0]
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_data_error_propagates_without_resource_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'bad training image'):
                self.execute_probe(root, fail_256='data')
            self.assertFalse((root / 'benchmark.json').exists())

    def test_smaller_batch_or_legacy_recipe_is_rejected_before_probe(self):
        for recipe, batch, accumulation in [('resolution384_rank32', 128, 2),
                                             ('rank32_dropout', 128, 2),
                                             ('matched_control', 256, 1),
                                             ('rank32', 256, 1)]:
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                validate_probe_args(SimpleNamespace(recipe=recipe, batch_size=batch,
                                                     gradient_accumulation=accumulation))

    def test_main_reports_cuda_setup_oom_without_loading_learned_weights(self):
        import benchmark_v19
        identity = {'base': 'sha'}
        manifest = SimpleNamespace(rows=[('train', 0, '0000'), ('validation', 0, '0000')],
            train_indices=[0], validation_indices=[1], clean_mask=np.ones(2, bool),
            class_names=['0000'], signature='dataset')
        checkpoint = {'format_version': 18, 'class_names': ['0000'], 'dataset_signature': 'dataset',
                      'config': {'recipe': 'rank32', 'conflict_policy': 'partial',
                                 'base_model_identity': identity}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'benchmark.json'
            argv = ['benchmark_v19.py', '--model-dir', 'base', '--train-dir', 'train',
                    '--data-manifest', 'manifest.json', '--checkpoint', 'metadata.pt',
                    '--feature-cache', 'features.npy', '--output', str(output),
                    '--recipe', 'rank32_dropout', '--device', 'cuda']
            with patch.object(sys, 'argv', argv), \
                 patch('benchmark_v19.resolve_device', return_value=torch.device('cuda')), \
                 patch('benchmark_v19.configure_determinism'), \
                 patch('benchmark_v19.torch.load', return_value=checkpoint), \
                 patch('benchmark_v19.load_manifest_dataset', return_value=manifest), \
                 patch('benchmark_v19.load_frozen_features', return_value=np.ones((4, 2, 8), np.float32)), \
                 patch('benchmark_v19.model_identity', return_value=identity), \
                 patch('benchmark_v19.load_clip', side_effect=torch.OutOfMemoryError('CUDA allocation')), \
                 patch('benchmark_v19.build_classifier') as build:
                result = benchmark_v19.main()
            build.assert_not_called()
            self.assertEqual(result['status'], 'cuda_oom')
            self.assertFalse(result['success'])
            self.assertEqual(json.loads(output.read_text())['configurations'][0]['phase'],
                             'model_or_probe_setup')


if __name__ == '__main__':
    unittest.main()
