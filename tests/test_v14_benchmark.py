"""Benchmark isolation, real late-epoch losses and OOM fallback behavior."""

from contextlib import ExitStack
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from benchmark_v14 import DeterministicImages, benchmark
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
    def execute_probe(self, root, fail_256=False):
        from train_v14 import run_epoch

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
        args = SimpleNamespace(steps=1, warmup_steps=1, output=str(root / 'benchmark.json'))

        def loader(dataset, **kwargs):
            observed_loaders.append(kwargs.copy())
            kwargs.pop('prefetch_factor')
            kwargs['num_workers'] = 0
            return DataLoader(dataset, **kwargs)

        def dataset(rows, indices, processor, config):
            self.assertNotIn('VALIDATION', [row[0] for row in rows])
            self.assertEqual(config['recipe'], 'expanded_robust')
            return TinyPairs(rows)

        def epoch(**kwargs):
            if kwargs['optimizer_step'] == 0:
                for key, value in trainable_state_dict(network).items():
                    torch.testing.assert_close(value, initial[key])
                self.assertTrue(kwargs['queue'].valid.all())
                self.assertTrue(kwargs['dynamic_noise'])
                self.assertTrue(kwargs['repair_plan'].any())
            observed_epochs.append(kwargs['gradient_accumulation'])
            if fail_256 and kwargs['gradient_accumulation'] == 1:
                raise torch.OutOfMemoryError('test OOM')
            # Run the actual V14 projection, repair, partial-label and EMA code.
            return run_epoch(**kwargs)

        def optimizer(classifier, config):
            return torch.optim.AdamW(classifier.parameters(), lr=.0001), [], []

        with ExitStack() as stack:
            stack.enter_context(patch('benchmark_v14.DataLoader', side_effect=loader))
            stack.enter_context(patch('benchmark_v14.training_dataset', side_effect=dataset))
            stack.enter_context(patch('benchmark_v14.build_optimizer', side_effect=optimizer))
            stack.enter_context(patch('train_v14.run_epoch', side_effect=epoch))
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
        self.assertIn(report['chosen_workers'], (4, 8))
        self.assertEqual(report['batch_size'], 256)
        self.assertEqual([row['num_workers'] for row in loaders], [4, 8])
        self.assertTrue(all(row['prefetch_factor'] == 1 and row['pin_memory'] is False for row in loaders))
        self.assertEqual(epochs, [1, 1, 1, 1])
        for row in report['configurations']:
            self.assertGreater(row['stats']['repaired_rows'], 0)
            self.assertGreater(row['stats']['ambiguous_rows_seen'], 0)
            self.assertGreater(row['stats']['soft_teacher_rows'], 0)
            self.assertGreater(row['images_per_s'], 0)

    def test_oom_is_the_only_trigger_for_smaller_microbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            report, loaders, epochs = self.execute_probe(Path(directory), fail_256=True)
        self.assertTrue(report['success'])
        self.assertEqual((report['batch_size'], report['gradient_accumulation']), (128, 2))
        self.assertEqual([row['status'] for row in report['configurations']],
                         ['out_of_memory', 'out_of_memory', 'ok', 'ok'])
        self.assertEqual([row['batch_size'] for row in loaders], [256, 256, 128, 128])
        self.assertEqual(epochs, [1, 1, 2, 2, 2, 2])

    def test_per_item_augmentations_are_independent_of_worker_rng(self):
        dataset = DeterministicImages(TinyPairs([('unused', 0, '0000')]))
        first = dataset[0]
        random.seed(99)
        np.random.seed(77)
        torch.manual_seed(22)
        second = dataset[0]
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
