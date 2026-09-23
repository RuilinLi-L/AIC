import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from train_v5 import clone_trainable
from train_v7 import (
    RESUME_VERSION,
    deterministic_loader,
    load_resume,
    partial_label_loss,
    partial_label_targets,
    run_epoch,
)
from v6_core import SelectiveFeatureQueue


class IndexedDataset(Dataset):
    def __init__(self, count):
        self.indices = list(range(count))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.indices[item]


class TinyPairedDataset(Dataset):
    def __init__(self):
        self.indices = list(range(4))
        self.labels = torch.tensor([0, 1, 0, 0], dtype=torch.long)
        self.features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.6, 0.8]])

    def __len__(self):
        return 4

    def __getitem__(self, item):
        feature = F.normalize(self.features[item], dim=0)
        return {
            "pixel_values_a": feature,
            "pixel_values_b": F.normalize(feature + torch.tensor([0.01, -0.01]), dim=0),
            "label": self.labels[item],
            "row_index": torch.tensor(item, dtype=torch.long),
        }


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Parameter(torch.tensor([[1.0, -0.1], [-0.1, 1.0]]))
        self.adapter_delta = nn.Parameter(torch.tensor([[0.01, 0.0], [0.0, -0.01]]))

    def forward(self, pixels, _):
        base = F.normalize(pixels.float(), dim=1)
        adapted = F.normalize(base + base @ self.adapter_delta, dim=1)
        logits = 5.0 * adapted @ F.normalize(self.classifier, dim=1).t()
        return logits, None, base, adapted


class TrainV7Tests(unittest.TestCase):
    def test_training_epoch_accepts_clean_and_partial_rows(self):
        dataset = TinyPairedDataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        classifier = TinyClassifier()
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        ema = clone_trainable(classifier)
        queue = SelectiveFeatureQueue(2, capacity=2, dim=2, device=torch.device("cpu"))
        labels = dataset.labels.numpy()
        ambiguous = np.asarray([False, False, False, True])
        temporal = np.zeros((4, 2), dtype=np.float16)
        stats, steps = run_epoch(
            classifier,
            loader,
            optimizer,
            scheduler,
            ema,
            queue,
            torch.device("cpu"),
            1,
            labels,
            np.asarray([0.8, 0.8, 0.8, 0.25], dtype=np.float32),
            np.asarray([True, True, True, False]),
            torch.tensor([0.5, 0.5]),
            F.normalize(dataset.features, dim=1).numpy(),
            torch.eye(2),
            np.zeros(4, dtype=np.bool_),
            temporal,
            np.zeros(4, dtype=np.bool_),
            np.full(4, -1, dtype=np.int64),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.int16),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.int16),
            0,
            ambiguous,
            [(0,), (1,), (0,), (0, 1)],
            1,
        )
        self.assertEqual(steps, 2)
        self.assertEqual(stats["ambiguous_rows_seen"], 1)
        self.assertTrue(np.isfinite(stats["loss"]))

    def test_partial_targets_are_supported_capped_and_normalized(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        targets = partial_label_targets(features, prototypes, [(0, 2), (0, 1)])
        self.assertTrue(torch.allclose(targets.sum(dim=1), torch.ones(2)))
        self.assertLessEqual(float(targets.max()), 0.800001)
        self.assertEqual(float(targets[0, 1]), 0.0)
        logits = torch.randn(2, 3, requires_grad=True)
        loss = partial_label_loss(logits, logits + 0.1, targets).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)

    def test_repeat_sampler_replays_same_epoch(self):
        dataset = IndexedDataset(12)
        factors = np.asarray([4.0] + [1.0] * 11)
        first = list(
            deterministic_loader(
                dataset, 3, 4, 0, torch.device("cpu"), "repeat-factor", factors
            )
        )
        second = list(
            deterministic_loader(
                dataset, 3, 4, 0, torch.device("cpu"), "repeat-factor", factors
            )
        )
        self.assertEqual(torch.cat(first).tolist(), torch.cat(second).tolist())
        self.assertEqual(len(torch.cat(first)), len(dataset))

    def test_resume_rejects_configuration_or_dataset_change(self):
        payload = {
            "kind": "train_v7_resume",
            "resume_version": RESUME_VERSION,
            "stage": "validation",
            "config": {"run": "x"},
            "dataset_signature": "abc",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            torch.save(payload, path)
            loaded = load_resume(str(path), {"run": "x"}, "abc", torch.device("cpu"))
            self.assertEqual(loaded["stage"], "validation")
            with self.assertRaises(ValueError):
                load_resume(str(path), {"run": "y"}, "abc", torch.device("cpu"))
            with self.assertRaises(ValueError):
                load_resume(str(path), {"run": "x"}, "different", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
