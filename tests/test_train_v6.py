import copy
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_v5 import clone_trainable, update_ema
from train_v6 import make_v6_scheduler, run_epoch
from v6_core import SelectiveFeatureQueue


class TinyPairedDataset(Dataset):
    def __init__(self):
        self.labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        self.features = torch.eye(2)[self.labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        feature = self.features[item]
        return {
            "pixel_values_a": feature,
            "pixel_values_b": F.normalize(feature + torch.tensor([0.02, 0.01]), dim=0),
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


def make_objects():
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = make_v6_scheduler(optimizer, total_steps=3)
    ema = clone_trainable(model)
    queue = SelectiveFeatureQueue(2, capacity=2, dim=2, device=torch.device("cpu"))
    return model, optimizer, scheduler, ema, queue


def make_arrays():
    return {
        "temporal": np.zeros((4, 2), dtype=np.float16),
        "seen": np.zeros(4, dtype=np.bool_),
        "previous": np.full(4, -1, dtype=np.int64),
        "stability_sum": np.zeros(4, dtype=np.float32),
        "stability_count": np.zeros(4, dtype=np.int16),
        "loss_sum": np.zeros(4, dtype=np.float32),
        "loss_count": np.zeros(4, dtype=np.int16),
    }


def run_tiny_epoch(model, optimizer, scheduler, ema, queue, arrays, epoch, step):
    dataset = TinyPairedDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    labels = dataset.labels.numpy()
    _, next_step = run_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        ema,
        queue,
        torch.device("cpu"),
        epoch,
        labels,
        np.full(4, 0.8, dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        torch.tensor([0.5, 0.5]),
        dataset.features.numpy(),
        torch.eye(2),
        np.zeros(4, dtype=np.bool_),
        arrays["temporal"],
        arrays["seen"],
        arrays["previous"],
        arrays["stability_sum"],
        arrays["stability_count"],
        arrays["loss_sum"],
        arrays["loss_count"],
        step,
    )
    return next_step


class TrainV6Tests(unittest.TestCase):
    def test_bias_corrected_ema_does_not_use_raw_decay_at_first_step(self):
        model = TinyClassifier()
        shadow = clone_trainable(model)
        before = shadow["classifier"].clone()
        with torch.no_grad():
            model.classifier.add_(1.0)
        update_ema(shadow, model, decay=0.999, step=1)
        expected_decay = 2.0 / 11.0
        expected = before * expected_decay + model.classifier.detach() * (1.0 - expected_decay)
        self.assertTrue(torch.equal(shadow["classifier"], expected))

    def test_epoch_boundary_resume_matches_uninterrupted_parameters(self):
        baseline = make_objects()
        baseline_arrays = make_arrays()
        baseline_step = 0
        for epoch in (1, 2, 3):
            baseline_step = run_tiny_epoch(*baseline, baseline_arrays, epoch, baseline_step)

        interrupted = make_objects()
        interrupted_arrays = make_arrays()
        interrupted_step = 0
        for epoch in (1, 2):
            interrupted_step = run_tiny_epoch(
                *interrupted, interrupted_arrays, epoch, interrupted_step
            )
        model, optimizer, scheduler, ema, queue = interrupted
        saved = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "ema": {name: value.clone() for name, value in ema.items()},
            "queue": copy.deepcopy(queue.state_dict()),
            "arrays": {name: value.copy() for name, value in interrupted_arrays.items()},
            "step": interrupted_step,
        }

        resumed = make_objects()
        resumed_model, resumed_optimizer, resumed_scheduler, resumed_ema, resumed_queue = resumed
        resumed_model.load_state_dict(saved["model"])
        resumed_optimizer.load_state_dict(saved["optimizer"])
        resumed_scheduler.load_state_dict(saved["scheduler"])
        resumed_ema = {name: value.clone() for name, value in saved["ema"].items()}
        resumed_queue.load_state_dict(saved["queue"])
        resumed = (
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            resumed_ema,
            resumed_queue,
        )
        resumed_arrays = saved["arrays"]
        resumed_step = run_tiny_epoch(*resumed, resumed_arrays, 3, saved["step"])

        baseline_model, _, _, baseline_ema, baseline_queue = baseline
        self.assertEqual(baseline_step, resumed_step)
        for name, expected in baseline_model.state_dict().items():
            self.assertTrue(torch.equal(expected, resumed_model.state_dict()[name]), name)
        for name, expected in baseline_ema.items():
            self.assertTrue(torch.equal(expected, resumed_ema[name]), name)
        for name, expected in baseline_queue.state_dict().items():
            self.assertTrue(torch.equal(expected, resumed_queue.state_dict()[name]), name)
        for name, expected in baseline_arrays.items():
            np.testing.assert_array_equal(expected, resumed_arrays[name])


if __name__ == "__main__":
    unittest.main()
