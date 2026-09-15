from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from robust_clip import TrainConfig
from train_v5 import (
    ResumableRandomSampler,
    ResumeTimer,
    atomic_torch_save,
    capture_rng_state,
    load_resume_checkpoint,
    make_scheduler,
    run_fit_epoch,
    save_resume_checkpoint,
    timestamped_resume_path,
)


class TinyDataset(Dataset):
    def __init__(self, labels: np.ndarray):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = int(self.labels[item])
        base = torch.tensor([1.0, 0.0] if label == 0 else [0.0, 1.0])
        return {
            "pixel_values_a": base + 0.05 * torch.rand(2),
            "pixel_values_b": base + 0.05 * torch.rand(2),
            "label": torch.tensor(label, dtype=torch.long),
            "row_index": torch.tensor(item, dtype=torch.long),
        }


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Parameter(torch.tensor([[1.0, -0.2], [-0.1, 0.9]]))

    def forward(self, pixels, _):
        base = F.normalize(pixels.float(), dim=1)
        logits = base @ self.classifier.t()
        return logits, None, base, None


def clone_state(module):
    return {name: value.detach().clone() for name, value in module.named_parameters() if value.requires_grad}


def tiny_args():
    return SimpleNamespace(
        prototype_temperature=0.07,
        label_smoothing=0.0,
        prototype_pseudo_start_epoch=2,
        temporal_decay=0.85,
        prototype_teacher_mix=0.5,
        pseudo_threshold=0.62,
        prototype_pseudo_weight=0.35,
        consistency_temperature=2.0,
        consistency_weight=0.05,
        anchor_weight=0.15,
        gradient_accumulation=2,
        ema_decay=0.997,
    )


class StopAfterCheckpoint(RuntimeError):
    pass


class TrainV5ResumeTests(unittest.TestCase):
    def test_resume_timer_uses_ninety_minute_boundary(self):
        values = iter([100.0, 5499.0, 5500.0, 5500.0, 10900.0])
        timer = ResumeTimer(interval_seconds=5400, clock=lambda: next(values))
        self.assertFalse(timer.due())
        self.assertTrue(timer.due())
        timer.mark_saved()
        self.assertTrue(timer.due())

    def test_timestamped_paths_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            now = datetime(2026, 9, 14, 19, 59, 45)
            first = timestamped_resume_path(output, now)
            self.assertEqual(first.name, "resume_20260914_195945.pt")
            first.touch()
            second = timestamped_resume_path(output, now)
            self.assertEqual(second.name, "resume_20260914_195945_01.pt")

    def test_atomic_save_leaves_only_complete_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "resume_20260914_200000.pt"
            atomic_torch_save({"value": torch.tensor([1, 2, 3])}, target)
            self.assertTrue(target.is_file())
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            loaded = torch.load(target, map_location="cpu")
            self.assertTrue(torch.equal(loaded["value"], torch.tensor([1, 2, 3])))

    def test_resume_loader_matches_uninterrupted_training(self):
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        features = np.eye(2, dtype=np.float32)[labels]
        prototypes = torch.eye(2)
        quality = np.ones(len(labels), dtype=np.float32)
        clean = np.ones(len(labels), dtype=np.bool_)
        margins = np.ones(len(labels), dtype=np.float32)
        class_weights = torch.ones(2)
        args = tiny_args()

        def make_training_objects():
            model = TinyClassifier()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = make_scheduler(optimizer, total_steps=4, warmup_ratio=0.0)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            return model, optimizer, scheduler, scaler, clone_state(model)

        torch.manual_seed(77)
        baseline_model, baseline_optimizer, baseline_scheduler, baseline_scaler, baseline_ema = make_training_objects()
        baseline_dataset = TinyDataset(labels)
        baseline_sampler = ResumableRandomSampler(baseline_dataset)
        baseline_loader = DataLoader(baseline_dataset, batch_size=2, sampler=baseline_sampler, num_workers=0)
        baseline_temporal = np.zeros((len(labels), 2), dtype=np.float16)
        baseline_seen = np.zeros(len(labels), dtype=np.bool_)
        torch.manual_seed(2026)
        baseline_stats, baseline_step, _ = run_fit_epoch(
            baseline_model, baseline_loader, torch.device("cpu"), args, quality, clean, labels,
            features, prototypes, labels.copy(), margins, class_weights, baseline_temporal,
            baseline_seen, baseline_optimizer, baseline_scheduler, baseline_scaler,
            baseline_ema, 0, 1,
        )

        torch.manual_seed(77)
        first_model, first_optimizer, first_scheduler, first_scaler, first_ema = make_training_objects()
        first_dataset = TinyDataset(labels)
        first_sampler = ResumableRandomSampler(first_dataset)
        first_loader = DataLoader(first_dataset, batch_size=2, sampler=first_sampler, num_workers=0)
        first_temporal = np.zeros((len(labels), 2), dtype=np.float16)
        first_seen = np.zeros(len(labels), dtype=np.bool_)
        saved = {}

        def stop_after_first_step(progress):
            saved.update(
                model={name: value.detach().clone() for name, value in first_model.state_dict().items()},
                optimizer=first_optimizer.state_dict(),
                scheduler=first_scheduler.state_dict(),
                scaler=first_scaler.state_dict(),
                ema={name: value.detach().clone() for name, value in first_ema.items()},
                temporal=first_temporal.copy(),
                seen=first_seen.copy(),
                order=list(first_sampler.order),
                progress=dict(progress),
                rng=capture_rng_state(),
            )
            raise StopAfterCheckpoint

        torch.manual_seed(2026)
        with self.assertRaises(StopAfterCheckpoint):
            run_fit_epoch(
                first_model, first_loader, torch.device("cpu"), args, quality, clean, labels,
                features, prototypes, labels.copy(), margins, class_weights, first_temporal,
                first_seen, first_optimizer, first_scheduler, first_scaler, first_ema, 0, 1,
                checkpoint_callback=stop_after_first_step,
            )

        resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler, _ = make_training_objects()
        resumed_model.load_state_dict(saved["model"])
        resumed_optimizer.load_state_dict(saved["optimizer"])
        resumed_scheduler.load_state_dict(saved["scheduler"])
        resumed_scaler.load_state_dict(saved["scaler"])
        resumed_ema = {name: value.detach().clone() for name, value in saved["ema"].items()}
        resumed_dataset = TinyDataset(labels)
        resumed_sampler = ResumableRandomSampler(
            resumed_dataset, saved["order"], saved["progress"]["samples_completed"]
        )
        resumed_loader = DataLoader(resumed_dataset, batch_size=2, sampler=resumed_sampler, num_workers=0)
        resumed_stats, resumed_step, _ = run_fit_epoch(
            resumed_model, resumed_loader, torch.device("cpu"), args, quality, clean, labels,
            features, prototypes, labels.copy(), margins, class_weights, saved["temporal"],
            saved["seen"], resumed_optimizer, resumed_scheduler, resumed_scaler, resumed_ema,
            saved["progress"]["optimizer_step"], 1,
            initial_stats=saved["progress"]["running_stats"],
            batch_offset=saved["progress"]["batches_completed"],
            resume_rng_state=saved["rng"],
        )

        self.assertEqual(baseline_step, resumed_step)
        self.assertEqual(baseline_stats, resumed_stats)
        for expected, actual in zip(baseline_model.parameters(), resumed_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))
        for name in baseline_ema:
            self.assertTrue(torch.equal(baseline_ema[name], resumed_ema[name]))

    def test_resume_checkpoint_rejects_changed_config_or_dataset(self):
        config = TrainConfig(model_dir="model", train_dir="train")
        payload = {
            "kind": "train_v5_resume",
            "format_version": 1,
            "class_names": ["a", "b"],
            "config": asdict(config),
            "dataset_signature": "expected",
            "stage": "validation",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            torch.save(payload, path)
            loaded = load_resume_checkpoint(path, torch.device("cpu"), ["a", "b"], config, "expected")
            self.assertEqual(loaded["stage"], "validation")
            changed = TrainConfig(model_dir="different", train_dir="train")
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_resume_checkpoint(path, torch.device("cpu"), ["a", "b"], changed, "expected")
            with self.assertRaisesRegex(ValueError, "dataset"):
                load_resume_checkpoint(path, torch.device("cpu"), ["a", "b"], config, "different")

    def test_complete_resume_payload_round_trip(self):
        config = TrainConfig(model_dir="model", train_dir="train")
        model = TinyClassifier()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = make_scheduler(optimizer, total_steps=4, warmup_ratio=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        state = clone_state(model)
        progress = {
            "batches_completed": 2,
            "samples_completed": 4,
            "running_stats": {"loss_sum": 3.0, "correct_sum": 2.0, "rows": 4.0, "pseudo_rows": 1.0},
            "optimizer_step": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_resume_checkpoint(
                Path(directory),
                class_names=["a", "b"],
                config=config,
                rows_signature="signature",
                classifier=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=state,
                temporal=np.zeros((4, 2), dtype=np.float16),
                temporal_seen=np.asarray([True, True, True, True]),
                best_state=state,
                best_score=0.75,
                best_full_accuracy=0.70,
                best_epoch=1,
                history=[{"epoch": 1.0}],
                stage="full_refit",
                epoch=2,
                sampler_order=[3, 2, 1, 0],
                progress=progress,
                validation_selected={"accuracy": 0.70},
                full_refit_history=[{"epoch": 1.0, "loss": 0.5}],
            )
            loaded = load_resume_checkpoint(
                path, torch.device("cpu"), ["a", "b"], config, "signature"
            )
        self.assertEqual(loaded["stage"], "full_refit")
        self.assertEqual(loaded["sampler_order"], [3, 2, 1, 0])
        self.assertEqual(loaded["optimizer_step"], 1)
        self.assertEqual(loaded["validation_selected"]["accuracy"], 0.70)
        self.assertIn("rng_state", loaded)


if __name__ == "__main__":
    unittest.main()
