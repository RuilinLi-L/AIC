"""V5 robust single-model training with paired views and prototype teachers.

The validation stage follows the V4 aggressive LoRA recipe while adding a
lightweight two-view consistency loss and a visual-prototype teacher for
low-quality/noisy rows.  After selecting the best EMA on a stratified holdout,
an optional short full-data refit produces the final inference checkpoint.
Only the official training directory is read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from robust_clip import (
    PairedFolderImageDataset,
    FolderImageDataset,
    RobustCLIPClassifier,
    TrainConfig,
    classwise_quality,
    classwise_rank_scores,
    effective_number_class_weights,
    inject_visual_lora,
    list_images,
    load_classifier_state,
    load_clip,
    prototype_pseudo_targets,
    resolve_device,
    robust_visual_prototypes_and_scores,
    save_checkpoint,
    seed_everything,
    stratified_split,
    symmetric_kl_loss,
    trainable_state_dict,
)
from train import load_or_create_feature_cache


RESUME_FORMAT_VERSION = 1
RESUME_INTERVAL_SECONDS = 90 * 60


class ResumableRandomSampler(Sampler[int]):
    """RandomSampler-compatible order that can restart at a saved position."""

    def __init__(
        self,
        data_source,
        order: Sequence[int] | None = None,
        start_position: int = 0,
    ) -> None:
        self.data_source = data_source
        self.order = None if order is None else [int(value) for value in order]
        self.start_position = int(start_position)
        if self.start_position < 0 or self.start_position > len(data_source):
            raise ValueError("sampler start position is outside the dataset")
        if self.order is not None:
            if len(self.order) != len(data_source) or sorted(self.order) != list(range(len(data_source))):
                raise ValueError("saved sampler order is not a permutation of the dataset")

    def __iter__(self):
        if self.order is None:
            # Match torch.utils.data.RandomSampler's generator behavior when no
            # explicit generator is supplied, while retaining the permutation.
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            self.order = torch.randperm(len(self.data_source), generator=generator).tolist()
        yield from self.order[self.start_position :]

    def __len__(self) -> int:
        return len(self.data_source) - self.start_position


class ResumeTimer:
    def __init__(
        self,
        interval_seconds: float = RESUME_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.interval_seconds = float(interval_seconds)
        self.clock = clock
        self.last_saved = self.clock()

    def due(self) -> bool:
        return self.clock() - self.last_saved >= self.interval_seconds

    def mark_saved(self) -> None:
        self.last_saved = self.clock()


def dataset_signature(rows: Sequence[tuple[str, int, str]]) -> str:
    digest = hashlib.sha256()
    for path, label, class_name in rows:
        resolved = str(Path(path).resolve())
        digest.update(resolved.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\0")
        digest.update(class_name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Resume checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def cpu_tensor_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def tensor_dict_to_device(state: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.detach().to(device).clone() for name, value in state.items()}


def timestamped_resume_path(output_dir: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"resume_{stamp}.pt"
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"resume_{stamp}_{suffix:02d}.pt"
        suffix += 1
    return candidate


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_resume_checkpoint(
    path: str | Path,
    device: torch.device,
    class_names: Sequence[str],
    config: TrainConfig,
    rows_signature: str,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    # Resume files contain Python/NumPy RNG state in addition to tensors, so
    # PyTorch 2.6's weights_only default cannot deserialize them. The path is
    # explicitly selected by the user and then validated below.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "train_v5_resume" or checkpoint.get("format_version") != RESUME_FORMAT_VERSION:
        raise ValueError(f"Unsupported V5 resume checkpoint: {checkpoint_path}")
    if list(checkpoint.get("class_names", [])) != list(class_names):
        raise ValueError("Resume checkpoint classes do not match train-dir")
    if checkpoint.get("config") != asdict(config):
        raise ValueError("Resume checkpoint training configuration does not match this run")
    if checkpoint.get("dataset_signature") != rows_signature:
        raise ValueError("Resume checkpoint dataset does not match train-dir")
    if checkpoint.get("stage") not in {"validation", "full_refit"}:
        raise ValueError("Resume checkpoint has an invalid training stage")
    return checkpoint


def save_resume_checkpoint(
    output_dir: Path,
    *,
    class_names: Sequence[str],
    config: TrainConfig,
    rows_signature: str,
    classifier: RobustCLIPClassifier,
    optimizer,
    scheduler,
    scaler,
    ema: dict[str, torch.Tensor],
    temporal: np.ndarray,
    temporal_seen: np.ndarray,
    best_state: dict[str, torch.Tensor],
    best_score: float,
    best_full_accuracy: float,
    best_epoch: int,
    history: list[dict[str, float]],
    stage: str,
    epoch: int,
    sampler_order: Sequence[int],
    progress: dict[str, Any],
    validation_selected: dict[str, float] | None,
    full_refit_history: list[dict[str, float]],
) -> Path:
    created_at = datetime.now()
    path = timestamped_resume_path(output_dir, created_at)
    payload = {
        "kind": "train_v5_resume",
        "format_version": RESUME_FORMAT_VERSION,
        "created_at": created_at.isoformat(timespec="seconds"),
        "class_names": list(class_names),
        "config": asdict(config),
        "dataset_signature": rows_signature,
        "model": trainable_state_dict(classifier),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": cpu_tensor_dict(ema),
        "temporal": np.asarray(temporal).copy(),
        "temporal_seen": np.asarray(temporal_seen).copy(),
        "best_state": cpu_tensor_dict(best_state),
        "best_score": float(best_score),
        "best_full_accuracy": float(best_full_accuracy),
        "best_epoch": int(best_epoch),
        "history": list(history),
        "stage": stage,
        "epoch": int(epoch),
        "sampler_order": [int(value) for value in sampler_order],
        "batches_completed": int(progress["batches_completed"]),
        "samples_completed": int(progress["samples_completed"]),
        "running_stats": dict(progress["running_stats"]),
        "optimizer_step": int(progress["optimizer_step"]),
        "validation_selected": validation_selected,
        "full_refit_history": list(full_refit_history),
        "rng_state": capture_rng_state(),
    }
    atomic_torch_save(payload, path)
    print(
        f"resume_checkpoint_saved={path.resolve()} stage={stage} epoch={epoch} "
        f"batch={payload['batches_completed']}",
        flush=True,
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--model-dir", default="clip-ViT-B-32")
    parser.add_argument("--output-dir", default="outputs/robust_visual_v5")
    parser.add_argument("--feature-cache", default="outputs/frozen_clip_train.npy")
    parser.add_argument("--head-init-checkpoint", default="outputs/robust_visual_v3_val05_tta_calibrated/model.pt")
    parser.add_argument("--resume", default=None, help="V5 resume checkpoint produced by an earlier run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--augmentation", choices=["none", "light", "strong"], default="light")
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-layers", type=int, default=6)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,out_proj")
    parser.add_argument("--tune-layernorm", action="store_true")
    parser.add_argument("--tune-visual-projection", action="store_true")
    parser.add_argument("--clean-fraction", type=float, default=0.75)
    parser.add_argument("--weight-floor", type=float, default=0.15)
    parser.add_argument("--class-weight-beta", type=float, default=0.9999)
    parser.add_argument("--weight-cap", type=float, default=2.0)
    parser.add_argument("--prototype-keep-fraction", type=float, default=0.70)
    parser.add_argument("--prototype-iterations", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.15)
    parser.add_argument("--pseudo-start-epoch", type=int, default=2)
    parser.add_argument("--pseudo-threshold", type=float, default=0.62)
    parser.add_argument("--pseudo-weight", type=float, default=0.72)
    parser.add_argument("--temporal-decay", type=float, default=0.85)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--trusted-val-fraction", type=float, default=0.70)
    parser.add_argument("--initial-logit-scale", type=float, default=30.0)
    parser.add_argument("--max-logit-scale", type=float, default=50.0)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--consistency-temperature", type=float, default=2.0)
    parser.add_argument("--prototype-pseudo-start-epoch", type=int, default=2)
    parser.add_argument("--prototype-pseudo-margin", type=float, default=0.08)
    parser.add_argument("--prototype-pseudo-weight", type=float, default=0.35)
    parser.add_argument("--prototype-teacher-mix", type=float, default=0.5)
    parser.add_argument("--prototype-temperature", type=float, default=0.07)
    parser.add_argument("--full-refit-epochs", type=int, default=2)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    targets = tuple(part.strip() for part in args.lora_targets.split(",") if part.strip())
    if not targets:
        raise ValueError("--lora-targets cannot be empty")
    if args.batch_size < 1 or args.epochs < 1 or args.gradient_accumulation < 1:
        raise ValueError("batch-size, epochs and gradient-accumulation must be positive")
    if args.full_refit_epochs < 0:
        raise ValueError("full-refit-epochs must be non-negative")
    if args.lora_rank < 1 or args.lora_layers < 1:
        raise ValueError("LoRA rank and layer count must be positive")
    for name in ("clean_fraction", "prototype_keep_fraction", "pseudo_threshold", "trusted_val_fraction"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    for name in (
        "label_smoothing",
        "pseudo_weight",
        "temporal_decay",
        "ema_decay",
        "warmup_ratio",
        "prototype_teacher_mix",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.consistency_weight < 0.0 or args.consistency_temperature <= 0.0:
        raise ValueError("consistency weight must be non-negative and temperature positive")
    if not 0.0 <= args.prototype_pseudo_weight <= 1.0 or args.prototype_pseudo_margin < 0.0:
        raise ValueError("prototype pseudo weight must be in [0, 1] and margin non-negative")
    if args.prototype_temperature <= 0.0:
        raise ValueError("prototype temperature must be positive")
    return targets


def enable_extra_visual_parameters(
    classifier: RobustCLIPClassifier,
    last_n: int,
    layernorm: bool,
    projection: bool,
) -> list[str]:
    names: list[str] = []
    layers = classifier.clip.vision_model.encoder.layers
    if layernorm:
        for layer_id in range(len(layers) - last_n, len(layers)):
            for norm_name in ("layer_norm1", "layer_norm2"):
                for parameter_name, parameter in getattr(layers[layer_id], norm_name).named_parameters():
                    parameter.requires_grad_(True)
                    names.append(f"vision_model.encoder.layers.{layer_id}.{norm_name}.{parameter_name}")
    if projection:
        for parameter_name, parameter in classifier.clip.visual_projection.named_parameters():
            parameter.requires_grad_(True)
            names.append(f"visual_projection.{parameter_name}")
    return names


def make_config(args: argparse.Namespace, targets: tuple[str, ...]) -> TrainConfig:
    return TrainConfig(
        model_dir=args.model_dir,
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        rounds=1,
        lr=args.head_lr,
        weight_decay=args.weight_decay,
        bottleneck=args.bottleneck,
        clean_fraction=args.clean_fraction,
        weight_floor=args.weight_floor,
        class_weight_beta=args.class_weight_beta,
        weight_cap=args.weight_cap,
        augmentation=args.augmentation,
        prototype_keep_fraction=args.prototype_keep_fraction,
        prototype_iterations=args.prototype_iterations,
        initial_logit_scale=args.initial_logit_scale,
        max_logit_scale=args.max_logit_scale,
        learn_logit_scale=False,
        device=args.device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        lora_targets=targets,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        anchor_weight=args.anchor_weight,
        label_smoothing=args.label_smoothing,
        pseudo_start_epoch=args.pseudo_start_epoch,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_weight=args.pseudo_weight,
        temporal_decay=args.temporal_decay,
        ema_decay=args.ema_decay,
        gradient_accumulation=args.gradient_accumulation,
        tune_layernorm=args.tune_layernorm,
        tune_visual_projection=args.tune_visual_projection,
        trusted_val_fraction=args.trusted_val_fraction,
        consistency_weight=args.consistency_weight,
        consistency_temperature=args.consistency_temperature,
        prototype_pseudo_start_epoch=args.prototype_pseudo_start_epoch,
        prototype_pseudo_margin=args.prototype_pseudo_margin,
        prototype_pseudo_weight=args.prototype_pseudo_weight,
        prototype_teacher_mix=args.prototype_teacher_mix,
        prototype_temperature=args.prototype_temperature,
        full_refit_epochs=args.full_refit_epochs,
    )


def trainable_parameters(classifier: RobustCLIPClassifier) -> dict[str, torch.nn.Parameter]:
    return {name: value for name, value in classifier.named_parameters() if value.requires_grad}


def clone_trainable(classifier: RobustCLIPClassifier) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in trainable_parameters(classifier).items()}


def update_ema(shadow: dict[str, torch.Tensor], classifier: RobustCLIPClassifier, decay: float, step: int) -> None:
    effective_decay = min(decay, (1.0 + step) / (10.0 + step))
    for name, parameter in trainable_parameters(classifier).items():
        shadow[name].mul_(effective_decay).add_(parameter.detach(), alpha=1.0 - effective_decay)


@contextmanager
def swapped_trainable(classifier: RobustCLIPClassifier, state: dict[str, torch.Tensor]):
    parameters = trainable_parameters(classifier)
    backup = {name: value.detach().clone() for name, value in parameters.items()}
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(state[name])
    try:
        yield
    finally:
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(backup[name])


def build_optimizer(classifier: RobustCLIPClassifier, args: argparse.Namespace):
    visual, head = [], []
    for name, parameter in classifier.named_parameters():
        if not parameter.requires_grad:
            continue
        (visual if name.startswith("clip.") else head).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": visual, "lr": args.lora_lr}, {"params": head, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    return optimizer, visual, head


def trusted_validation_mask(
    features: np.ndarray,
    labels: np.ndarray,
    val_indices: list[int],
    prototypes: np.ndarray,
    fraction: float,
) -> np.ndarray:
    indices = np.asarray(val_indices, dtype=np.int64)
    scores = np.sum(np.asarray(features[indices], dtype=np.float32) * prototypes[labels[indices]], axis=1)
    trusted = np.zeros(len(indices), dtype=np.bool_)
    val_labels = labels[indices]
    for label in np.unique(val_labels):
        local = np.where(val_labels == label)[0]
        keep = max(1, int(math.ceil(len(local) * fraction)))
        trusted[local[np.argsort(scores[local])[::-1][:keep]]] = True
    return trusted


@torch.no_grad()
def evaluate(
    classifier: RobustCLIPClassifier,
    loader: DataLoader,
    device: torch.device,
    trusted_mask: np.ndarray,
    save_logits: Path | None = None,
) -> dict[str, float]:
    classifier.eval()
    logits_batches, labels_batches = [], []
    amp = device.type == "cuda"
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = classifier(pixels, None)[0]
        logits_batches.append(logits.float().cpu())
        labels_batches.append(batch["label"])
    logits = torch.cat(logits_batches)
    labels = torch.cat(labels_batches)
    trusted = torch.as_tensor(trusted_mask, dtype=torch.bool)
    result = {
        "loss": float(F.cross_entropy(logits, labels)),
        "accuracy": float((logits.argmax(dim=1) == labels).float().mean()),
        "trusted_accuracy": float((logits[trusted].argmax(dim=1) == labels[trusted]).float().mean()),
        "trusted_rows": int(trusted.sum()),
    }
    if save_logits is not None:
        save_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_logits, logits.numpy())
    return result


def make_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-3, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def run_fit_epoch(
    classifier: RobustCLIPClassifier,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    quality: np.ndarray,
    clean: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    prototypes: torch.Tensor,
    prototype_pseudo: np.ndarray,
    prototype_margin: np.ndarray,
    class_weights: torch.Tensor,
    temporal: np.ndarray,
    temporal_seen: np.ndarray,
    optimizer,
    scheduler,
    scaler,
    ema: dict[str, torch.Tensor],
    optimizer_step: int,
    epoch: int,
    initial_stats: dict[str, float] | None = None,
    batch_offset: int = 0,
    resume_rng_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, float], int, int]:
    classifier.train()
    optimizer.zero_grad(set_to_none=True)
    initial = initial_stats or {}
    running_loss = float(initial.get("loss_sum", 0.0))
    running_correct = float(initial.get("correct_sum", 0.0))
    running_rows = float(initial.get("rows", 0.0))
    pseudo_rows = float(initial.get("pseudo_rows", 0.0))
    amp = device.type == "cuda"
    loader_iterator = iter(loader)
    if resume_rng_state is not None:
        # DataLoader iterator construction consumes an RNG value for its base
        # seed. Restore after construction so the next augmentation/model RNG
        # draw exactly follows the saved batch.
        restore_rng_state(resume_rng_state)
    completed_batches = int(batch_offset)
    for local_batch_id, batch in enumerate(loader_iterator, 1):
        batch_id = batch_offset + local_batch_id
        pixels_a = batch["pixel_values_a"].to(device, non_blocking=True)
        pixels_b = batch["pixel_values_b"].to(device, non_blocking=True)
        y = batch["label"].to(device)
        idx = batch["row_index"].numpy()
        weights = torch.as_tensor(quality[idx], dtype=torch.float32, device=device) * class_weights[y]
        frozen = torch.as_tensor(np.asarray(features[idx]), dtype=torch.float32, device=device)
        proto_logits = F.normalize(frozen, dim=1) @ F.normalize(prototypes, dim=1).t()
        proto_soft = F.softmax(proto_logits / args.prototype_temperature, dim=1)
        proto_label = torch.as_tensor(prototype_pseudo[idx], dtype=torch.long, device=device)
        # A 500-way softmax is naturally diffuse even when the top prototype
        # has a healthy cosine margin.  Keep a calibrated soft component, but
        # put most teacher mass on the screened prototype class so the existing
        # confidence threshold remains meaningful.
        proto_teacher = 0.25 * proto_soft + 0.75 * F.one_hot(
            proto_label, num_classes=proto_soft.shape[1]
        ).float()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits_a, _, base_a, _ = classifier(pixels_a, None)
            logits_b, _, base_b, _ = classifier(pixels_b, None)
            hard_a = F.cross_entropy(logits_a, y, reduction="none", label_smoothing=args.label_smoothing)
            hard_b = F.cross_entropy(logits_b, y, reduction="none", label_smoothing=args.label_smoothing)
            sample_a, sample_b = hard_a, hard_b
            eligible = torch.zeros(len(y), dtype=torch.bool, device=device)
            if epoch > args.prototype_pseudo_start_epoch:
                old_seen = torch.as_tensor(temporal_seen[idx], dtype=torch.bool, device=device)
                temporal_teacher = torch.as_tensor(np.asarray(temporal[idx]), dtype=torch.float32, device=device)
                temporal_teacher = temporal_teacher / temporal_teacher.sum(dim=1, keepdim=True).clamp_min(1e-6)
                mix = float(args.prototype_teacher_mix)
                teacher = torch.where(
                    old_seen[:, None],
                    mix * proto_teacher + (1.0 - mix) * temporal_teacher,
                    proto_teacher,
                )
                teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-6)
                confidence, pseudo_label = teacher.max(dim=1)
                proto_ok = torch.as_tensor(
                    (~clean[idx])
                    & (prototype_pseudo[idx] != labels[idx])
                    & (prototype_margin[idx] >= args.prototype_pseudo_margin),
                    dtype=torch.bool,
                    device=device,
                )
                eligible = proto_ok & (confidence >= args.pseudo_threshold) & pseudo_label.ne(y)
                soft_a = -(teacher.detach() * F.log_softmax(logits_a, dim=1)).sum(dim=1)
                soft_b = -(teacher.detach() * F.log_softmax(logits_b, dim=1)).sum(dim=1)
                repaired_a = (1.0 - args.prototype_pseudo_weight) * hard_a + args.prototype_pseudo_weight * soft_a
                repaired_b = (1.0 - args.prototype_pseudo_weight) * hard_b + args.prototype_pseudo_weight * soft_b
                sample_a = torch.where(eligible, repaired_a, hard_a)
                sample_b = torch.where(eligible, repaired_b, hard_b)
                pseudo_rows += float(eligible.sum())
            weighted = weights
            loss_a = (weighted * sample_a).sum() / weighted.sum().clamp_min(1e-6)
            loss_b = (weighted * sample_b).sum() / weighted.sum().clamp_min(1e-6)
            consistency = symmetric_kl_loss(logits_a, logits_b, args.consistency_temperature)
            loss_consistency = (weighted * consistency).sum() / weighted.sum().clamp_min(1e-6)
            loss_anchor = 0.5 * (
                (1.0 - (base_a * F.normalize(frozen, dim=1)).sum(dim=1)).mean()
                + (1.0 - (base_b * F.normalize(frozen, dim=1)).sum(dim=1)).mean()
            )
            loss = 0.5 * (loss_a + loss_b) + args.consistency_weight * loss_consistency + args.anchor_weight * loss_anchor

        current_probs = 0.5 * (logits_a.detach().softmax(dim=1) + logits_b.detach().softmax(dim=1))
        current_probs = current_probs.float().cpu().numpy()
        seen = temporal_seen[idx]
        if seen.any():
            temporal[idx[seen]] = (
                args.temporal_decay * temporal[idx[seen]].astype(np.float32)
                + (1.0 - args.temporal_decay) * current_probs[seen]
            ).astype(np.float16)
        if (~seen).any():
            temporal[idx[~seen]] = current_probs[~seen].astype(np.float16)
        temporal_seen[idx] = True

        scaler.scale(loss / args.gradient_accumulation).backward()
        should_step = batch_id % args.gradient_accumulation == 0 or local_batch_id == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            train_params = [p for p in classifier.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_step += 1
            update_ema(ema, classifier, args.ema_decay, optimizer_step)
        running_loss += float(loss.detach()) * len(y)
        running_correct += float((logits_a.argmax(dim=1) == y).sum())
        running_rows += len(y)
        completed_batches = batch_id
        if should_step and checkpoint_callback is not None:
            checkpoint_callback(
                {
                    "batches_completed": completed_batches,
                    "samples_completed": int(running_rows),
                    "running_stats": {
                        "loss_sum": running_loss,
                        "correct_sum": running_correct,
                        "rows": running_rows,
                        "pseudo_rows": pseudo_rows,
                    },
                    "optimizer_step": optimizer_step,
                }
            )
    return {
        "loss": running_loss / max(running_rows, 1.0),
        "accuracy": running_correct / max(running_rows, 1.0),
        "pseudo_rows": int(pseudo_rows),
    }, optimizer_step, completed_batches


def main() -> None:
    args = parse_args()
    targets = validate_args(args)
    config = make_config(args, targets)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = list_images(args.train_dir)
    labels = np.asarray([label for _, label, _ in rows], dtype=np.int64)
    class_names = [name for _, name in sorted({(label, name) for _, label, name in rows})]
    all_indices = list(range(len(rows)))
    train_indices, val_indices = stratified_split(rows, args.val_ratio, args.seed)
    rows_signature = dataset_signature(rows)
    model, processor = load_clip(args.model_dir, device)
    features, cached_labels = load_or_create_feature_cache(model, processor, rows, config, device, Path(args.feature_cache))
    if not np.array_equal(labels, cached_labels):
        raise ValueError("Feature-cache labels do not match the training folders")

    prototypes, prototype_scores = robust_visual_prototypes_and_scores(
        features,
        labels,
        len(class_names),
        candidate_indices=train_indices,
        keep_fraction=args.prototype_keep_fraction,
        iterations=args.prototype_iterations,
    )
    ranked_scores = classwise_rank_scores(labels, prototype_scores, train_indices)
    quality, clean = classwise_quality(
        rows, ranked_scores, args.clean_fraction, args.weight_floor, candidate_indices=train_indices
    )
    prototype_pseudo, prototype_confidence, prototype_margin = prototype_pseudo_targets(
        features, prototypes, args.prototype_temperature
    )
    trusted_mask = trusted_validation_mask(features, labels, val_indices, prototypes, args.trusted_val_fraction)

    classifier = RobustCLIPClassifier(
        model,
        classifier_init=torch.from_numpy(prototypes).to(device),
        bottleneck=args.bottleneck,
        initial_logit_scale=args.initial_logit_scale,
        max_logit_scale=args.max_logit_scale,
        learn_logit_scale=False,
    ).to(device)
    if args.resume is None and args.head_init_checkpoint:
        head_path = Path(args.head_init_checkpoint)
        if head_path.is_file():
            head_checkpoint = torch.load(head_path, map_location="cpu")
            if list(head_checkpoint.get("class_names", [])) != class_names:
                raise ValueError("Head warm-start classes do not match train-dir")
            load_classifier_state(classifier, head_checkpoint["model"])
            print(f"head_warm_start={head_path.resolve()}", flush=True)
        else:
            print(f"head_warm_start_missing={head_path}; using robust visual prototypes", flush=True)

    replaced = inject_visual_lora(
        classifier.clip,
        last_n_layers=args.lora_layers,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=targets,
    )
    extra = enable_extra_visual_parameters(classifier, args.lora_layers, args.tune_layernorm, args.tune_visual_projection)
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = load_resume_checkpoint(args.resume, device, class_names, config, rows_signature)
        load_classifier_state(classifier, resume_checkpoint["model"])

    optimizer, visual_params, head_params = build_optimizer(classifier, args)
    steps_per_epoch = math.ceil(len(train_indices) / args.batch_size / args.gradient_accumulation)
    scheduler = make_scheduler(optimizer, max(1, args.epochs * steps_per_epoch), args.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_dataset = PairedFolderImageDataset(rows, train_indices, processor, augment=args.augmentation)
    val_loader = DataLoader(
        FolderImageDataset(rows, val_indices, processor, augment="none"),
        batch_size=max(args.batch_size, 32),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    class_counts = np.bincount(labels[np.asarray(train_indices)], minlength=len(class_names))
    class_weights = torch.tensor(effective_number_class_weights(class_counts, args.class_weight_beta, args.weight_cap), dtype=torch.float32, device=device)
    prototypes_tensor = torch.from_numpy(prototypes).to(device=device, dtype=torch.float32)
    temporal = np.zeros((len(rows), len(class_names)), dtype=np.float16)
    temporal_seen = np.zeros(len(rows), dtype=np.bool_)
    ema = clone_trainable(classifier)
    best_state = clone_trainable(classifier)
    best_score = -1.0
    best_full_accuracy = -1.0
    best_epoch = 0
    optimizer_step = 0
    history: list[dict[str, float]] = []
    validation_selected: dict[str, float] | None = None
    full_refit_history: list[dict[str, float]] = []

    if resume_checkpoint is not None:
        best_state = tensor_dict_to_device(resume_checkpoint["best_state"], device)
        best_score = float(resume_checkpoint["best_score"])
        best_full_accuracy = float(resume_checkpoint["best_full_accuracy"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        history = list(resume_checkpoint["history"])
        validation_selected = resume_checkpoint.get("validation_selected")
        full_refit_history = list(resume_checkpoint.get("full_refit_history", []))
        print(
            f"resumed_from={Path(args.resume).resolve()} stage={resume_checkpoint['stage']} "
            f"epoch={resume_checkpoint['epoch']} batch={resume_checkpoint['batches_completed']}",
            flush=True,
        )

    print(
        f"device={device} train={len(train_indices)} val={len(val_indices)} trusted_val={int(trusted_mask.sum())} "
        f"lora_modules={len(replaced)} extra_visual_tensors={len(extra)} "
        f"trainable_visual={sum(p.numel() for p in visual_params):,} trainable_head={sum(p.numel() for p in head_params):,} "
        f"prototype_pseudo_candidates={int(((~clean) & (prototype_pseudo != labels) & (prototype_margin >= args.prototype_pseudo_margin)).sum())}",
        flush=True,
    )

    checkpoint_timer = ResumeTimer()
    resume_stage = resume_checkpoint["stage"] if resume_checkpoint is not None else None

    if resume_stage != "full_refit":
        if resume_checkpoint is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
            scaler.load_state_dict(resume_checkpoint["scaler"])
            ema = tensor_dict_to_device(resume_checkpoint["ema"], device)
            temporal = np.asarray(resume_checkpoint["temporal"], dtype=np.float16).copy()
            temporal_seen = np.asarray(resume_checkpoint["temporal_seen"], dtype=np.bool_).copy()
            optimizer_step = int(resume_checkpoint["optimizer_step"])
            start_epoch = int(resume_checkpoint["epoch"])
            if not 1 <= start_epoch <= args.epochs:
                raise ValueError("Resume checkpoint validation epoch is outside configured epochs")
        else:
            start_epoch = 1

        for epoch in range(start_epoch, args.epochs + 1):
            continuing = resume_checkpoint is not None and epoch == start_epoch
            saved_order = resume_checkpoint["sampler_order"] if continuing else None
            samples_completed = int(resume_checkpoint["samples_completed"]) if continuing else 0
            batch_offset = int(resume_checkpoint["batches_completed"]) if continuing else 0
            initial_stats = resume_checkpoint["running_stats"] if continuing else None
            resume_rng = resume_checkpoint["rng_state"] if continuing else None
            sampler = ResumableRandomSampler(train_dataset, saved_order, samples_completed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=False,
            )

            def save_validation_resume(progress: dict[str, Any]) -> None:
                if not checkpoint_timer.due():
                    return
                if sampler.order is None:
                    raise RuntimeError("Training sampler did not expose its current order")
                save_resume_checkpoint(
                    out,
                    class_names=class_names,
                    config=config,
                    rows_signature=rows_signature,
                    classifier=classifier,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    temporal=temporal,
                    temporal_seen=temporal_seen,
                    best_state=best_state,
                    best_score=best_score,
                    best_full_accuracy=best_full_accuracy,
                    best_epoch=best_epoch,
                    history=history,
                    stage="validation",
                    epoch=epoch,
                    sampler_order=sampler.order,
                    progress=progress,
                    validation_selected=None,
                    full_refit_history=full_refit_history,
                )
                checkpoint_timer.mark_saved()

            stats, optimizer_step, _ = run_fit_epoch(
                classifier, train_loader, device, args, quality, clean, labels, features, prototypes_tensor,
                prototype_pseudo, prototype_margin, class_weights, temporal, temporal_seen,
                optimizer, scheduler, scaler, ema, optimizer_step, epoch,
                initial_stats=initial_stats,
                batch_offset=batch_offset,
                resume_rng_state=resume_rng,
                checkpoint_callback=save_validation_resume,
            )
            resume_checkpoint = None
            with swapped_trainable(classifier, ema):
                val_stats = evaluate(classifier, val_loader, device, trusted_mask)
            selection_score = 0.6 * val_stats["trusted_accuracy"] + 0.4 * val_stats["accuracy"]
            record = {
                "epoch": epoch,
                "train_loss": stats["loss"],
                "train_noisy_accuracy": stats["accuracy"],
                "pseudo_repaired_rows": stats["pseudo_rows"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "trusted_val_accuracy": val_stats["trusted_accuracy"],
                "selection_score": selection_score,
                "visual_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
            }
            history.append(record)
            print(
                f"epoch={epoch}/{args.epochs} train_loss={record['train_loss']:.5f} train_acc={record['train_noisy_accuracy']:.4f} "
                f"pseudo={int(stats['pseudo_rows'])} val_acc={val_stats['accuracy']:.4f} "
                f"trusted_acc={val_stats['trusted_accuracy']:.4f} score={selection_score:.4f}",
                flush=True,
            )
            improved = selection_score > best_score or (selection_score == best_score and val_stats["accuracy"] > best_full_accuracy)
            if improved:
                best_score = selection_score
                best_full_accuracy = val_stats["accuracy"]
                best_epoch = epoch
                best_state = {name: value.detach().clone() for name, value in ema.items()}
                with swapped_trainable(classifier, best_state):
                    save_checkpoint(out / "best_model.pt", classifier, class_names, config, quality)
                (out / "metrics_progress.json").write_text(json.dumps({"selected_epoch": best_epoch, "history": history}, indent=2), encoding="utf-8")

        with torch.no_grad():
            for name, parameter in trainable_parameters(classifier).items():
                parameter.copy_(best_state[name])
        validation_selected = evaluate(classifier, val_loader, device, trusted_mask)

    final_quality = quality
    final_clean = clean
    if args.full_refit_epochs > 0:
        # Build full-data weights from prototypes that were constructed using
        # the original train split.  The refit intentionally consumes all
        # official training labels after validation selection.
        full_scores = np.sum(features * prototypes[labels], axis=1).astype(np.float32)
        full_ranked = classwise_rank_scores(labels, full_scores, all_indices)
        final_quality, final_clean = classwise_quality(
            rows, full_ranked, args.clean_fraction, args.weight_floor, candidate_indices=all_indices
        )
        full_dataset = PairedFolderImageDataset(rows, all_indices, processor, augment=args.augmentation)
        full_counts = np.bincount(labels, minlength=len(class_names))
        full_class_weights = torch.tensor(effective_number_class_weights(full_counts, args.class_weight_beta, args.weight_cap), dtype=torch.float32, device=device)
        full_optimizer, _, _ = build_optimizer(classifier, args)
        full_steps = math.ceil(len(rows) / args.batch_size / args.gradient_accumulation)
        full_scheduler = make_scheduler(full_optimizer, max(1, args.full_refit_epochs * full_steps), args.warmup_ratio)
        full_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        if resume_stage == "full_refit":
            if resume_checkpoint is None:
                raise RuntimeError("Full-refit resume state was unexpectedly cleared")
            if validation_selected is None:
                raise ValueError("Full-refit resume checkpoint is missing validation metrics")
            full_optimizer.load_state_dict(resume_checkpoint["optimizer"])
            full_scheduler.load_state_dict(resume_checkpoint["scheduler"])
            full_scaler.load_state_dict(resume_checkpoint["scaler"])
            full_temporal = np.asarray(resume_checkpoint["temporal"], dtype=np.float16).copy()
            full_seen = np.asarray(resume_checkpoint["temporal_seen"], dtype=np.bool_).copy()
            full_ema = tensor_dict_to_device(resume_checkpoint["ema"], device)
            full_step = int(resume_checkpoint["optimizer_step"])
            start_refit_epoch = int(resume_checkpoint["epoch"])
            if not 1 <= start_refit_epoch <= args.full_refit_epochs:
                raise ValueError("Resume checkpoint full-refit epoch is outside configured epochs")
        else:
            full_temporal = np.zeros((len(rows), len(class_names)), dtype=np.float16)
            full_seen = np.zeros(len(rows), dtype=np.bool_)
            full_ema = clone_trainable(classifier)
            full_step = 0
            start_refit_epoch = 1

        for refit_epoch in range(start_refit_epoch, args.full_refit_epochs + 1):
            continuing = resume_stage == "full_refit" and resume_checkpoint is not None and refit_epoch == start_refit_epoch
            saved_order = resume_checkpoint["sampler_order"] if continuing else None
            samples_completed = int(resume_checkpoint["samples_completed"]) if continuing else 0
            batch_offset = int(resume_checkpoint["batches_completed"]) if continuing else 0
            initial_stats = resume_checkpoint["running_stats"] if continuing else None
            resume_rng = resume_checkpoint["rng_state"] if continuing else None
            sampler = ResumableRandomSampler(full_dataset, saved_order, samples_completed)
            full_loader = DataLoader(
                full_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=False,
            )

            def save_full_refit_resume(progress: dict[str, Any]) -> None:
                if not checkpoint_timer.due():
                    return
                if sampler.order is None:
                    raise RuntimeError("Full-refit sampler did not expose its current order")
                save_resume_checkpoint(
                    out,
                    class_names=class_names,
                    config=config,
                    rows_signature=rows_signature,
                    classifier=classifier,
                    optimizer=full_optimizer,
                    scheduler=full_scheduler,
                    scaler=full_scaler,
                    ema=full_ema,
                    temporal=full_temporal,
                    temporal_seen=full_seen,
                    best_state=best_state,
                    best_score=best_score,
                    best_full_accuracy=best_full_accuracy,
                    best_epoch=best_epoch,
                    history=history,
                    stage="full_refit",
                    epoch=refit_epoch,
                    sampler_order=sampler.order,
                    progress=progress,
                    validation_selected=validation_selected,
                    full_refit_history=full_refit_history,
                )
                checkpoint_timer.mark_saved()

            stats, full_step, _ = run_fit_epoch(
                classifier, full_loader, device, args, final_quality, final_clean, labels, features, prototypes_tensor,
                prototype_pseudo, prototype_margin, full_class_weights, full_temporal, full_seen,
                # The refit starts from a mature validation checkpoint, so
                # enable prototype repair immediately instead of waiting for
                # two additional cold-start epochs.
                full_optimizer, full_scheduler, full_scaler, full_ema, full_step,
                args.prototype_pseudo_start_epoch + refit_epoch,
                initial_stats=initial_stats,
                batch_offset=batch_offset,
                resume_rng_state=resume_rng,
                checkpoint_callback=save_full_refit_resume,
            )
            resume_checkpoint = None
            full_refit_history.append({"epoch": refit_epoch, **stats})
            print(f"full_refit_epoch={refit_epoch}/{args.full_refit_epochs} loss={stats['loss']:.5f} acc={stats['accuracy']:.4f} pseudo={int(stats['pseudo_rows'])}", flush=True)
        with torch.no_grad():
            for name, parameter in trainable_parameters(classifier).items():
                parameter.copy_(full_ema[name])

    if validation_selected is None:
        raise RuntimeError("Validation-selected metrics are unavailable")
    selected = evaluate(classifier, val_loader, device, trusted_mask, out / "val_logits.npy")
    save_checkpoint(out / "model.pt", classifier, class_names, config, final_quality)
    (out / "classes.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "selected_epoch": best_epoch,
                "validation_selected": validation_selected,
                "selected": selected,
                "selection_score": best_score,
                "history": history,
                "full_refit_epochs": args.full_refit_epochs,
                "full_refit_history": full_refit_history,
                "full_refit_quality_mean": float(final_quality.mean()),
                "prototype_pseudo_candidate_rows": int(((final_quality > 0) & (~final_clean) & (prototype_pseudo != labels) & (prototype_margin >= args.prototype_pseudo_margin)).sum()),
                "prototype_pseudo_confidence_mean": float(prototype_confidence.mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(out / "sample_quality.npy", final_quality)
    print(f"selected_epoch={best_epoch} final_val_acc={selected['accuracy']:.4f} saved={out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
