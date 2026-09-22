"""Train the V6 single-model robust CLIP system.

V6 uses only the official training directory and the local official
CLIP ViT-B/32 checkpoint.  It first performs a validation run for model
selection, then rebuilds a fresh model and repeats the selected schedule on
all official training rows.  Test images are never accepted by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from robust_clip import (
    FolderImageDataset,
    PairedFolderImageDataset,
    RobustCLIPClassifier,
    _clip_image_transform,
    _feature_tensor,
    inject_visual_lora,
    list_images,
    load_classifier_state,
    load_clip,
    resolve_device,
    safe_open_image,
    seed_everything,
    symmetric_kl_loss,
    trainable_state_dict,
)
from train_v5 import atomic_torch_save, clone_trainable, swapped_trainable, update_ema
from v6_core import (
    PrototypeSignals,
    SelectiveFeatureQueue,
    balanced_softmax_cross_entropy,
    build_repair_plan,
    capped_stratified_split,
    classwise_percentile,
    cross_fitted_visual_signals,
    fixed_prototype_signals,
    fused_reliability,
    generalized_cross_entropy,
    head_mid_tail_accuracy,
    initial_reliability,
    macro_accuracy,
    macro_nll,
    project_conflicting_gradients,
    prototype_margin_thresholds,
    reliable_class_prior,
    selective_supervised_contrastive_loss,
)


FORMAT_VERSION = 6
RESUME_VERSION = 1
SEED = 2026
WARMUP_EPOCHS = 2
MAX_EPOCHS = 12
GRADIENT_ACCUMULATION = 4
QUEUE_PER_CLASS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--model-dir", default="clip-ViT-B-32")
    parser.add_argument("--output-dir", default="outputs/robust_visual_v6")
    parser.add_argument("--feature-cache", default="outputs/frozen_clip_multiview_v6.npy")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    if Path(args.train_dir).resolve() == Path(args.output_dir).resolve():
        raise ValueError("output-dir must not be the training directory")


def configure_determinism(seed: int) -> None:
    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def dataset_signature(rows: Sequence[tuple[str, int, str]]) -> str:
    digest = hashlib.sha256()
    for path, label, class_name in rows:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\0")
        digest.update(class_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": "v6",
        "model_dir": args.model_dir,
        "train_dir": args.train_dir,
        "output_dir": args.output_dir,
        "seed": SEED,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "epochs": MAX_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "augmentation": "light",
        "bottleneck": 128,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.0,
        "lora_layers": 4,
        "lora_targets": ("q_proj", "v_proj"),
        "lora_lr": 1e-5,
        "head_lr": 5e-5,
        "weight_decay": 1e-4,
        "warmup_ratio": 0.05,
        "initial_logit_scale": 30.0,
        "max_logit_scale": 50.0,
        "learn_logit_scale": False,
        "ema_decay": 0.999,
        "consistency_weight": 0.05,
        "anchor_weight": 0.10,
        "contrastive_weight": 0.10,
        "contrastive_temperature": 0.07,
        "gce_q": 0.7,
        "quality_threshold": 0.70,
        "repair_start_epoch": 5,
        "repair_confidence": 0.80,
        "repair_fraction": 0.15,
        "repair_max_weight": 0.50,
        "prototype_temperature": 0.07,
        "prototype_keep_fraction": 0.70,
        "prototype_iterations": 2,
        "prototype_folds": 5,
        "queue_per_class": QUEUE_PER_CLASS,
        "temporal_decay": 0.85,
        "consistency_temperature": 2.0,
        "gradient_projection": "uncertain_orthogonal_to_trusted_on_negative_dot",
        "repair_margin_quantile": 0.75,
        "validation_split": "tail_safe_capped_8_percent",
        "fixed_feature_views": ["center", "hflip", "light_seed_2", "light_seed_3"],
    }


def validate_clip_vit_b32(model) -> None:
    vision = model.config.vision_config
    observed = {
        "projection_dim": int(model.config.projection_dim),
        "hidden_size": int(vision.hidden_size),
        "layers": int(vision.num_hidden_layers),
        "patch_size": int(vision.patch_size),
        "image_size": int(vision.image_size),
    }
    expected = {
        "projection_dim": 512,
        "hidden_size": 768,
        "layers": 12,
        "patch_size": 32,
        "image_size": 224,
    }
    if observed != expected:
        raise ValueError(f"V6 requires official CLIP ViT-B/32 architecture: {observed}")


class DeterministicMultiViewDataset(Dataset):
    """Center, horizontal-flip and two deterministic light views."""

    def __init__(self, rows, processor, seed: int = SEED):
        self.rows = rows
        self.seed = int(seed)
        self.center = _clip_image_transform(processor, "none")
        self.light_a = _clip_image_transform(processor, "light")
        self.light_b = _clip_image_transform(processor, "light")

    def __len__(self) -> int:
        return len(self.rows)

    def _fixed_light(self, transform, image, row_index: int, view_id: int) -> torch.Tensor:
        # These transforms run on CPU, often inside spawned DataLoader workers.
        # Forking CUDA RNG there can initialize a CUDA context in every worker.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + row_index * 1009 + view_id * 9176)
            return transform(image)

    def __getitem__(self, item: int) -> dict[str, object]:
        path, label, class_name = self.rows[item]
        image = safe_open_image(path)
        center = self.center(image)
        views = torch.stack(
            [
                center,
                torch.flip(center, dims=[2]),
                self._fixed_light(self.light_a, image, item, 2),
                self._fixed_light(self.light_b, image, item, 3),
            ],
            dim=0,
        )
        return {
            "views": views,
            "label": torch.tensor(label, dtype=torch.long),
            "row_index": torch.tensor(item, dtype=torch.long),
            "class_name": class_name,
        }


def _feature_cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


@torch.no_grad()
def load_or_create_multiview_features(
    model,
    processor,
    rows,
    cache_path: Path,
    signature: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    metadata_path = _feature_cache_metadata_path(cache_path)
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = np.load(cache_path, mmap_mode="r")
        expected = (4, len(rows), int(model.config.projection_dim))
        if metadata.get("dataset_signature") == signature and tuple(features.shape) == expected:
            print(f"loaded_multiview_cache={cache_path.resolve()} shape={features.shape}", flush=True)
            return features
        raise ValueError(
            f"Stale V6 feature cache at {cache_path}; remove it or choose another --feature-cache"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    features = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(4, len(rows), int(model.config.projection_dim)),
    )
    dataset = DeterministicMultiViewDataset(rows, processor)
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(batch_size, 32)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model.eval()
    for batch_id, batch in enumerate(loader, 1):
        views = batch["views"]
        batch_rows, num_views = views.shape[:2]
        flattened = views.reshape(batch_rows * num_views, *views.shape[2:]).to(device, non_blocking=True)
        encoded = _feature_tensor(model.get_image_features(pixel_values=flattened))
        encoded = F.normalize(encoded.float(), dim=-1).reshape(batch_rows, num_views, -1)
        indices = batch["row_index"].numpy()
        features[:, indices] = encoded.permute(1, 0, 2).cpu().numpy().astype(np.float16)
        if batch_id % 50 == 0:
            print(f"feature_cache_batches={batch_id}/{len(loader)}", flush=True)
    features.flush()
    del features
    os.replace(temporary, cache_path)
    metadata_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "dataset_signature": signature,
                "views": ["center", "hflip", "light_seed_2", "light_seed_3"],
                "shape": [4, len(rows), int(model.config.projection_dim)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved_multiview_cache={cache_path.resolve()}", flush=True)
    return np.load(cache_path, mmap_mode="r")


def build_classifier(model, prototypes: np.ndarray, device: torch.device) -> RobustCLIPClassifier:
    classifier = RobustCLIPClassifier(
        model,
        classifier_init=torch.as_tensor(prototypes, dtype=torch.float32, device=device),
        bottleneck=128,
        initial_logit_scale=30.0,
        max_logit_scale=50.0,
        learn_logit_scale=False,
    ).to(device)
    replaced = inject_visual_lora(
        classifier.clip,
        last_n_layers=4,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        targets=("q_proj", "v_proj"),
    )
    if len(replaced) != 8:
        raise RuntimeError(f"expected 8 V6 Q/V LoRA modules, got {len(replaced)}")
    print(f"v6_lora_modules={len(replaced)}", flush=True)
    return classifier


def build_optimizer(classifier: RobustCLIPClassifier):
    visual, head = [], []
    for name, parameter in classifier.named_parameters():
        if parameter.requires_grad:
            (visual if name.startswith("clip.") else head).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": visual, "lr": 1e-5},
            {"params": head, "lr": 5e-5},
        ],
        weight_decay=1e-4,
    )
    return optimizer, visual, head


def make_v6_scheduler(optimizer, total_steps: int):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(round(total_steps * 0.05)))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, float(step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def _seed_loader_worker(worker_id: int, epoch: int) -> None:
    worker_seed = (SEED + int(epoch) * 7919 + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def deterministic_loader(
    dataset,
    epoch: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED + int(epoch) * 7919)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        generator=generator,
        worker_init_fn=partial(_seed_loader_worker, epoch=epoch),
    )


def trainable_parameters(classifier: RobustCLIPClassifier) -> list[torch.nn.Parameter]:
    return [parameter for parameter in classifier.parameters() if parameter.requires_grad]


def _autocast_context(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _add_gradients(
    parameters: Sequence[torch.nn.Parameter],
    trusted_grads: Sequence[torch.Tensor | None],
    uncertain_grads: Sequence[torch.Tensor | None],
    regularizer_grads: Sequence[torch.Tensor | None],
    divisor: float,
) -> None:
    for parameter, trusted, uncertain, regularizer in zip(
        parameters, trusted_grads, uncertain_grads, regularizer_grads
    ):
        pieces = [value for value in (trusted, uncertain, regularizer) if value is not None]
        if not pieces:
            continue
        combined = sum(pieces) / float(divisor)
        if parameter.grad is None:
            parameter.grad = combined.detach().clone()
        else:
            parameter.grad.add_(combined.detach())


def _prototype_distribution(
    frozen_features: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = F.normalize(frozen_features.float(), dim=-1) @ F.normalize(prototypes.float(), dim=-1).t()
    return F.softmax(logits / float(temperature), dim=1)


def run_epoch(
    classifier: RobustCLIPClassifier,
    loader: DataLoader,
    optimizer,
    scheduler,
    ema: dict[str, torch.Tensor],
    queue: SelectiveFeatureQueue,
    device: torch.device,
    epoch: int,
    labels: np.ndarray,
    quality: np.ndarray,
    trusted: np.ndarray,
    class_prior: torch.Tensor,
    center_features: np.ndarray,
    prototypes: torch.Tensor,
    repair_plan: np.ndarray,
    temporal: np.ndarray,
    temporal_seen: np.ndarray,
    previous_view_prediction: np.ndarray,
    stability_sum: np.ndarray,
    stability_count: np.ndarray,
    loss_sum: np.ndarray,
    loss_count: np.ndarray,
    optimizer_step: int,
) -> tuple[dict[str, float], int]:
    classifier.train()
    parameters = trainable_parameters(classifier)
    optimizer.zero_grad(set_to_none=True)
    loss_total = correct = rows_seen = repaired_rows = projected_batches = 0.0
    repair_weight = 0.0
    if epoch >= 5:
        repair_weight = min(0.50, 0.50 * (epoch - 4) / max(1, MAX_EPOCHS - 4))
    for batch_id, batch in enumerate(loader, 1):
        pixels_a = batch["pixel_values_a"].to(device, non_blocking=True)
        pixels_b = batch["pixel_values_b"].to(device, non_blocking=True)
        y = batch["label"].to(device)
        indices = batch["row_index"].numpy()
        q = torch.as_tensor(quality[indices], dtype=torch.float32, device=device)
        trusted_batch = torch.as_tensor(trusted[indices], dtype=torch.bool, device=device)
        frozen = torch.as_tensor(np.asarray(center_features[indices]), dtype=torch.float32, device=device)
        with _autocast_context(device):
            logits_a, _, base_a, adapted_a = classifier(pixels_a, None)
            logits_b, _, base_b, adapted_b = classifier(pixels_b, None)
            ce = 0.5 * (
                balanced_softmax_cross_entropy(logits_a, y, class_prior)
                + balanced_softmax_cross_entropy(logits_b, y, class_prior)
            )
            gce = 0.5 * (
                generalized_cross_entropy(logits_a, y, q=0.7)
                + generalized_cross_entropy(logits_b, y, q=0.7)
            )
            uncertain_sample = (1.0 - q) * gce
            repair_batch = torch.as_tensor(repair_plan[indices], dtype=torch.bool, device=device)
            if repair_weight > 0.0 and repair_batch.any():
                proto_prob = _prototype_distribution(frozen, prototypes)
                temporal_prob = torch.as_tensor(
                    np.asarray(temporal[indices]), dtype=torch.float32, device=device
                )
                temporal_prob = temporal_prob / temporal_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
                teacher = torch.sqrt((proto_prob * temporal_prob).clamp_min(1e-12))
                teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
                soft = -0.5 * (
                    (teacher.detach() * F.log_softmax(logits_a.float(), dim=1)).sum(dim=1)
                    + (teacher.detach() * F.log_softmax(logits_b.float(), dim=1)).sum(dim=1)
                )
                repaired = (1.0 - repair_weight) * uncertain_sample + repair_weight * soft
                uncertain_sample = torch.where(repair_batch, repaired, uncertain_sample)
                repaired_rows += float(repair_batch.sum())
            batch_normalizer = max(len(y), 1)
            trusted_loss = (q * ce).sum() / batch_normalizer
            if epoch > WARMUP_EPOCHS:
                contrastive = selective_supervised_contrastive_loss(
                    adapted_a,
                    adapted_b,
                    y,
                    trusted_batch,
                    queue,
                    temperature=0.07,
                )
                trusted_loss = trusted_loss + 0.10 * contrastive
            uncertain_loss = uncertain_sample.sum() / batch_normalizer
            consistency = 0.5 * (1.0 + q) * symmetric_kl_loss(logits_a, logits_b, 2.0)
            loss_consistency = consistency.sum() / (0.5 * (1.0 + q)).sum().clamp_min(1e-6)
            anchor = 0.5 * (
                1.0 - (base_a.float() * F.normalize(frozen, dim=1)).sum(dim=1)
                + 1.0 - (base_b.float() * F.normalize(frozen, dim=1)).sum(dim=1)
            ).mean()
            regularizer = 0.05 * loss_consistency + 0.10 * anchor

        trusted_grads = torch.autograd.grad(
            trusted_loss, parameters, retain_graph=True, allow_unused=True
        )
        uncertain_grads = torch.autograd.grad(
            uncertain_loss, parameters, retain_graph=True, allow_unused=True
        )
        uncertain_grads, was_projected = project_conflicting_gradients(
            trusted_grads, uncertain_grads
        )
        projected_batches += float(was_projected)
        regularizer_grads = torch.autograd.grad(
            regularizer, parameters, retain_graph=False, allow_unused=True
        )
        group_start = ((batch_id - 1) // GRADIENT_ACCUMULATION) * GRADIENT_ACCUMULATION + 1
        group_size = min(GRADIENT_ACCUMULATION, len(loader) - group_start + 1)
        _add_gradients(
            parameters,
            trusted_grads,
            uncertain_grads,
            regularizer_grads,
            group_size,
        )
        should_step = batch_id % GRADIENT_ACCUMULATION == 0 or batch_id == len(loader)
        if should_step:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_step += 1
            update_ema(ema, classifier, 0.999, optimizer_step)

        # The temporal teacher and contrastive queue are both populated from
        # deterministic, bias-corrected EMA parameters, never from the online
        # student or an adapter dropout realization.
        was_training = classifier.training
        classifier.eval()
        try:
            with torch.no_grad(), swapped_trainable(classifier, ema):
                ema_logits_a, _, _, ema_adapted_a = classifier(pixels_a, None)
                ema_logits_b, _, _, ema_adapted_b = classifier(pixels_b, None)
        finally:
            classifier.train(was_training)
        with torch.no_grad():
            average_prob = 0.5 * (
                ema_logits_a.float().softmax(dim=1) + ema_logits_b.float().softmax(dim=1)
            )
            current = average_prob.cpu().numpy()
            seen = temporal_seen[indices]
            if seen.any():
                temporal[indices[seen]] = (
                    0.85 * temporal[indices[seen]].astype(np.float32) + 0.15 * current[seen]
                ).astype(np.float16)
            if (~seen).any():
                temporal[indices[~seen]] = current[~seen].astype(np.float16)
            temporal_seen[indices] = True
            pred_a = ema_logits_a.argmax(dim=1).cpu().numpy()
            pred_b = ema_logits_b.argmax(dim=1).cpu().numpy()
            consensus = np.where(pred_a == pred_b, pred_a, -1)
            previous = previous_view_prediction[indices]
            stability_sum[indices] += np.where(
                previous >= 0,
                (consensus >= 0) & (consensus == previous),
                consensus >= 0,
            ).astype(np.float32)
            stability_count[indices] += 1
            previous_view_prediction[indices] = consensus
            per_sample = (q * ce.detach().float() + uncertain_sample.detach().float()).cpu().numpy()
            loss_sum[indices] += per_sample
            loss_count[indices] += 1
            if epoch > WARMUP_EPOCHS:
                queue_mask = (
                    trusted_batch
                    & torch.as_tensor(consensus, dtype=torch.long, device=device).eq(y)
                    & average_prob.max(dim=1).values.ge(0.80)
                )
                queue.enqueue(0.5 * (ema_adapted_a + ema_adapted_b), y, queue_mask)

        total = trusted_loss.detach().float() + uncertain_loss.detach().float() + regularizer.detach().float()
        loss_total += float(total) * len(y)
        correct += float((logits_a.argmax(dim=1) == y).sum())
        rows_seen += len(y)
    return {
        "loss": loss_total / max(rows_seen, 1.0),
        "noisy_accuracy": correct / max(rows_seen, 1.0),
        "repaired_rows": int(repaired_rows),
        "projected_batches": int(projected_batches),
    }, optimizer_step


@torch.no_grad()
def evaluate(
    classifier: RobustCLIPClassifier,
    loader: DataLoader,
    device: torch.device,
    trusted_global: np.ndarray,
    class_counts: np.ndarray,
    save_logits: Path | None = None,
) -> dict[str, Any]:
    classifier.eval()
    logits_rows, labels_rows, index_rows = [], [], []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with _autocast_context(device):
            logits = classifier(pixels, None)[0]
        logits_rows.append(logits.float().cpu())
        labels_rows.append(batch["label"].cpu())
        index_rows.append(batch["row_index"].cpu())
    logits = torch.cat(logits_rows)
    labels = torch.cat(labels_rows)
    indices = torch.cat(index_rows).numpy()
    trusted = torch.as_tensor(trusted_global[indices], dtype=torch.bool)
    result: dict[str, Any] = {
        "macro_accuracy": macro_accuracy(logits, labels),
        "trusted_macro_accuracy": macro_accuracy(logits, labels, trusted),
        "trusted_macro_nll": macro_nll(logits, labels, trusted),
        "accuracy": float((logits.argmax(dim=1) == labels).float().mean()),
        "trusted_rows": int(trusted.sum()),
        "rows": int(len(labels)),
    }
    group_accuracy = head_mid_tail_accuracy(logits, labels, class_counts)
    result.update({f"{name}_accuracy": value for name, value in group_accuracy.items()})
    result["head_mid_tail_mean_accuracy"] = float(np.mean(list(group_accuracy.values())))
    if save_logits is not None:
        save_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_logits, logits.numpy())
    return result


def trusted_validation_mask(
    labels: np.ndarray,
    val_indices: Sequence[int],
    signals: PrototypeSignals,
) -> np.ndarray:
    indices = np.asarray(val_indices, dtype=np.int64)
    proto_rank = classwise_percentile(labels, signals.label_margin, indices)
    result = np.zeros(len(labels), dtype=np.bool_)
    result[indices] = (
        (signals.prototype_label[indices] == labels[indices])
        & (proto_rank[indices] >= 0.70)
        & (signals.view_agreement[indices] >= 0.75)
    )
    return result


def save_inference_checkpoint(
    path: Path,
    classifier: RobustCLIPClassifier,
    class_names: Sequence[str],
    config: dict[str, Any],
    quality: np.ndarray,
    class_prior: np.ndarray,
    labels: np.ndarray,
    training_indices: Sequence[int],
    selected_epoch: int,
    stage: str,
    validation_indices: Sequence[int],
    trusted_validation: np.ndarray,
    dataset_digest: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = np.asarray(training_indices, dtype=np.int64)
    class_quality = np.bincount(
        labels[selected], weights=quality[selected], minlength=len(class_names)
    ).astype(np.float32)
    atomic_torch_save(
        {
            "format_version": FORMAT_VERSION,
            "kind": "robust_clip_v6",
            "model": trainable_state_dict(classifier),
            "class_names": list(class_names),
            "config": config,
            "quality_summary": {
                "n": int(len(selected)),
                "mean": float(quality[selected].mean()),
                "p10": float(np.quantile(quality[selected], 0.10)),
                "p50": float(np.quantile(quality[selected], 0.50)),
                "p90": float(np.quantile(quality[selected], 0.90)),
            },
            "reliable_class_quality": torch.from_numpy(class_quality),
            "clean_class_prior": torch.from_numpy(class_prior),
            "selected_epoch": int(selected_epoch),
            "training_stage": stage,
            "dataset_signature": dataset_digest,
            "reliability": {
                "signals": [
                    "cross_fitted_label_margin",
                    "multi_view_agreement",
                    "warmup_loss_rank",
                    "temporal_prediction_stability",
                ],
                "exponents": [0.40, 0.20, 0.25, 0.15],
                "trusted_threshold": 0.70,
            },
            "validation": {
                "indices": [int(value) for value in validation_indices],
                "trusted_indices": np.where(trusted_validation)[0].astype(np.int64).tolist(),
            },
        },
        path,
    )


def save_resume(
    path: Path,
    *,
    stage: str,
    epoch: int,
    classifier: RobustCLIPClassifier,
    optimizer,
    scheduler,
    ema: dict[str, torch.Tensor],
    queue: SelectiveFeatureQueue,
    quality: np.ndarray,
    trusted: np.ndarray,
    class_prior: np.ndarray,
    temporal: np.ndarray,
    temporal_seen: np.ndarray,
    previous_view_prediction: np.ndarray,
    stability_sum: np.ndarray,
    stability_count: np.ndarray,
    loss_sum: np.ndarray,
    loss_count: np.ndarray,
    optimizer_step: int,
    history: list[dict[str, Any]],
    best_state: dict[str, torch.Tensor],
    best_key: tuple[float, float, float],
    best_epoch: int,
    selected_epoch: int | None,
    config: dict[str, Any],
    dataset_digest: str,
) -> None:
    payload = {
        "kind": "train_v6_resume",
        "resume_version": RESUME_VERSION,
        "stage": stage,
        "epoch": int(epoch),
        "model": trainable_state_dict(classifier),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": {name: value.detach().cpu() for name, value in ema.items()},
        "queue": queue.state_dict(),
        "quality": quality,
        "trusted": trusted,
        "class_prior": class_prior,
        "temporal": temporal,
        "temporal_seen": temporal_seen,
        "previous_view_prediction": previous_view_prediction,
        "stability_sum": stability_sum,
        "stability_count": stability_count,
        "loss_sum": loss_sum,
        "loss_count": loss_count,
        "optimizer_step": int(optimizer_step),
        "history": history,
        "best_state": {name: value.detach().cpu() for name, value in best_state.items()},
        "best_key": tuple(float(value) for value in best_key),
        "best_epoch": int(best_epoch),
        "selected_epoch": selected_epoch,
        "config": config,
        "dataset_signature": dataset_digest,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    atomic_torch_save(payload, path)


def load_resume(path: str | None, config: dict[str, Any], dataset_digest: str, device: torch.device):
    if path is None:
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "train_v6_resume" or payload.get("resume_version") != RESUME_VERSION:
        raise ValueError("unsupported V6 resume checkpoint")
    if payload.get("config") != config:
        raise ValueError("resume configuration does not match this run")
    if payload.get("dataset_signature") != dataset_digest:
        raise ValueError("resume dataset does not match this run")
    if payload.get("stage") not in {"validation", "final"}:
        raise ValueError("invalid V6 resume stage")
    return payload


def restore_rng(payload: dict[str, Any]) -> None:
    state = payload["rng"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def fit(
    *,
    classifier: RobustCLIPClassifier,
    processor,
    rows,
    train_indices: Sequence[int],
    labels: np.ndarray,
    view_features: np.ndarray,
    signals: PrototypeSignals,
    quality: np.ndarray,
    validation_loader: DataLoader | None,
    trusted_validation: np.ndarray,
    class_counts: np.ndarray,
    class_names: Sequence[str],
    config: dict[str, Any],
    dataset_digest: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    epochs: int,
    stage: str,
    validation_indices: Sequence[int],
    resume: dict[str, Any] | None,
) -> tuple[dict[str, torch.Tensor], int, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    dataset = PairedFolderImageDataset(rows, train_indices, processor, augment="light")
    optimizer, visual, head = build_optimizer(classifier)
    steps_per_epoch = math.ceil(len(dataset) / batch_size / GRADIENT_ACCUMULATION)
    scheduler = make_v6_scheduler(optimizer, epochs * steps_per_epoch)
    dim = int(classifier.classifier.shape[1])
    queue = SelectiveFeatureQueue(len(class_names), QUEUE_PER_CLASS, dim, device)
    center_features = view_features[0]
    prototypes = torch.as_tensor(signals.prototypes, dtype=torch.float32, device=device)
    trusted = (quality >= 0.70) & (signals.prototype_label == labels)
    class_prior_np = reliable_class_prior(labels, quality, train_indices, len(class_names))
    class_prior = torch.as_tensor(class_prior_np, dtype=torch.float32, device=device)
    temporal = np.zeros((len(rows), len(class_names)), dtype=np.float16)
    temporal_seen = np.zeros(len(rows), dtype=np.bool_)
    previous_view_prediction = np.full(len(rows), -1, dtype=np.int64)
    stability_sum = np.zeros(len(rows), dtype=np.float32)
    stability_count = np.zeros(len(rows), dtype=np.int16)
    loss_sum = np.zeros(len(rows), dtype=np.float32)
    loss_count = np.zeros(len(rows), dtype=np.int16)
    ema = clone_trainable(classifier)
    best_state = clone_trainable(classifier)
    best_key = (-1.0, -1.0, -float("inf"))
    best_epoch = 0
    optimizer_step = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if resume is not None:
        load_classifier_state(classifier, resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        ema = {name: value.to(device).clone() for name, value in resume["ema"].items()}
        queue.load_state_dict(resume["queue"])
        quality = np.asarray(resume["quality"], dtype=np.float32).copy()
        trusted = np.asarray(resume["trusted"], dtype=np.bool_).copy()
        class_prior_np = np.asarray(resume["class_prior"], dtype=np.float32).copy()
        class_prior = torch.as_tensor(class_prior_np, dtype=torch.float32, device=device)
        temporal = np.asarray(resume["temporal"], dtype=np.float16).copy()
        temporal_seen = np.asarray(resume["temporal_seen"], dtype=np.bool_).copy()
        previous_view_prediction = np.asarray(resume["previous_view_prediction"], dtype=np.int64).copy()
        stability_sum = np.asarray(resume["stability_sum"], dtype=np.float32).copy()
        stability_count = np.asarray(resume["stability_count"], dtype=np.int16).copy()
        loss_sum = np.asarray(resume["loss_sum"], dtype=np.float32).copy()
        loss_count = np.asarray(resume["loss_count"], dtype=np.int16).copy()
        optimizer_step = int(resume["optimizer_step"])
        history = list(resume["history"])
        best_state = {name: value.to(device).clone() for name, value in resume["best_state"].items()}
        best_key = tuple(float(value) for value in resume["best_key"])
        best_epoch = int(resume["best_epoch"])
        start_epoch = int(resume["epoch"]) + 1
        restore_rng(resume)
        print(f"resumed_v6 stage={stage} next_epoch={start_epoch}", flush=True)

    margin_thresholds = prototype_margin_thresholds(
        labels, signals.prototype_margin, train_indices, 0.75
    )
    print(
        f"v6_stage={stage} rows={len(train_indices)} visual_params={sum(p.numel() for p in visual):,} "
        f"head_params={sum(p.numel() for p in head):,} initial_trusted={int(trusted[np.asarray(train_indices)].sum())}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs + 1):
        if epoch == WARMUP_EPOCHS + 1 and not (resume is not None and start_epoch > WARMUP_EPOCHS + 1):
            warmup_loss = loss_sum / np.maximum(loss_count, 1)
            temporal_agreement = stability_sum / np.maximum(stability_count, 1)
            quality = fused_reliability(
                labels,
                train_indices,
                signals.label_margin,
                signals.view_agreement,
                warmup_loss,
                temporal_agreement,
            )
            trusted = (quality >= 0.70) & (signals.prototype_label == labels)
            class_prior_np = reliable_class_prior(labels, quality, train_indices, len(class_names))
            class_prior = torch.as_tensor(class_prior_np, dtype=torch.float32, device=device)
            print(
                f"reliability_fused mean={quality[np.asarray(train_indices)].mean():.4f} "
                f"trusted={int(trusted[np.asarray(train_indices)].sum())}",
                flush=True,
            )

        repair_plan = np.zeros(len(rows), dtype=np.bool_)
        if epoch >= 5:
            repair_plan = build_repair_plan(
                labels,
                train_indices,
                quality,
                signals.prototype_label,
                signals.prototype_margin,
                margin_thresholds,
                temporal,
                previous_view_prediction,
                confidence_threshold=0.80,
                per_class_fraction=0.15,
            )
        loader = deterministic_loader(dataset, epoch, batch_size, workers, device)
        stats, optimizer_step = run_epoch(
            classifier,
            loader,
            optimizer,
            scheduler,
            ema,
            queue,
            device,
            epoch,
            labels,
            quality,
            trusted,
            class_prior,
            center_features,
            prototypes,
            repair_plan,
            temporal,
            temporal_seen,
            previous_view_prediction,
            stability_sum,
            stability_count,
            loss_sum,
            loss_count,
            optimizer_step,
        )
        record: dict[str, Any] = {"epoch": epoch, **stats}
        if validation_loader is not None:
            with swapped_trainable(classifier, ema):
                metrics = evaluate(
                    classifier,
                    validation_loader,
                    device,
                    trusted_validation,
                    class_counts,
                )
            record.update(metrics)
            key = (
                float(metrics["trusted_macro_accuracy"]),
                float(metrics["macro_accuracy"]),
                -float(metrics["trusted_macro_nll"]),
            )
            if epoch > WARMUP_EPOCHS and key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = {name: value.detach().clone() for name, value in ema.items()}
                with swapped_trainable(classifier, best_state):
                    save_inference_checkpoint(
                        output_dir / "best_model.pt",
                        classifier,
                        class_names,
                        config,
                        quality,
                        class_prior_np,
                        labels,
                        train_indices,
                        best_epoch,
                        "validation",
                        validation_indices,
                        trusted_validation,
                        dataset_digest,
                    )
            print(
                f"epoch={epoch}/{epochs} loss={stats['loss']:.5f} noisy_acc={stats['noisy_accuracy']:.4f} "
                f"trusted_macro={metrics['trusted_macro_accuracy']:.4f} macro={metrics['macro_accuracy']:.4f} "
                f"tail={metrics['tail_accuracy']:.4f} repairs={stats['repaired_rows']} projections={stats['projected_batches']}",
                flush=True,
            )
        else:
            best_epoch = epochs
            best_state = {name: value.detach().clone() for name, value in ema.items()}
            print(
                f"final_epoch={epoch}/{epochs} loss={stats['loss']:.5f} noisy_acc={stats['noisy_accuracy']:.4f} "
                f"repairs={stats['repaired_rows']} projections={stats['projected_batches']}",
                flush=True,
            )
        history.append(record)
        save_resume(
            output_dir / "resume_latest.pt",
            stage=stage,
            epoch=epoch,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            queue=queue,
            quality=quality,
            trusted=trusted,
            class_prior=class_prior_np,
            temporal=temporal,
            temporal_seen=temporal_seen,
            previous_view_prediction=previous_view_prediction,
            stability_sum=stability_sum,
            stability_count=stability_count,
            loss_sum=loss_sum,
            loss_count=loss_count,
            optimizer_step=optimizer_step,
            history=history,
            best_state=best_state,
            best_key=best_key,
            best_epoch=best_epoch,
            selected_epoch=epochs if stage == "final" else None,
            config=config,
            dataset_digest=dataset_digest,
        )

    if validation_loader is not None and best_epoch == 0:
        raise RuntimeError("V6 validation did not produce a selectable checkpoint")
    return best_state, best_epoch, quality, class_prior_np, history


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_determinism(SEED)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list_images(args.train_dir)
    labels = np.asarray([label for _, label, _ in rows], dtype=np.int64)
    class_names = [name for _, name in sorted({(label, name) for _, label, name in rows})]
    if len(class_names) != int(labels.max()) + 1:
        raise ValueError("training class indices are not contiguous")
    digest = dataset_signature(rows)
    config = checkpoint_config(args)
    resume = load_resume(args.resume, config, digest, device)

    model, processor = load_clip(args.model_dir, device)
    validate_clip_vit_b32(model)
    view_features = load_or_create_multiview_features(
        model,
        processor,
        rows,
        Path(args.feature_cache),
        digest,
        device,
        args.batch_size,
        args.workers,
    )
    train_indices, validation_indices = capped_stratified_split(rows, SEED)
    if len(validation_indices) < 5:
        raise ValueError("V6 requires at least five non-singleton rows for validation and calibration")
    validation_signals_source = cross_fitted_visual_signals(
        view_features,
        labels,
        train_indices,
        len(class_names),
        folds=5,
        seed=SEED,
        keep_fraction=0.70,
        iterations=2,
    )
    validation_signals = fixed_prototype_signals(
        view_features,
        labels,
        validation_indices,
        validation_signals_source.prototypes,
    )
    trusted_validation = trusted_validation_mask(labels, validation_indices, validation_signals)
    validation_quality = initial_reliability(
        labels,
        train_indices,
        validation_signals_source.label_margin,
        validation_signals_source.view_agreement,
    )
    class_counts = np.bincount(labels[np.asarray(train_indices)], minlength=len(class_names))
    validation_loader = DataLoader(
        FolderImageDataset(rows, validation_indices, processor, augment="none"),
        batch_size=max(args.batch_size, 64),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    resume_stage = None if resume is None else resume["stage"]
    validation_history: list[dict[str, Any]] = []
    if resume_stage != "final":
        configure_determinism(SEED + 101)
        classifier = build_classifier(model, validation_signals_source.prototypes, device)
        best_state, selected_epoch, _, _, validation_history = fit(
            classifier=classifier,
            processor=processor,
            rows=rows,
            train_indices=train_indices,
            labels=labels,
            view_features=view_features,
            signals=validation_signals_source,
            quality=validation_quality,
            validation_loader=validation_loader,
            trusted_validation=trusted_validation,
            class_counts=class_counts,
            class_names=class_names,
            config=config,
            dataset_digest=digest,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            epochs=MAX_EPOCHS,
            stage="validation",
            validation_indices=validation_indices,
            resume=resume,
        )
        with torch.no_grad():
            for name, parameter in classifier.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(best_state[name])
        validation_selected = evaluate(
            classifier,
            validation_loader,
            device,
            trusted_validation,
            class_counts,
            output_dir / "val_logits.npy",
        )
        (output_dir / "validation_metrics.json").write_text(
            json.dumps(
                {
                    "selected_epoch": selected_epoch,
                    "selected": validation_selected,
                    "history": validation_history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        del classifier
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        selected_epoch = int(resume["selected_epoch"])
        validation_metrics_path = output_dir / "validation_metrics.json"
        if validation_metrics_path.is_file():
            validation_payload = json.loads(validation_metrics_path.read_text(encoding="utf-8"))
            validation_selected = validation_payload.get("selected", {})
            validation_history = validation_payload.get("history", [])
        else:
            validation_selected = {}
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Rebuild from the official CLIP initialization.  This is a fresh full-data
    # training run, not a continuation of the validation checkpoint.
    final_signals = cross_fitted_visual_signals(
        view_features,
        labels,
        list(range(len(rows))),
        len(class_names),
        folds=5,
        seed=SEED,
        keep_fraction=0.70,
        iterations=2,
    )
    final_quality = initial_reliability(
        labels,
        list(range(len(rows))),
        final_signals.label_margin,
        final_signals.view_agreement,
    )
    configure_determinism(SEED + 202)
    final_model, final_processor = load_clip(args.model_dir, device)
    validate_clip_vit_b32(final_model)
    final_classifier = build_classifier(final_model, final_signals.prototypes, device)
    final_resume = resume if resume_stage == "final" else None
    final_state, _, final_quality, final_prior, final_history = fit(
        classifier=final_classifier,
        processor=final_processor,
        rows=rows,
        train_indices=list(range(len(rows))),
        labels=labels,
        view_features=view_features,
        signals=final_signals,
        quality=final_quality,
        validation_loader=None,
        trusted_validation=np.zeros(len(rows), dtype=np.bool_),
        class_counts=np.bincount(labels, minlength=len(class_names)),
        class_names=class_names,
        config=config,
        dataset_digest=digest,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=selected_epoch,
        stage="final",
        validation_indices=validation_indices,
        resume=final_resume,
    )
    with torch.no_grad():
        for name, parameter in final_classifier.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(final_state[name])
    save_inference_checkpoint(
        output_dir / "model.pt",
        final_classifier,
        class_names,
        config,
        final_quality,
        final_prior,
        labels,
        list(range(len(rows))),
        selected_epoch,
        "final",
        validation_indices,
        trusted_validation,
        digest,
    )
    np.savez_compressed(
        output_dir / "sample_reliability.npz",
        quality=final_quality,
        trusted=(final_quality >= 0.70) & (final_signals.prototype_label == labels),
        label_margin=final_signals.label_margin,
        prototype_margin=final_signals.prototype_margin,
        prototype_label=final_signals.prototype_label,
        prototype_confidence=final_signals.prototype_confidence,
        view_agreement=final_signals.view_agreement,
        fold_ids=final_signals.fold_ids,
        class_prior=final_prior,
        dataset_signature=np.asarray(digest),
    )
    metrics = {
        "format_version": FORMAT_VERSION,
        "selected_epoch": selected_epoch,
        "validation_selected": validation_selected,
        "validation_history": validation_history,
        "final_history": final_history,
        "trusted_validation_rows": int(trusted_validation.sum()),
        "train_rows": len(rows),
        "classes": len(class_names),
        "dataset_signature": digest,
        "config": config,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"v6_complete selected_epoch={selected_epoch} checkpoint={output_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
