"""Train V20 attention/MLP LoRA with V14 views and frozen-contract refits."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from v20_runtime import (pin_memory_enabled, scientific_config, autocast_context as _autocast_context, BatchedFeatureQueue as SelectiveFeatureQueue,
                         loader_options, synchronized_time, Progress, precision_name, pipeline_eta, training_history_seconds)
from v20_core import (RECIPES, recipe_config, validate_recipe_config, build_optimizer, model_identity, file_sha256,
                      pool_signature, stage_indices, should_audit, dynamic_repair_plan,
                      repaired_soft_loss, repair_weight as dynamic_repair_weight, repair_diagnostics, optimizer_group_metadata)
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from robust_clip import (
    FolderImageDataset,
    RobustCLIPClassifier,
    load_clip,
    resolve_device,
    symmetric_kl_loss,
    trainable_state_dict,
)
from train_v5 import atomic_torch_save, clone_trainable, swapped_trainable, update_ema
from train_v6 import (
    DeterministicMultiViewDataset,
    _add_gradients,
    _prototype_distribution,
    configure_determinism,
    load_or_create_multiview_features,
    make_v6_scheduler,
    trainable_parameters,
    trusted_validation_mask,
    validate_clip_vit_b32,
)
from v6_core import (
    PrototypeSignals,
    balanced_softmax_cross_entropy,
    build_repair_plan,
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
from v7_data import ManifestDataset, effective_repeat_factors, load_manifest_dataset, read_manifest
from v20_views import training_dataset
from v20_model import (
    HighResolutionPairedDataset, build_classifier, resolution_processor,
    load_classifier_state, trainable_parameter_names, validate_trainable_metadata, validate_trainable_state,
)
from v8_neighbors import (
    NeighborEvidence,
    blend_quality,
    load_or_create_neighbor_evidence,
    trusted_mask,
)


FORMAT_VERSION = 20
RESUME_VERSION = 1
SEED = 2026
WARMUP_EPOCHS = 2
MAX_EPOCHS = 24
GRADIENT_ACCUMULATION = 1
QUEUE_PER_CLASS = 16
PARTIAL_WEIGHT = 0.25
PARTIAL_PROTOTYPE_MIX = 0.75
PARTIAL_TEMPERATURE = 0.07
PARTIAL_MAX_PROBABILITY = 0.80
QUALITY_FLOOR = 0.25
SOFT_TEACHER_CUTOFF = 0.55
SOFT_TEACHER_WEIGHT = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--conflict-policy", choices=["drop", "partial"], default="partial")
    parser.add_argument("--sampler", choices=["shuffle", "repeat-factor"], default="repeat-factor")
    parser.add_argument("--model-dir", default="clip-ViT-B-32")
    parser.add_argument("--output-dir", default="outputs/v20_dora_rank32")
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--neighbor-cache", default=None)
    parser.add_argument("--recipe", choices=RECIPES, default="dora_rank32")
    parser.add_argument("--stage", choices=["validate", "refit"], default="validate")
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--resource-json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.workers < 0 or args.gradient_accumulation < 1:
        raise ValueError("batch-size and gradient-accumulation must be positive; workers non-negative")
    if (isinstance(args.batch_size, bool) or isinstance(args.gradient_accumulation, bool)
            or (args.batch_size, args.gradient_accumulation) != (256, 1)):
        raise ValueError("V20 requires actual microbatch256x1; no smaller-batch fallback")
    if args.recipe not in RECIPES:
        raise ValueError("V20 only trains the frozen V20 recipes and deduplicated control")
    loader_options(args.workers, args.prefetch_factor)
    if args.eval_batch_size < 1:
        raise ValueError("eval-batch-size must be positive")
    if (args.stage == "refit") != bool(args.selection_json):
        raise ValueError("refit requires --selection-json; validation must not use it")
    if args.stage == "refit" and args.neighbor_cache:
        raise ValueError("refit must rebuild neighbors in its own output directory")
    if Path(args.train_dir).resolve() == Path(args.output_dir).resolve():
        raise ValueError("output-dir must not be the training directory")
    if args.conflict_policy != "partial" or args.sampler != "repeat-factor":
        raise ValueError("V20 preserves V15 partial conflicts and repeat-factor sampling")


def bind_resource_plan(args, selection=None):
    """Validate the preflight decision before loading images or initializing CUDA."""
    from v20_pipeline_support import read_resource_plan, resource_for_recipe

    plan = read_resource_plan(args.resource_json)
    entry = resource_for_recipe(plan, args.recipe)
    if entry.get("status") != "runnable":
        raise ValueError("V20 candidate is resource_infeasible and cannot be trained")
    expected = recipe_config(args.recipe)
    for key in ("image_size", "zoom_shortest_edge", "lora_rank", "lora_alpha",
                "lora_dropout", "mlp_lora_dropout"):
        if entry.get(key) != expected[key]:
            raise ValueError(f"V20 frozen resource plan conflicts with {key}")
    for key in ("batch_size", "gradient_accumulation", "workers"):
        if getattr(args, key) != entry[key]:
            raise ValueError(f"V20 {key} differs from the frozen resource plan")
    if args.stage == "validate" and Path(args.output_dir).resolve() != Path(entry["run_dir"]).resolve():
        raise ValueError("V20 validation output differs from the frozen resource plan")
    if selection is not None:
        selected_config = selection["training_config"]
        if selected_config.get("resource_sha256") != file_sha256(args.resource_json):
            raise ValueError("refit resource plan differs from the selected validation run")
        for key in ("batch_size", "gradient_accumulation", "workers"):
            if getattr(args, key) != selected_config.get(key):
                raise ValueError(f"refit must preserve selected {key}")
    args.resource_plan = plan
    return plan


def validate_neighbor_evidence(evidence: NeighborEvidence, rows: int) -> None:
    for field in ("label_support", "top_label", "top_support", "percentile", "fold_ids"):
        value = np.asarray(getattr(evidence, field))
        if value.shape != (rows,):
            raise ValueError(f"neighbor cache {field} has the wrong shape")
        if not np.isfinite(value).all():
            raise ValueError(f"neighbor cache {field} contains nonfinite values")


def save_epoch_logits(
    epoch_dir: Path,
    epoch: int,
    original: np.ndarray,
    hflip: np.ndarray,
    labels: np.ndarray,
    validation_indices: Sequence[int],
    precision: str = "fp32",
    *, config: dict[str, Any],
) -> None:
    """Keep native validation evidence for outer-fold epoch selection."""
    indices = np.asarray(validation_indices, dtype=np.int64)
    if original.shape != hflip.shape or original.shape[0] != len(labels) or len(labels) != len(indices):
        raise ValueError("per-epoch validation logits do not match validation rows")
    destination = epoch_dir / f"epoch_{epoch:02d}_logits.npz"
    temporary = epoch_dir / f".epoch_{epoch:02d}_logits.tmp.npz"
    np.savez_compressed(
        temporary,
        center=np.asarray(original, dtype=np.float32),
        hflip=np.asarray(hflip, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        row_indices=indices,
        precision=np.asarray(precision),
        format_version=np.asarray(FORMAT_VERSION),
        image_size=np.asarray(config["image_size"]),
        zoom_shortest_edge=np.asarray(config["zoom_shortest_edge"]),
        lora_rank=np.asarray(config["lora_rank"]),
        lora_dropout=np.asarray(config["lora_dropout"]),
        mlp_lora_dropout=np.asarray(config["mlp_lora_dropout"]),
        checkpoint_sha256=np.asarray(hashlib.sha256((epoch_dir / f"epoch_{epoch:02d}.pt").read_bytes()).hexdigest()),
    )
    temporary.replace(destination)


def checkpoint_config(args: argparse.Namespace, manifest: ManifestDataset) -> dict[str, Any]:
    config = {
        "version": "v20",
        "model_dir": args.model_dir,
        "train_dir": args.train_dir,
        "data_manifest": str(Path(args.data_manifest).resolve()),
        "manifest_signature": manifest.signature,
        "conflict_policy": args.conflict_policy,
        "sampler": args.sampler,
        "output_dir": args.output_dir,
        "seed": SEED,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": pin_memory_enabled(resolve_device(args.device)),
        "eval_batch_size": args.eval_batch_size,
        "precision": precision_name(resolve_device(args.device)),
        "loss_precision": "fp32",
        "epochs": MAX_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "gradient_accumulation": args.gradient_accumulation,
        "augmentation": "light_320_both_views",
        "image_size": recipe_config(args.recipe)["image_size"],
        "zoom_shortest_edge": recipe_config(args.recipe)["zoom_shortest_edge"],
        "interpolate_pos_encoding": True,
        "frozen_feature_image_size": 224,
        "bottleneck": 128,
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
        "validation_split": "none_refit_all_manifest_rows" if args.stage == "refit" else "manifest_clean_tail_safe_capped_8_percent",
        "fixed_feature_views": ["center", "hflip", "light_seed_2", "light_seed_3"],
        "partial_weight": PARTIAL_WEIGHT,
        "partial_prototype_mix": PARTIAL_PROTOTYPE_MIX,
        "partial_temperature": PARTIAL_TEMPERATURE,
        "partial_max_probability": PARTIAL_MAX_PROBABILITY,
        "repeat_factor": "min(4,sqrt(median_count/class_count))",
        "epoch_selection": "frozen_selection" if args.stage == "refit" else "joint_epoch_tta_bias_nested_cv",
        "neighbor_method": "four_view_cosine_32_crossfit_5fold",
        "neighbor_quality_mix": [0.75, 0.25],
        "neighbor_support_threshold": 0.50,
        "noise_audit": "ema_center_hflip_all_train_after_epochs_1_2",
        "quality_floor": QUALITY_FLOOR,
        "soft_teacher": bool(args.soft_teacher),
        "v9_base_run": "v9_soft_teacher",
        "soft_teacher_cutoff": SOFT_TEACHER_CUTOFF,
        "soft_teacher_weight": SOFT_TEACHER_WEIGHT,
        **recipe_config(args.recipe),
        "stage": args.stage,
        "base_model_identity": model_identity(args.model_dir),
        "selection_sha256": file_sha256(args.selection_json) if args.selection_json else None,
        "resource_json": str(Path(args.resource_json).resolve()),
        "resource_sha256": file_sha256(args.resource_json),
        "resource_branch": args.resource_plan["branch"],
        "audit_json": str(Path(args.resource_plan["audit"]["path"]).resolve()),
        "audit_sha256": args.resource_plan["audit"]["sha256"],
    }
    validate_recipe_config(config)
    scientific_config(config)
    return config


def partial_label_targets(
    frozen_features: torch.Tensor,
    prototypes: torch.Tensor,
    candidate_labels: Sequence[Sequence[int]],
    temperature: float = PARTIAL_TEMPERATURE,
    prototype_mix: float = PARTIAL_PROTOTYPE_MIX,
    maximum: float = PARTIAL_MAX_PROBABILITY,
) -> torch.Tensor:
    """Build capped prototype targets supported only on each candidate set."""
    if len(candidate_labels) != len(frozen_features):
        raise ValueError("candidate label rows do not match feature rows")
    class_count = int(prototypes.shape[0])
    result = torch.zeros((len(candidate_labels), class_count), device=frozen_features.device)
    scores = F.normalize(frozen_features.float(), dim=1) @ F.normalize(prototypes.float(), dim=1).t()
    for row, candidates_value in enumerate(candidate_labels):
        candidates = tuple(sorted({int(value) for value in candidates_value}))
        if len(candidates) < 2:
            raise ValueError("partial-label rows require at least two candidate labels")
        ids = torch.as_tensor(candidates, dtype=torch.long, device=frozen_features.device)
        proto = F.softmax(scores[row, ids] / float(temperature), dim=0)
        uniform = torch.full_like(proto, 1.0 / len(candidates))
        beta = float(prototype_mix)
        maximum_proto = float(proto.max())
        uniform_value = 1.0 / len(candidates)
        if maximum_proto > uniform_value:
            beta = min(beta, (float(maximum) - uniform_value) / (maximum_proto - uniform_value))
        target = uniform + max(0.0, beta) * (proto - uniform)
        target = target / target.sum().clamp_min(1e-8)
        result[row, ids] = target
    return result


def partial_label_loss(
    logits_a: torch.Tensor, logits_b: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    if logits_a.shape != logits_b.shape or logits_a.shape != targets.shape:
        raise ValueError("partial-label logits and targets must have identical shapes")
    return -0.5 * (
        (targets * F.log_softmax(logits_a.float(), dim=1)).sum(dim=1)
        + (targets * F.log_softmax(logits_b.float(), dim=1)).sum(dim=1)
    )


def _seed_loader_worker(worker_id: int, epoch: int) -> None:
    worker_seed = (SEED + int(epoch) * 7919 + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def deterministic_loader(
    dataset: HighResolutionPairedDataset,
    epoch: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    sampler_mode: str,
    row_repeat_factors: np.ndarray,
    prefetch_factor: int = 2,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED + int(epoch) * 7919)
    sampler = None
    shuffle = sampler_mode == "shuffle"
    if sampler_mode == "repeat-factor":
        weights = torch.as_tensor(
            [float(row_repeat_factors[index]) for index in dataset.indices], dtype=torch.double
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        **loader_options(workers, prefetch_factor),
        pin_memory=pin_memory_enabled(device),
        persistent_workers=False,
        generator=generator if sampler is None else None,
        worker_init_fn=partial(_seed_loader_worker, epoch=epoch),
    )


@torch.no_grad()
def audit_ema_coverage(
    classifier: RobustCLIPClassifier,
    ema: dict[str, torch.Tensor],
    processor,
    rows,
    supervised_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    workers: int,
    first_prob: np.ndarray | None = None,
    prefetch_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score each selected training row once with deterministic EMA views."""
    selected = np.asarray(supervised_indices, dtype=np.int64)
    if len(selected) == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError("EMA audit requires unique nonempty supervised indices")
    class_count = int(classifier.classifier.shape[0])
    shape = (len(rows), class_count)
    if first_prob is not None and first_prob.shape != shape:
        raise ValueError("first EMA audit has the wrong shape")
    probability = np.zeros(shape, dtype=np.float16)
    nll = np.zeros(len(rows), dtype=np.float32)
    stability = np.zeros(len(rows), dtype=np.float32)
    seen = np.zeros(len(rows), dtype=np.bool_)
    consensus = np.full(len(rows), -1, dtype=np.int64)
    loader = DataLoader(
        FolderImageDataset(rows, selected.tolist(), processor, augment="none"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        **loader_options(workers, prefetch_factor),
        pin_memory=pin_memory_enabled(device),
        persistent_workers=False,
    )
    was_training = classifier.training
    classifier.eval()
    try:
        with swapped_trainable(classifier, ema):
            progress = Progress("ema_audit", device, len(loader))
            for batch_id, batch in enumerate(loader, 1):
                indices = batch["row_index"].numpy()
                if seen[indices].any():
                    raise RuntimeError("EMA audit visited a training row twice")
                pixels = batch["pixel_values"].to(device, non_blocking=True)
                with _autocast_context(device):
                    center = classifier(pixels, None)[0]
                    hflip = classifier(torch.flip(pixels, dims=[3]), None)[0]
                current = 0.5 * (
                    center.float().softmax(dim=1) + hflip.float().softmax(dim=1)
                )
                labels_batch = batch["label"].to(device, non_blocking=True)
                nll[indices] = -current.gather(1, labels_batch[:, None]).squeeze(1).clamp_min(1e-8).log().cpu().numpy()
                if first_prob is not None:
                    previous = torch.as_tensor(np.asarray(first_prob[indices]), device=device).float()
                    previous = previous / previous.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    middle = 0.5 * (previous + current)
                    js = 0.5 * (
                        (previous * (previous.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(dim=1)
                        + (current * (current.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(dim=1)
                    )
                    stability[indices] = (1.0 - js / math.log(2.0)).clamp(0.0, 1.0).cpu().numpy()
                ca, cb = center.argmax(1), hflip.argmax(1)
                consensus[indices] = torch.where(ca == cb, ca, -1).cpu().numpy()
                probability[indices] = current.cpu().numpy().astype(np.float16)
                seen[indices] = True
                progress.update(batch_id, len(indices))
    finally:
        classifier.train(was_training)
    if not seen[selected].all() or int(seen.sum()) != len(selected):
        raise RuntimeError("EMA audit did not cover every supervised training row")
    if not np.isfinite(nll[selected]).all() or not np.isfinite(stability[selected]).all():
        raise RuntimeError("EMA audit produced nonfinite scores")
    return probability, nll, stability, seen, consensus


def _class_prior(
    labels: np.ndarray,
    quality: np.ndarray,
    supervised_indices: Sequence[int],
    row_repeat_factors: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    effective_quality = np.asarray(quality, dtype=np.float64) * np.asarray(
        row_repeat_factors, dtype=np.float64
    )
    return reliable_class_prior(labels, effective_quality, supervised_indices, num_classes)


def apply_clean_quality_floor(quality: np.ndarray, supervised_indices: Sequence[int], floor: float = QUALITY_FLOOR) -> np.ndarray:
    result = np.asarray(quality, dtype=np.float32).copy()
    selected = np.asarray(supervised_indices, dtype=np.int64)
    result[selected] = np.maximum(result[selected], floor)
    return result


def soft_teacher_regularizer(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    quality: torch.Tensor,
    clean: torch.Tensor,
    historical_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Use prior EMA predictions as soft targets without rewriting labels."""
    selected = clean & (quality < SOFT_TEACHER_CUTOFF)
    if not selected.any():
        return logits_a.sum() * 0.0, 0
    teacher = historical_probabilities[selected].float().detach()
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    per_view = 0.5 * (
        F.kl_div(F.log_softmax(logits_a[selected].float(), dim=1), teacher, reduction="none").sum(dim=1)
        + F.kl_div(F.log_softmax(logits_b[selected].float(), dim=1), teacher, reduction="none").sum(dim=1)
    )
    weights = SOFT_TEACHER_WEIGHT * (1.0 - quality[selected]) * teacher.max(dim=1).values
    return (weights * per_view).sum() / max(len(quality), 1), selected.sum().detach()


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
    ambiguous_mask: np.ndarray,
    candidate_labels: Sequence[Sequence[int]],
    gradient_accumulation: int,
    soft_teacher: bool,
    dynamic_noise: bool = False,
    audit_probabilities: np.ndarray | None = None,
    schedule_epochs: int = MAX_EPOCHS,
) -> tuple[dict[str, float], int]:
    if gradient_accumulation != 1:
        raise ValueError("V20 student updates require gradient_accumulation=1")
    classifier.train()
    parameters = trainable_parameters(classifier)
    optimizer.zero_grad(set_to_none=True)
    counters = torch.zeros(6, device=device, dtype=torch.float64)
    all_rows_seen = projected_batches = 0
    progress = Progress(f"train_epoch_{epoch}", device, len(loader))
    repair_weight = 0.0
    if epoch >= 5:
        repair_weight = min(0.50, 0.50 * (epoch - 4) / max(1, schedule_epochs - 4))
    for batch_id, batch in enumerate(loader, 1):
        pixels_a = batch["pixel_values_a"].to(device, non_blocking=True)
        pixels_b = batch["pixel_values_b"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        indices = batch["row_index"].numpy()
        ambiguous_np = np.asarray(ambiguous_mask[indices], dtype=np.bool_)
        ambiguous = torch.as_tensor(ambiguous_np, dtype=torch.bool, device=device)
        clean = ~ambiguous
        q = torch.as_tensor(quality[indices], dtype=torch.float32, device=device)
        repair_batch = torch.as_tensor(repair_plan[indices], dtype=torch.bool, device=device) & clean
        original_supervision = clean & ~repair_batch if dynamic_noise else clean
        trusted_batch = torch.as_tensor(trusted[indices], dtype=torch.bool, device=device) & original_supervision
        frozen = torch.as_tensor(np.asarray(center_features[indices]), dtype=torch.float32, device=device)
        with _autocast_context(device):
            logits_a, _, base_a, adapted_a = classifier(pixels_a, None)
            logits_b, _, base_b, adapted_b = classifier(pixels_b, None)
        # Keep probability targets and all loss arithmetic outside autocast.
        logits_a, logits_b = logits_a.float(), logits_b.float()
        ce = 0.5 * (
            balanced_softmax_cross_entropy(logits_a, y, class_prior)
            + balanced_softmax_cross_entropy(logits_b, y, class_prior)
        )
        gce = 0.5 * (
            generalized_cross_entropy(logits_a, y, q=0.7)
            + generalized_cross_entropy(logits_b, y, q=0.7)
        )
        uncertain_sample = (1.0 - q) * gce * original_supervision.float()
        if not dynamic_noise and repair_weight > 0.0 and repair_batch.any():
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
            counters[3] += repair_batch.sum()
        if dynamic_noise and repair_batch.any():
            if audit_probabilities is None:
                raise ValueError("dynamic repairs require complete audit targets")
            target = torch.as_tensor(np.asarray(audit_probabilities[indices]), device=device)
            uncertain_sample = uncertain_sample + repaired_soft_loss(
                logits_a, logits_b, target, repair_batch, dynamic_repair_weight(epoch))
            counters[3] += repair_batch.sum()
        if ambiguous.any():
            batch_candidates = [candidate_labels[index] for index in indices[ambiguous_np]]
            partial_target = partial_label_targets(
                frozen[ambiguous], prototypes, batch_candidates
            ).detach()
            partial_soft = partial_label_loss(
                logits_a[ambiguous], logits_b[ambiguous], partial_target
            )
            uncertain_sample = uncertain_sample.clone()
            uncertain_sample[ambiguous] = PARTIAL_WEIGHT * partial_soft
        batch_normalizer = max(len(y), 1)
        trusted_loss = (q * ce * original_supervision.float()).sum() / batch_normalizer
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
        sample_scale = torch.where(
            ambiguous,
            torch.full_like(q, PARTIAL_WEIGHT),
            torch.ones_like(q),
        )
        consistency = (
            sample_scale
            * 0.5
            * (1.0 + q)
            * symmetric_kl_loss(logits_a, logits_b, 2.0)
        )
        # Keep the V6 denominator: sample_scale must reduce an ambiguous
        # row's *total* contribution instead of being cancelled by a
        # weighted-mean denominator.
        loss_consistency = consistency.sum() / (0.5 * (1.0 + q)).sum().clamp_min(1e-6)
        anchor_rows = 0.5 * (
            1.0 - (base_a.float() * F.normalize(frozen, dim=1)).sum(dim=1)
            + 1.0 - (base_b.float() * F.normalize(frozen, dim=1)).sum(dim=1)
        )
        anchor = (sample_scale * anchor_rows).sum() / batch_normalizer
        regularizer = 0.05 * loss_consistency + 0.10 * anchor
        if soft_teacher and epoch > WARMUP_EPOCHS:
            historical = torch.as_tensor(
                np.asarray(temporal[indices]), dtype=torch.float32, device=device
            )
            teacher_loss, teacher_count = soft_teacher_regularizer(
                logits_a, logits_b, q, original_supervision, historical
            )
            regularizer = regularizer + teacher_loss
            counters[5] += teacher_count

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
        group_start = ((batch_id - 1) // gradient_accumulation) * gradient_accumulation + 1
        group_size = min(gradient_accumulation, len(loader) - group_start + 1)
        _add_gradients(
            parameters,
            trusted_grads,
            uncertain_grads,
            regularizer_grads,
            group_size,
        )
        should_step = batch_id % gradient_accumulation == 0 or batch_id == len(loader)
        if should_step:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_step += 1
            update_ema(ema, classifier, 0.999, optimizer_step)

        was_training = classifier.training
        classifier.eval()
        try:
            with torch.no_grad(), swapped_trainable(classifier, ema), _autocast_context(device):
                ema_logits_a, _, _, ema_adapted_a = classifier(pixels_a, None)
                ema_logits_b, _, _, ema_adapted_b = classifier(pixels_b, None)
        finally:
            classifier.train(was_training)
        with torch.no_grad():
            average_prob = 0.5 * (
                ema_logits_a.float().softmax(dim=1) + ema_logits_b.float().softmax(dim=1)
            )
            per_sample_tensor = q * ce.detach().float() * original_supervision.float() + uncertain_sample.detach().float()
            packed = torch.cat((average_prob, ema_logits_a.argmax(dim=1)[:, None].float(),
                                ema_logits_b.argmax(dim=1)[:, None].float(), per_sample_tensor[:, None]), dim=1).cpu().numpy()
            current = packed[:, :-3]
            seen = temporal_seen[indices]
            if seen.any():
                temporal[indices[seen]] = (
                    0.85 * temporal[indices[seen]].astype(np.float32) + 0.15 * current[seen]
                ).astype(np.float16)
            if (~seen).any():
                temporal[indices[~seen]] = current[~seen].astype(np.float16)
            temporal_seen[indices] = True
            pred_a = packed[:, -3].astype(np.int64)
            pred_b = packed[:, -2].astype(np.int64)
            consensus = np.where(pred_a == pred_b, pred_a, -1)
            previous = previous_view_prediction[indices]
            stability_delta = np.where(
                previous >= 0,
                (consensus >= 0) & (consensus == previous),
                consensus >= 0,
            ).astype(np.float32)
            # WeightedRandomSampler may draw a row more than once in one
            # batch.  NumPy advanced ``+=`` only applies one of those writes;
            # add.at preserves every observation deterministically.
            np.add.at(stability_sum, indices, stability_delta)
            np.add.at(stability_count, indices, 1)
            previous_view_prediction[indices] = consensus
            per_sample = packed[:, -1]
            np.add.at(loss_sum, indices, per_sample)
            np.add.at(loss_count, indices, 1)
            if epoch > WARMUP_EPOCHS:
                queue_mask = (
                    trusted_batch
                    & torch.as_tensor(consensus, dtype=torch.long, device=device).eq(y)
                    & average_prob.max(dim=1).values.ge(0.80)
                )
                queue.enqueue(0.5 * (ema_adapted_a + ema_adapted_b), y, queue_mask)

        total = trusted_loss.detach().float() + uncertain_loss.detach().float() + regularizer.detach().float()
        counters[0] += total.double() * len(y)
        counters[1] += ((logits_a.argmax(dim=1) == y) & clean).sum()
        counters[2] += clean.sum()
        counters[4] += ambiguous.sum()
        all_rows_seen += len(y)
        progress.update(batch_id, len(y))
    loss_total, correct, clean_rows_seen, repaired_rows, ambiguous_rows_seen, soft_teacher_rows = counters.cpu().tolist()
    return {
        "loss": loss_total / max(all_rows_seen, 1.0),
        "noisy_accuracy": correct / max(clean_rows_seen, 1.0),
        "ambiguous_rows_seen": int(ambiguous_rows_seen),
        "repaired_rows": int(repaired_rows),
        "projected_batches": int(projected_batches),
        "soft_teacher_rows": int(soft_teacher_rows),
    }, optimizer_step


@torch.no_grad()
def evaluate_two_view_cv(
    classifier: RobustCLIPClassifier,
    loader: DataLoader,
    device: torch.device,
    trusted_global: np.ndarray,
    class_counts: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    classifier.eval()
    original_rows, hflip_rows, labels_rows, index_rows = [], [], [], []
    progress = Progress("validation", device, len(loader))
    for batch_id, batch in enumerate(loader, 1):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with _autocast_context(device):
            original = classifier(pixels, None)[0]
            hflip = classifier(torch.flip(pixels, dims=[3]), None)[0]
        original_rows.append(original.float().cpu())
        hflip_rows.append(hflip.float().cpu())
        labels_rows.append(batch["label"].cpu())
        index_rows.append(batch["row_index"].cpu())
        progress.update(batch_id, len(pixels))
    original = torch.cat(original_rows)
    hflip = torch.cat(hflip_rows)
    labels = torch.cat(labels_rows)
    indices = torch.cat(index_rows).numpy()
    trusted = torch.as_tensor(trusted_global[indices], dtype=torch.bool)
    result: dict[str, Any] = {
        "macro_accuracy": macro_accuracy(original, labels),
        "trusted_macro_accuracy": macro_accuracy(original, labels, trusted),
        "trusted_macro_nll": macro_nll(original, labels, trusted),
        "accuracy": float((original.argmax(dim=1) == labels).float().mean()),
        "trusted_rows": int(trusted.sum()),
        "rows": int(len(labels)),
    }
    group_accuracy = head_mid_tail_accuracy(original, labels, class_counts)
    result.update({f"{name}_accuracy": value for name, value in group_accuracy.items()})
    result["head_mid_tail_mean_accuracy"] = float(np.mean(list(group_accuracy.values())))
    mixed = 0.5 * (original + hflip)
    result.update(
        {
            "selection_macro_accuracy": macro_accuracy(mixed, labels),
            "selection_macro_nll": macro_nll(mixed, labels),
        }
    )
    return result, original.numpy(), hflip.numpy(), labels.numpy(), {"original_weight": 0.5}


def _is_better_cv(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    candidate_key = (float(candidate["selection_macro_accuracy"]),
                     -float(candidate["selection_macro_nll"]), -int(candidate["epoch"]))
    best_key = (float(best["selection_macro_accuracy"]),
                -float(best["selection_macro_nll"]), -int(best["epoch"]))
    return candidate_key > best_key


def save_inference_checkpoint(
    path: Path,
    classifier: RobustCLIPClassifier,
    class_names: Sequence[str],
    config: dict[str, Any],
    quality: np.ndarray,
    class_prior: np.ndarray,
    labels: np.ndarray,
    supervised_indices: Sequence[int],
    selected_epoch: int,
    stage: str,
    validation_indices: Sequence[int],
    trusted_validation: np.ndarray,
    dataset_digest: str,
    selection_metrics: dict[str, Any] | None = None,
) -> None:
    selected = np.asarray(supervised_indices, dtype=np.int64)
    class_quality = np.bincount(
        labels[selected], weights=quality[selected], minlength=len(class_names)
    ).astype(np.float32)
    atomic_torch_save(
        {
            "format_version": FORMAT_VERSION,
            "kind": "robust_clip_v20",
            "model": trainable_state_dict(classifier),
            "trainable_parameter_names": trainable_parameter_names(classifier),
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
            "selection_metrics": selection_metrics,
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
    arrays: dict[str, np.ndarray],
    optimizer_step: int,
    history: list[dict[str, Any]],
    best_state: dict[str, torch.Tensor],
    best_record: dict[str, Any] | None,
    selected_epoch: int | None,
    config: dict[str, Any],
    dataset_digest: str,
) -> None:
    payload = {
        "kind": "train_v20_resume",
        "resume_version": RESUME_VERSION,
        "stage": stage,
        "epoch": int(epoch),
        "model": trainable_state_dict(classifier),
        "trainable_parameter_names": trainable_parameter_names(classifier),
        "optimizer": optimizer.state_dict(),
        "optimizer_group_metadata": optimizer_group_metadata(optimizer),
        "scheduler": scheduler.state_dict(),
        "ema": {name: value.detach().cpu() for name, value in ema.items()},
        "queue": queue.state_dict(),
        **arrays,
        "optimizer_step": int(optimizer_step),
        "history": history,
        "best_state": {name: value.detach().cpu() for name, value in best_state.items()},
        "best_record": best_record,
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


def load_resume(
    path: str | None,
    config: dict[str, Any],
    dataset_digest: str,
    device: torch.device,
) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "train_v20_resume" or payload.get("resume_version") != RESUME_VERSION:
        raise ValueError("unsupported V20 resume checkpoint")
    if scientific_config(payload.get("config", {})) != scientific_config(config):
        raise ValueError("resume configuration does not match this V20 run")
    if payload.get("dataset_signature") != dataset_digest:
        raise ValueError("resume dataset does not match this V20 run")
    if payload.get("stage") != config.get("stage", "validate"):
        raise ValueError("resume stage differs from requested stage")
    if payload.get("stage") not in ("validate", "refit"):
        raise ValueError("invalid V20 resume stage")
    return payload


def _restore_rng(payload: dict[str, Any]) -> None:
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
    supervised_indices: Sequence[int],
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
    ambiguous_mask: np.ndarray,
    candidate_labels: Sequence[Sequence[int]],
    sampler_mode: str,
    row_repeat_factors: np.ndarray,
    gradient_accumulation: int,
    neighbor_evidence: NeighborEvidence,
) -> tuple[dict[str, torch.Tensor], int, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if (batch_size, gradient_accumulation) != (256, 1):
        raise ValueError("V20 training requires actual batch_size=256 and gradient_accumulation=1")
    validate_recipe_config(config)
    dataset = training_dataset(rows, train_indices, processor, config)
    optimizer, visual, head = build_optimizer(classifier, config)
    steps_per_epoch = math.ceil(len(dataset) / batch_size / gradient_accumulation)
    scheduler = make_v6_scheduler(optimizer, config["schedule_epochs"] * steps_per_epoch)
    dim = int(classifier.classifier.shape[1])
    queue = SelectiveFeatureQueue(len(class_names), QUEUE_PER_CLASS, dim, device)
    center_features = view_features[0]
    prototypes = torch.as_tensor(signals.prototypes, dtype=torch.float32, device=device)
    trusted = trusted_mask(
        quality, signals.prototype_label, labels, neighbor_evidence, ambiguous_mask
    )
    class_prior_np = _class_prior(
        labels, quality, supervised_indices, row_repeat_factors, len(class_names)
    )
    class_prior = torch.as_tensor(class_prior_np, dtype=torch.float32, device=device)
    arrays: dict[str, np.ndarray] = {
        "quality": quality,
        "trusted": trusted,
        "class_prior": class_prior_np,
        "temporal": np.zeros((len(rows), len(class_names)), dtype=np.float16),
        "temporal_seen": np.zeros(len(rows), dtype=np.bool_),
        "audit_previous_prob": np.zeros((len(rows), len(class_names)), dtype=np.float16),
        "audit_current_prob": np.zeros((len(rows), len(class_names)), dtype=np.float16),
        "audit_view_consensus": np.full(len(rows), -1, dtype=np.int64),
        "last_repair_plan": np.zeros(len(rows), dtype=np.bool_),
        "audit_epoch1_seen": np.zeros(len(rows), dtype=np.bool_),
        "audit_epoch2_seen": np.zeros(len(rows), dtype=np.bool_),
        "audit_nll": np.zeros(len(rows), dtype=np.float32),
        "audit_stability": np.zeros(len(rows), dtype=np.float32),
        "previous_view_prediction": np.full(len(rows), -1, dtype=np.int64),
        "stability_sum": np.zeros(len(rows), dtype=np.float32),
        "stability_count": np.zeros(len(rows), dtype=np.int16),
        "loss_sum": np.zeros(len(rows), dtype=np.float32),
        "loss_count": np.zeros(len(rows), dtype=np.int16),
    }
    arrays["quality"][ambiguous_mask] = PARTIAL_WEIGHT
    ema = clone_trainable(classifier)
    best_state = clone_trainable(classifier)
    best_record: dict[str, Any] | None = None
    optimizer_step = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if resume is not None:
        validate_trainable_metadata(classifier, resume)
        validate_trainable_state(classifier, resume["ema"], exact=True)
        validate_trainable_state(classifier, resume["best_state"], exact=True)
        load_classifier_state(classifier, resume["model"])
        expected_groups = optimizer_group_metadata(optimizer)
        if resume.get("optimizer_group_metadata") != expected_groups:
            raise ValueError("resume optimizer parameter order or learning-rate groups differ")
        optimizer.load_state_dict(resume["optimizer"])
        if optimizer_group_metadata(optimizer) != expected_groups:
            raise ValueError("restored optimizer groups differ from frozen configuration")
        scheduler.load_state_dict(resume["scheduler"])
        saved_step = int(resume["optimizer_step"])
        expected_rates = [group["initial_lr"] for group in optimizer.param_groups]
        current_rates = [group["lr"] for group in optimizer.param_groups]
        scheduled_rates = [rate * schedule(saved_step)
                           for rate, schedule in zip(expected_rates, scheduler.lr_lambdas)]
        if (scheduler.last_epoch != saved_step
                or scheduler.base_lrs != expected_rates
                or current_rates != scheduled_rates
                or scheduler.get_last_lr() != scheduled_rates
                or saved_step != int(resume["epoch"]) * steps_per_epoch):
            raise ValueError("resume optimizer learning rates or scheduler step differ")
        ema = {name: value.to(device).clone() for name, value in resume["ema"].items()}
        queue.load_state_dict(resume["queue"])
        for name in arrays:
            arrays[name] = np.asarray(resume[name]).copy()
        quality = arrays["quality"]
        trusted = arrays["trusted"]
        class_prior_np = arrays["class_prior"]
        class_prior = torch.as_tensor(class_prior_np, dtype=torch.float32, device=device)
        optimizer_step = int(resume["optimizer_step"])
        history = list(resume["history"])
        best_state = {name: value.to(device).clone() for name, value in resume["best_state"].items()}
        best_record = resume.get("best_record")
        start_epoch = int(resume["epoch"]) + 1
        _restore_rng(resume)
        print(f"resumed_v20 stage={stage} next_epoch={start_epoch}", flush=True)

    selected_rows = np.asarray(supervised_indices, dtype=np.int64)
    audit_rows = np.asarray(train_indices, dtype=np.int64)
    if start_epoch > WARMUP_EPOCHS and not arrays["audit_epoch2_seen"][audit_rows].all():
        raise RuntimeError("resumed V20 run lacks complete second-epoch EMA audit")

    margin_thresholds = prototype_margin_thresholds(
        labels, signals.prototype_margin, supervised_indices, 0.75
    )
    if resume is None:
        # Establish a recoverable state before the first validation checkpoint.
        # A failure after best_model.pt but before epoch-one completion must not
        # strand an immutable resource plan with no legal resume checkpoint.
        save_resume(
            output_dir / "resume_latest.pt", stage=stage, epoch=0,
            classifier=classifier, optimizer=optimizer, scheduler=scheduler,
            ema=ema, queue=queue, arrays=arrays, optimizer_step=optimizer_step,
            history=history, best_state=best_state, best_record=best_record,
            selected_epoch=epochs if stage == "refit" else None,
            config=config, dataset_digest=dataset_digest,
        )
    print(
        f"v20_stage={stage} rows={len(train_indices)} supervised={len(supervised_indices)} "
        f"ambiguous={int(ambiguous_mask[np.asarray(train_indices)].sum())} "
        f"visual_params={sum(p.numel() for p in visual):,} head_params={sum(p.numel() for p in head):,} "
        f"initial_trusted={int(trusted[np.asarray(supervised_indices)].sum())}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs + 1):
        supervision_started = synchronized_time(device)
        old_quality = quality.copy()
        refreshed = epoch == WARMUP_EPOCHS + 1 or (config["dynamic_noise"] and epoch > 3 and epoch % 2 == 1)
        if refreshed:
            if not arrays["audit_epoch1_seen"][audit_rows].all() or not arrays["audit_epoch2_seen"][audit_rows].all():
                raise RuntimeError("V20 quality fusion requires complete EMA audits from both warmup epochs")
            base_quality = fused_reliability(
                labels,
                supervised_indices,
                signals.label_margin,
                signals.view_agreement,
                arrays["audit_nll"],
                arrays["audit_stability"],
            )
            quality = apply_clean_quality_floor(
                blend_quality(base_quality, neighbor_evidence, ambiguous_mask), supervised_indices, config["quality_floor"]
            )
            if not np.isfinite(quality[selected_rows]).all() or np.any(quality[selected_rows] < config["quality_floor"]):
                raise RuntimeError("V20 fused clean-training quality is nonfinite or below its floor")
            trusted = trusted_mask(
                quality, signals.prototype_label, labels, neighbor_evidence, ambiguous_mask
            )
            class_prior_np = _class_prior(
                labels, quality, supervised_indices, row_repeat_factors, len(class_names)
            )
            class_prior = torch.as_tensor(class_prior_np, dtype=torch.float32, device=device)
            arrays["quality"] = quality
            arrays["trusted"] = trusted
            arrays["class_prior"] = class_prior_np
            if config["dynamic_noise"]:
                # Historical queue entries lack row IDs; clear them when trust changes.
                queue.valid.zero_()
            print(
                f"reliability_fused mean={quality[np.asarray(supervised_indices)].mean():.4f} "
                f"trusted={int(trusted[np.asarray(supervised_indices)].sum())}",
                flush=True,
            )

        repair_plan = np.zeros(len(rows), dtype=np.bool_)
        if epoch >= 5 and not config["dynamic_noise"]:
            repair_plan = build_repair_plan(
                labels,
                supervised_indices,
                quality,
                signals.prototype_label,
                signals.prototype_margin,
                margin_thresholds,
                arrays["temporal"],
                arrays["previous_view_prediction"],
                confidence_threshold=0.80,
                per_class_fraction=0.15,
            )
        if epoch >= 5 and config["dynamic_noise"]:
            repair_plan = dynamic_repair_plan(
                labels, supervised_indices, signals.prototype_label, neighbor_evidence,
                arrays["audit_previous_prob"], arrays["audit_current_prob"], arrays["audit_view_consensus"],
                confidence=config["repair_confidence"], fraction=config["repair_fraction"],
            )
        diagnostics = repair_diagnostics(
            labels, supervised_indices, arrays["last_repair_plan"], repair_plan, old_quality, quality,
            arrays["audit_current_prob"], len(class_names))
        arrays["last_repair_plan"] = repair_plan.copy()
        loader = deterministic_loader(
            dataset,
            epoch,
            batch_size,
            workers,
            device,
            sampler_mode,
            row_repeat_factors,
            config["prefetch_factor"],
        )
        supervision_seconds = synchronized_time(device) - supervision_started
        train_started = synchronized_time(device)
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
            arrays["temporal"],
            arrays["temporal_seen"],
            arrays["previous_view_prediction"],
            arrays["stability_sum"],
            arrays["stability_count"],
            arrays["loss_sum"],
            arrays["loss_count"],
            optimizer_step,
            ambiguous_mask,
            candidate_labels,
            gradient_accumulation,
            bool(config["soft_teacher"]),
            dynamic_noise=config["dynamic_noise"], audit_probabilities=arrays["audit_current_prob"],
            schedule_epochs=config["schedule_epochs"],
        )
        train_seconds = synchronized_time(device) - train_started
        audit_started = synchronized_time(device)
        if should_audit(epoch, config["dynamic_noise"]):
            current, nll, stability, seen, consensus = audit_ema_coverage(
                classifier, ema, processor, rows, train_indices, device, config["eval_batch_size"], workers,
                prefetch_factor=config["prefetch_factor"],
                first_prob=arrays["audit_current_prob"] if epoch > 1 else None,
            )
            arrays["audit_previous_prob"] = arrays["audit_current_prob"]
            arrays["audit_current_prob"] = current
            arrays["audit_view_consensus"] = consensus
            arrays["audit_nll"], arrays["audit_stability"] = nll, stability
            arrays["audit_epoch1_seen" if epoch == 1 else "audit_epoch2_seen"] = seen
            if epoch >= 2:
                arrays["temporal"][audit_rows] = current[audit_rows]
                arrays["temporal_seen"][audit_rows] = True
            print(f"v20_ema_audit epoch={epoch} covered={int(seen.sum())}", flush=True)
        audit_seconds = synchronized_time(device) - audit_started
        record: dict[str, Any] = {
            "train_seconds": train_seconds,
            "supervision_seconds": supervision_seconds,
            "audit_seconds": audit_seconds,
            "epoch": epoch,
            **stats,
            "noise_diagnostics": diagnostics,
            "quality_refreshed": refreshed,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "repair_plan_unique": int(repair_plan[selected_rows].sum()),
            "repair_plan_by_quality_bin": {
                f"{low:.2f}-{high:.2f}": int(
                    (
                        repair_plan[selected_rows]
                        & (quality[selected_rows] >= low)
                        & (quality[selected_rows] < high)
                    ).sum()
                )
                for low, high in (
                    (0.0, 0.25), (0.25, 0.40), (0.40, 0.55),
                    (0.55, 0.70), (0.70, 1.01),
                )
            },
            "audit_epoch1_covered": int(arrays["audit_epoch1_seen"][audit_rows].sum()),
            "audit_epoch2_covered": int(arrays["audit_epoch2_seen"][audit_rows].sum()),
        }
        validation_started = synchronized_time(device)
        if validation_loader is not None:
            with swapped_trainable(classifier, ema):
                metrics, original, hflip, validation_labels, selection = evaluate_two_view_cv(
                    classifier,
                    validation_loader,
                    device,
                    trusted_validation,
                    class_counts,
                )
            record["validation_seconds"] = synchronized_time(device) - validation_started
            record.update(metrics)
            epoch_dir = output_dir / "validation_epochs"
            epoch_checkpoint = epoch_dir / f"epoch_{epoch:02d}.pt"
            with swapped_trainable(classifier, ema):
                save_inference_checkpoint(
                    epoch_checkpoint,
                    classifier,
                    class_names,
                    config,
                    quality,
                    class_prior_np,
                    labels,
                    supervised_indices,
                    epoch,
                    "validation",
                    validation_indices,
                    trusted_validation,
                    dataset_digest,
                    record,
                )
            save_epoch_logits(
                epoch_dir, epoch, original, hflip, validation_labels, validation_indices, config["precision"],
                config=config,
            )
            if _is_better_cv(record, best_record):
                best_record = dict(record)
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
                        supervised_indices,
                        epoch,
                        "validation",
                        validation_indices,
                        trusted_validation,
                        dataset_digest,
                        best_record,
                    )
                np.save(output_dir / "val_original_logits.npy", original)
                np.save(output_dir / "val_hflip_logits.npy", hflip)
                np.save(output_dir / "val_labels.npy", validation_labels)
            print(
                f"epoch={epoch}/{epochs} loss={stats['loss']:.5f} noisy_acc={stats['noisy_accuracy']:.4f} "
                f"selection_macro={metrics['selection_macro_accuracy']:.4f} macro={metrics['macro_accuracy']:.4f} "
                f"tail={metrics['tail_accuracy']:.4f} repairs={stats['repaired_rows']} "
                f"ambiguous_seen={stats['ambiguous_rows_seen']}",
                flush=True,
            )
        else:
            best_record = {"epoch": epoch}
            best_state = {name: value.detach().clone() for name, value in ema.items()}
            print(
                f"final_epoch={epoch}/{epochs} loss={stats['loss']:.5f} "
                f"noisy_acc={stats['noisy_accuracy']:.4f} repairs={stats['repaired_rows']} "
                f"ambiguous_seen={stats['ambiguous_rows_seen']}",
                flush=True,
            )
        print(f"v20_timing epoch={epoch} train_s={train_seconds:.1f} audit_s={audit_seconds:.1f} "
              f"validation_s={record.get('validation_seconds', 0):.1f} supervision_s={supervision_seconds:.1f}", flush=True)
        history.append(record)
        history_tmp = output_dir / ".history.json.tmp"
        history_tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
        history_tmp.replace(output_dir / "history.json")
        arrays["quality"] = quality
        arrays["trusted"] = trusted
        arrays["class_prior"] = class_prior_np
        save_resume(
            output_dir / "resume_latest.pt",
            stage=stage,
            epoch=epoch,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            queue=queue,
            arrays=arrays,
            optimizer_step=optimizer_step,
            history=history,
            best_state=best_state,
            best_record=best_record,
            selected_epoch=epochs if stage == "refit" else None,
            config=config,
            dataset_digest=dataset_digest,
        )
        if len(history) >= 2:
            recent = history[-2:]
            per_epoch = float(np.mean([r['train_seconds'] + r.get('supervision_seconds', 0) + r.get('validation_seconds', 0) for r in recent]))
            audits = [r['audit_seconds'] for r in history if should_audit(r['epoch'], config['dynamic_noise'])]
            audit_mean = float(np.mean(audits[-2:])) if audits else 0.0
            remaining_audits = sum(should_audit(e, config['dynamic_noise']) for e in range(epoch + 1, epochs + 1))
            remaining_seconds = (epochs - epoch) * per_epoch + remaining_audits * audit_mean
            print(f"v20_eta stage={stage} completed_epoch={epoch} remaining_hours={remaining_seconds / 3600:.2f} "
                  "estimate=recent_two_epochs_excludes_future_eval_and_refit", flush=True)
            eta = pipeline_eta(
                history, stage=stage, stop_epoch=epochs, train_rows=len(train_indices),
                final_rows=config.get("eta_final_rows", len(train_indices)),
                prior_elapsed_seconds=config.get("eta_prior_elapsed_seconds", 0.0),
                prior_elapsed_known=config.get("eta_prior_elapsed_known", stage == "validate"),
            )
            eta_temporary = output_dir / ".eta.json.tmp"
            eta_temporary.write_text(json.dumps(eta, indent=2), encoding="utf-8")
            eta_temporary.replace(output_dir / "eta.json")
            low, high = (value / 3600 for value in eta["remaining_seconds_range"])
            print(f"v20_pipeline_eta stage={stage} completed_epoch={epoch} remaining_hours={low:.2f}-{high:.2f} "
                  f"budget_48h_exceeded={eta['budget_exceeded']} "
                  f"prior_elapsed_known={eta['prior_elapsed_known']} elapsed_wait_included={eta['external_dependency_wait_included']} future_wait=unknown", flush=True)

    if validation_loader is not None and best_record is None:
        raise RuntimeError("V20 validation did not produce a selectable checkpoint")
    best_epoch = int(best_record["epoch"]) if best_record is not None else epochs
    return best_state, best_epoch, quality, class_prior_np, history


def main() -> None:
    from select_v20 import load_selection

    args = parse_args()
    validate_args(args)
    args.soft_teacher = True
    selection = load_selection(args.selection_json) if args.selection_json else None
    if selection:
        if selection.get('recipe') not in RECIPES:
            raise ValueError('V20 refit requires a winning candidate; reuse fallback artifacts directly')
        if args.recipe != selection['recipe']:
            raise ValueError('refit recipe differs from selected winner')
        # Training data policy must be identical to the chosen validation recipe.
        for key in ('conflict_policy', 'sampler'):
            if getattr(args, key) != selection['training_config'][key]:
                raise ValueError(f'refit {key} differs from selection')
    resource_plan = bind_resource_plan(args, selection)
    configure_determinism(SEED)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    if (output_dir / 'resume_latest.pt').exists() and not args.resume:
        raise ValueError('existing V20 run requires --resume or a new directory')
    if any(output_dir.glob('*.pt')) and not args.resume:
        raise ValueError('output contains checkpoints; use a fresh V20 directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest_dataset(args.data_manifest, args.train_dir, args.conflict_policy)
    rows, class_names = manifest.rows, manifest.class_names
    labels = np.asarray([row[1] for row in rows], dtype=np.int64)
    train_indices, clean_train_indices, validation_indices = stage_indices(manifest, args.stage)
    if args.stage == 'validate' and len(validation_indices) < 5:
        raise ValueError('validation needs at least five held-out rows')
    config = checkpoint_config(args, manifest)
    if resource_plan['dataset_signature'] != manifest.signature:
        raise ValueError('resource plan dataset differs from the current manifest')
    if resource_plan['base_model_identity'] != config['base_model_identity']:
        raise ValueError('resource plan official weights differ from this training run')
    config['eta_final_rows'] = len(manifest.final_indices)
    config['eta_prior_elapsed_seconds'] = 0.0
    config['eta_prior_elapsed_known'] = not bool(selection)
    if selection:
        load_selection(args.selection_json, dataset_signature=manifest.signature,
                       class_names=class_names, base_identity=config['base_model_identity'])
        if selection['calibration']['precision'] != config['precision']:
            raise ValueError('refit must preserve the selected calibration precision')
        stop_epoch = int(selection['selected_epoch'])
        validation_output = selection['training_config'].get('output_dir')
        if validation_output:
            try:
                metrics = json.loads((Path(validation_output) / 'metrics.json').read_text(encoding='utf-8'))
                validation_history = metrics['training_history']
                if validation_history and metrics.get('dataset_signature') == manifest.signature:
                    config['eta_prior_elapsed_seconds'] = training_history_seconds(validation_history)
                    config['eta_prior_elapsed_known'] = True
            except (OSError, ValueError, KeyError, TypeError):
                print('v20_pipeline_eta prior_validation_elapsed=unavailable', flush=True)
    else:
        stop_epoch = config['schedule_epochs']
    config['stop_epoch'] = stop_epoch
    # Frozen encoder/preprocessing are identical to V13: preserve cache content version 13.
    feature_signature = hashlib.sha256(json.dumps({
        'dataset': manifest.signature, 'base': config['base_model_identity'],
        'views': config['fixed_feature_views'], 'size': 224, 'version': 13,
    }, sort_keys=True).encode()).hexdigest()
    # Frozen per-image encodings do not use labels; they may be reused for refit.
    feature_cache = Path(args.feature_cache or (
        selection['training_config'].get('feature_cache') if selection
        else output_dir / 'cache' / 'frozen.npy'))
    config['feature_cache'] = str(feature_cache.resolve())
    config['feature_signature'] = feature_signature
    pool_id = pool_signature(feature_signature, clean_train_indices, validation_indices)
    config['training_pool_signature'] = pool_id
    neighbor_path = Path(args.neighbor_cache or output_dir / 'cache' / 'neighbors.npz')
    config['neighbor_cache'] = str(neighbor_path.resolve())
    resume = load_resume(args.resume, config, manifest.signature, device)
    row_repeat_factors = np.ones(len(rows), dtype=np.float64)
    counts_ids = np.asarray(clean_train_indices, dtype=np.int64)
    effective_counts = np.bincount(labels[counts_ids], minlength=len(class_names)).astype(np.float64)
    if args.sampler == 'repeat-factor':
        row_repeat_factors, effective_counts = effective_repeat_factors(labels, manifest.clean_mask, train_indices)
    model, processor = load_clip(args.model_dir, device)
    validate_clip_vit_b32(model)
    view_features = load_or_create_multiview_features(
        model, processor, rows, feature_cache, feature_signature, device, args.batch_size, args.workers,
        pin_memory=pin_memory_enabled(device), prefetch_factor=args.prefetch_factor)
    processor = resolution_processor(processor, config["image_size"])
    payload = read_manifest(args.data_manifest)
    selected_rows = sorted([row for row in payload['rows'] if row['role'] == 'clean' or
                            (args.conflict_policy == 'partial' and row['role'] == 'ambiguous')],
                           key=lambda row: row['relative_path'])
    root = Path(args.train_dir).resolve()
    if len(selected_rows) != len(rows) or any(
            Path(rows[i][0]).resolve().relative_to(root).as_posix() != row['relative_path']
            for i, row in enumerate(selected_rows)):
        raise ValueError('manifest row order mismatch')
    neighbors = load_or_create_neighbor_evidence(
        neighbor_path, pool_id, view_features, labels, [row['sha256'] for row in selected_rows],
        clean_train_indices, validation_indices, len(class_names), device)
    validate_neighbor_evidence(neighbors, len(rows))
    signals = cross_fitted_visual_signals(
        view_features, labels, clean_train_indices, len(class_names), folds=5,
        seed=SEED, keep_fraction=.70, iterations=2)
    trusted_validation = np.zeros(len(rows), dtype=bool)
    validation_loader = None
    if validation_indices:
        val_signals = fixed_prototype_signals(view_features, labels, validation_indices, signals.prototypes)
        trusted_validation = trusted_validation_mask(labels, validation_indices, val_signals)
        validation_loader = DataLoader(
            FolderImageDataset(rows, validation_indices, processor, augment='none'),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers,
            **loader_options(args.workers, args.prefetch_factor), pin_memory=pin_memory_enabled(device))
    quality = blend_quality(initial_reliability(labels, clean_train_indices, signals.label_margin,
                                               signals.view_agreement), neighbors, manifest.ambiguous_mask)
    class_counts = np.bincount(labels[counts_ids], minlength=len(class_names))
    configure_determinism(SEED + 101)
    classifier = build_classifier(model, signals.prototypes, device, config)
    print(f"v20_start recipe={args.recipe} stage={args.stage} stop_epoch={stop_epoch} "
          f"schedule_epochs={config['schedule_epochs']} train_rows={len(train_indices)} "
          f"validation_rows={len(validation_indices)}", flush=True)
    best_state, selected_epoch, final_quality, final_prior, history = fit(
        classifier=classifier, processor=processor, rows=rows, train_indices=train_indices,
        supervised_indices=clean_train_indices, labels=labels, view_features=view_features,
        signals=signals, quality=quality, validation_loader=validation_loader,
        trusted_validation=trusted_validation, class_counts=class_counts, class_names=class_names,
        config=config, dataset_digest=manifest.signature, output_dir=output_dir, device=device,
        batch_size=args.batch_size, workers=args.workers, epochs=stop_epoch, stage=args.stage,
        validation_indices=validation_indices, resume=resume, ambiguous_mask=manifest.ambiguous_mask,
        candidate_labels=manifest.candidate_labels, sampler_mode=args.sampler,
        row_repeat_factors=row_repeat_factors, gradient_accumulation=args.gradient_accumulation,
        neighbor_evidence=neighbors)
    if selection:
        # No validation or new calibration after the former holdout joins training.
        with swapped_trainable(classifier, best_state):
            save_inference_checkpoint(output_dir / 'refit_uncalibrated.pt', classifier, class_names, config,
                                      final_quality, final_prior, labels, clean_train_indices, selected_epoch,
                                      'refit', [], trusted_validation, manifest.signature)
        checkpoint = torch.load(output_dir / 'refit_uncalibrated.pt', map_location='cpu', weights_only=False)
        checkpoint['class_bias'] = torch.tensor(selection['class_bias'], dtype=torch.float32)
        checkpoint['calibration'] = dict(selection['calibration'])
        checkpoint['refit_selection_sha256'] = config['selection_sha256']
        checkpoint['calibration']['frozen_before_refit'] = True
        atomic_torch_save(checkpoint, output_dir / 'model.pt')
    result = {
        'format_version': FORMAT_VERSION, 'recipe': args.recipe, 'stage': args.stage,
        'selected_epoch': selected_epoch, 'validation_history': history if validation_indices else [],
        'training_history': history, 'train_rows': len(train_indices), 'validation_rows': len(validation_indices),
        'classes': len(class_names), 'dataset_signature': manifest.signature,
        'manifest_summary': manifest.summary, 'effective_class_counts': effective_counts.tolist(), 'config': config,
        'online_improvement_confirmed': False,
    }
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(output_dir / 'sample_reliability.npz', quality=final_quality,
                        train_indices=np.asarray(train_indices), class_prior=final_prior,
                        dataset_signature=np.asarray(manifest.signature), pool_signature=np.asarray(pool_id))
    print(f'v20_complete recipe={args.recipe} stage={args.stage} selected_epoch={selected_epoch}', flush=True)


if __name__ == '__main__':
    main()
