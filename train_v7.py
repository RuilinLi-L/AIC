"""Train the manifest-driven V7 robust single-model CLIP system."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from evaluate_tta_v6 import _grid, _search_mix_macro, strict_stratified_folds
from robust_clip import (
    FolderImageDataset,
    PairedFolderImageDataset,
    RobustCLIPClassifier,
    load_classifier_state,
    load_clip,
    resolve_device,
    symmetric_kl_loss,
    trainable_state_dict,
)
from train_v5 import atomic_torch_save, clone_trainable, swapped_trainable, update_ema
from train_v6 import (
    DeterministicMultiViewDataset,
    _add_gradients,
    _autocast_context,
    _prototype_distribution,
    build_classifier,
    build_optimizer,
    configure_determinism,
    load_or_create_multiview_features,
    make_v6_scheduler,
    trainable_parameters,
    trusted_validation_mask,
    validate_clip_vit_b32,
)
from v6_core import (
    PrototypeSignals,
    SelectiveFeatureQueue,
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
from v7_data import ManifestDataset, effective_repeat_factors, load_manifest_dataset


FORMAT_VERSION = 7
RESUME_VERSION = 1
SEED = 2026
WARMUP_EPOCHS = 2
MAX_EPOCHS = 12
GRADIENT_ACCUMULATION = 4
QUEUE_PER_CLASS = 16
PARTIAL_WEIGHT = 0.25
PARTIAL_PROTOTYPE_MIX = 0.75
PARTIAL_TEMPERATURE = 0.07
PARTIAL_MAX_PROBABILITY = 0.80
CV_TIE_TOLERANCE = 0.0005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--conflict-policy", choices=["drop", "partial"], default="drop")
    parser.add_argument("--sampler", choices=["shuffle", "repeat-factor"], default="shuffle")
    parser.add_argument("--model-dir", default="clip-ViT-B-32")
    parser.add_argument("--output-dir", default="outputs/v7_clean_drop")
    parser.add_argument("--feature-cache", default="outputs/frozen_clip_multiview_v7.npy")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.workers < 0 or args.gradient_accumulation < 1:
        raise ValueError("batch-size and gradient-accumulation must be positive; workers non-negative")
    if Path(args.train_dir).resolve() == Path(args.output_dir).resolve():
        raise ValueError("output-dir must not be the training directory")


def checkpoint_config(args: argparse.Namespace, manifest: ManifestDataset) -> dict[str, Any]:
    return {
        "version": "v7",
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
        "epochs": MAX_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "gradient_accumulation": args.gradient_accumulation,
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
        "validation_split": "manifest_clean_tail_safe_capped_8_percent",
        "fixed_feature_views": ["center", "hflip", "light_seed_2", "light_seed_3"],
        "partial_weight": PARTIAL_WEIGHT,
        "partial_prototype_mix": PARTIAL_PROTOTYPE_MIX,
        "partial_temperature": PARTIAL_TEMPERATURE,
        "partial_max_probability": PARTIAL_MAX_PROBABILITY,
        "repeat_factor": "min(4,sqrt(median_count/class_count))",
        "epoch_selection": "strict_5fold_macro_hflip_tta",
    }


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
    dataset: PairedFolderImageDataset,
    epoch: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    sampler_mode: str,
    row_repeat_factors: np.ndarray,
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
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        generator=generator if sampler is None else None,
        worker_init_fn=partial(_seed_loader_worker, epoch=epoch),
    )


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
) -> tuple[dict[str, float], int]:
    classifier.train()
    parameters = trainable_parameters(classifier)
    optimizer.zero_grad(set_to_none=True)
    loss_total = correct = clean_rows_seen = all_rows_seen = repaired_rows = projected_batches = 0.0
    ambiguous_rows_seen = 0.0
    repair_weight = 0.0
    if epoch >= 5:
        repair_weight = min(0.50, 0.50 * (epoch - 4) / max(1, MAX_EPOCHS - 4))
    for batch_id, batch in enumerate(loader, 1):
        pixels_a = batch["pixel_values_a"].to(device, non_blocking=True)
        pixels_b = batch["pixel_values_b"].to(device, non_blocking=True)
        y = batch["label"].to(device)
        indices = batch["row_index"].numpy()
        ambiguous_np = np.asarray(ambiguous_mask[indices], dtype=np.bool_)
        ambiguous = torch.as_tensor(ambiguous_np, dtype=torch.bool, device=device)
        clean = ~ambiguous
        q = torch.as_tensor(quality[indices], dtype=torch.float32, device=device)
        trusted_batch = torch.as_tensor(trusted[indices], dtype=torch.bool, device=device) & clean
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
            uncertain_sample = (1.0 - q) * gce * clean.float()
            repair_batch = (
                torch.as_tensor(repair_plan[indices], dtype=torch.bool, device=device) & clean
            )
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
            trusted_loss = (q * ce * clean.float()).sum() / batch_normalizer
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
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_step += 1
            update_ema(ema, classifier, 0.999, optimizer_step)

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
            per_sample = (
                q * ce.detach().float() * clean.float() + uncertain_sample.detach().float()
            ).cpu().numpy()
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
        loss_total += float(total) * len(y)
        correct += float(((logits_a.argmax(dim=1) == y) & clean).sum())
        clean_rows_seen += float(clean.sum())
        ambiguous_rows_seen += float(ambiguous.sum())
        all_rows_seen += len(y)
    return {
        "loss": loss_total / max(all_rows_seen, 1.0),
        "noisy_accuracy": correct / max(clean_rows_seen, 1.0),
        "ambiguous_rows_seen": int(ambiguous_rows_seen),
        "repaired_rows": int(repaired_rows),
        "projected_batches": int(projected_batches),
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
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with _autocast_context(device):
            original = classifier(pixels, None)[0]
            hflip = classifier(torch.flip(pixels, dims=[3]), None)[0]
        original_rows.append(original.float().cpu())
        hflip_rows.append(hflip.float().cpu())
        labels_rows.append(batch["label"].cpu())
        index_rows.append(batch["row_index"].cpu())
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
    labels_np = labels.numpy()
    fold_ids = strict_stratified_folds(labels_np, folds=5, seed=SEED)
    selection = _search_mix_macro(
        original.numpy(),
        hflip.numpy(),
        labels_np,
        fold_ids,
        _grid(1.0, 0.05),
        _grid(2.0, 0.05),
        device,
    )
    result.update(
        {
            "cv_macro_accuracy": selection["cv_macro_accuracy"],
            "cv_macro_nll": selection["cv_macro_nll"],
            "cv_original_weight": selection["original_weight"],
            "cv_alpha": selection["alpha"],
        }
    )
    return result, original.numpy(), hflip.numpy(), labels_np, selection


def _is_better_cv(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    accuracy_delta = float(candidate["cv_macro_accuracy"]) - float(best["cv_macro_accuracy"])
    if accuracy_delta > CV_TIE_TOLERANCE:
        return True
    if accuracy_delta < -CV_TIE_TOLERANCE:
        return False
    nll_delta = float(candidate["cv_macro_nll"]) - float(best["cv_macro_nll"])
    if nll_delta < -1e-9:
        return True
    if nll_delta > 1e-9:
        return False
    return int(candidate["epoch"]) < int(best["epoch"])


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
            "kind": "robust_clip_v7",
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
        "kind": "train_v7_resume",
        "resume_version": RESUME_VERSION,
        "stage": stage,
        "epoch": int(epoch),
        "model": trainable_state_dict(classifier),
        "optimizer": optimizer.state_dict(),
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
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "train_v7_resume" or payload.get("resume_version") != RESUME_VERSION:
        raise ValueError("unsupported V7 resume checkpoint")
    if payload.get("config") != config:
        raise ValueError("resume configuration does not match this V7 run")
    if payload.get("dataset_signature") != dataset_digest:
        raise ValueError("resume dataset does not match this V7 run")
    if payload.get("stage") not in {"validation", "final"}:
        raise ValueError("invalid V7 resume stage")
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
) -> tuple[dict[str, torch.Tensor], int, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    dataset = PairedFolderImageDataset(rows, train_indices, processor, augment="light")
    optimizer, visual, head = build_optimizer(classifier)
    steps_per_epoch = math.ceil(len(dataset) / batch_size / gradient_accumulation)
    scheduler = make_v6_scheduler(optimizer, epochs * steps_per_epoch)
    dim = int(classifier.classifier.shape[1])
    queue = SelectiveFeatureQueue(len(class_names), QUEUE_PER_CLASS, dim, device)
    center_features = view_features[0]
    prototypes = torch.as_tensor(signals.prototypes, dtype=torch.float32, device=device)
    trusted = (quality >= 0.70) & (signals.prototype_label == labels) & ~ambiguous_mask
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
        load_classifier_state(classifier, resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
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
        print(f"resumed_v7 stage={stage} next_epoch={start_epoch}", flush=True)

    margin_thresholds = prototype_margin_thresholds(
        labels, signals.prototype_margin, supervised_indices, 0.75
    )
    print(
        f"v7_stage={stage} rows={len(train_indices)} supervised={len(supervised_indices)} "
        f"ambiguous={int(ambiguous_mask[np.asarray(train_indices)].sum())} "
        f"visual_params={sum(p.numel() for p in visual):,} head_params={sum(p.numel() for p in head):,} "
        f"initial_trusted={int(trusted[np.asarray(supervised_indices)].sum())}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs + 1):
        if epoch == WARMUP_EPOCHS + 1 and not (
            resume is not None and start_epoch > WARMUP_EPOCHS + 1
        ):
            warmup_loss = arrays["loss_sum"] / np.maximum(arrays["loss_count"], 1)
            temporal_agreement = arrays["stability_sum"] / np.maximum(
                arrays["stability_count"], 1
            )
            quality = fused_reliability(
                labels,
                supervised_indices,
                signals.label_margin,
                signals.view_agreement,
                warmup_loss,
                temporal_agreement,
            )
            quality[ambiguous_mask] = PARTIAL_WEIGHT
            trusted = (quality >= 0.70) & (signals.prototype_label == labels) & ~ambiguous_mask
            class_prior_np = _class_prior(
                labels, quality, supervised_indices, row_repeat_factors, len(class_names)
            )
            class_prior = torch.as_tensor(class_prior_np, dtype=torch.float32, device=device)
            arrays["quality"] = quality
            arrays["trusted"] = trusted
            arrays["class_prior"] = class_prior_np
            print(
                f"reliability_fused mean={quality[np.asarray(supervised_indices)].mean():.4f} "
                f"trusted={int(trusted[np.asarray(supervised_indices)].sum())}",
                flush=True,
            )

        repair_plan = np.zeros(len(rows), dtype=np.bool_)
        if epoch >= 5:
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
        loader = deterministic_loader(
            dataset,
            epoch,
            batch_size,
            workers,
            device,
            sampler_mode,
            row_repeat_factors,
        )
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
        )
        record: dict[str, Any] = {"epoch": epoch, **stats}
        if validation_loader is not None:
            with swapped_trainable(classifier, ema):
                metrics, original, hflip, validation_labels, selection = evaluate_two_view_cv(
                    classifier,
                    validation_loader,
                    device,
                    trusted_validation,
                    class_counts,
                )
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
                f"cv_macro={metrics['cv_macro_accuracy']:.4f} macro={metrics['macro_accuracy']:.4f} "
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
        history.append(record)
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
            selected_epoch=epochs if stage == "final" else None,
            config=config,
            dataset_digest=dataset_digest,
        )

    if validation_loader is not None and best_record is None:
        raise RuntimeError("V7 validation did not produce a selectable checkpoint")
    best_epoch = int(best_record["epoch"]) if best_record is not None else epochs
    return best_state, best_epoch, quality, class_prior_np, history


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_determinism(SEED)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest_dataset(args.data_manifest, args.train_dir, args.conflict_policy)
    rows = manifest.rows
    labels = np.asarray([label for _, label, _ in rows], dtype=np.int64)
    class_names = manifest.class_names
    if len(class_names) != int(labels.max()) + 1:
        raise ValueError("training class indices are not contiguous")
    config = checkpoint_config(args, manifest)
    resume = load_resume(args.resume, config, manifest.signature, device)

    row_repeat_factors = np.ones(len(rows), dtype=np.float64)
    clean_train_for_counts = np.asarray(
        [index for index in manifest.train_indices if manifest.clean_mask[index]], dtype=np.int64
    )
    effective_counts = np.bincount(
        labels[clean_train_for_counts], minlength=len(class_names)
    ).astype(np.float64)
    if args.sampler == "repeat-factor":
        row_repeat_factors, effective_counts = effective_repeat_factors(
            labels, manifest.clean_mask, manifest.train_indices
        )

    model, processor = load_clip(args.model_dir, device)
    validate_clip_vit_b32(model)
    view_features = load_or_create_multiview_features(
        model,
        processor,
        rows,
        Path(args.feature_cache),
        manifest.signature,
        device,
        args.batch_size,
        args.workers,
    )
    validation_indices = manifest.validation_indices
    clean_train_indices = [index for index in manifest.train_indices if manifest.clean_mask[index]]
    if len(validation_indices) < 5:
        raise ValueError("V7 requires at least five clean validation rows")
    validation_signals_source = cross_fitted_visual_signals(
        view_features,
        labels,
        clean_train_indices,
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
        clean_train_indices,
        validation_signals_source.label_margin,
        validation_signals_source.view_agreement,
    )
    validation_quality[manifest.ambiguous_mask] = PARTIAL_WEIGHT
    class_counts = np.bincount(labels[np.asarray(clean_train_indices)], minlength=len(class_names))
    validation_loader = DataLoader(
        FolderImageDataset(rows, validation_indices, processor, augment="none"),
        batch_size=args.batch_size,
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
            train_indices=manifest.train_indices,
            supervised_indices=clean_train_indices,
            labels=labels,
            view_features=view_features,
            signals=validation_signals_source,
            quality=validation_quality,
            validation_loader=validation_loader,
            trusted_validation=trusted_validation,
            class_counts=class_counts,
            class_names=class_names,
            config=config,
            dataset_digest=manifest.signature,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            epochs=MAX_EPOCHS,
            stage="validation",
            validation_indices=validation_indices,
            resume=resume,
            ambiguous_mask=manifest.ambiguous_mask,
            candidate_labels=manifest.candidate_labels,
            sampler_mode=args.sampler,
            row_repeat_factors=row_repeat_factors,
            gradient_accumulation=args.gradient_accumulation,
        )
        with torch.no_grad():
            for name, parameter in classifier.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(best_state[name])
        selected_record = next(
            record for record in validation_history if int(record["epoch"]) == selected_epoch
        )
        (output_dir / "validation_metrics.json").write_text(
            json.dumps(
                {
                    "selected_epoch": selected_epoch,
                    "selected": selected_record,
                    "history": validation_history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        del classifier, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        selected_epoch = int(resume["selected_epoch"])
        validation_payload = json.loads(
            (output_dir / "validation_metrics.json").read_text(encoding="utf-8")
        )
        selected_record = validation_payload["selected"]
        validation_history = validation_payload["history"]
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    clean_final_indices = [index for index in manifest.final_indices if manifest.clean_mask[index]]
    final_signals = cross_fitted_visual_signals(
        view_features,
        labels,
        clean_final_indices,
        len(class_names),
        folds=5,
        seed=SEED,
        keep_fraction=0.70,
        iterations=2,
    )
    final_quality = initial_reliability(
        labels,
        clean_final_indices,
        final_signals.label_margin,
        final_signals.view_agreement,
    )
    final_quality[manifest.ambiguous_mask] = PARTIAL_WEIGHT
    final_repeat_factors = np.ones(len(rows), dtype=np.float64)
    if args.sampler == "repeat-factor":
        final_repeat_factors, _ = effective_repeat_factors(
            labels, manifest.clean_mask, manifest.final_indices
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
        train_indices=manifest.final_indices,
        supervised_indices=clean_final_indices,
        labels=labels,
        view_features=view_features,
        signals=final_signals,
        quality=final_quality,
        validation_loader=None,
        trusted_validation=np.zeros(len(rows), dtype=np.bool_),
        class_counts=np.bincount(labels[np.asarray(clean_final_indices)], minlength=len(class_names)),
        class_names=class_names,
        config=config,
        dataset_digest=manifest.signature,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=selected_epoch,
        stage="final",
        validation_indices=validation_indices,
        resume=final_resume,
        ambiguous_mask=manifest.ambiguous_mask,
        candidate_labels=manifest.candidate_labels,
        sampler_mode=args.sampler,
        row_repeat_factors=final_repeat_factors,
        gradient_accumulation=args.gradient_accumulation,
    )
    with torch.no_grad():
        for name, parameter in final_classifier.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(final_state[name])
    save_inference_checkpoint(
        output_dir / "model_uncalibrated.pt",
        final_classifier,
        class_names,
        config,
        final_quality,
        final_prior,
        labels,
        clean_final_indices,
        selected_epoch,
        "final",
        validation_indices,
        trusted_validation,
        manifest.signature,
        selected_record,
    )
    np.savez_compressed(
        output_dir / "sample_reliability.npz",
        quality=final_quality,
        trusted=(final_quality >= 0.70) & (final_signals.prototype_label == labels),
        ambiguous=manifest.ambiguous_mask,
        label_margin=final_signals.label_margin,
        prototype_margin=final_signals.prototype_margin,
        prototype_label=final_signals.prototype_label,
        prototype_confidence=final_signals.prototype_confidence,
        view_agreement=final_signals.view_agreement,
        fold_ids=final_signals.fold_ids,
        class_prior=final_prior,
        dataset_signature=np.asarray(manifest.signature),
    )
    metrics = {
        "format_version": FORMAT_VERSION,
        "selected_epoch": selected_epoch,
        "validation_selected": selected_record,
        "validation_history": validation_history,
        "final_history": final_history,
        "trusted_validation_rows": int(trusted_validation.sum()),
        "train_rows": len(manifest.final_indices),
        "clean_rows": int(manifest.clean_mask.sum()),
        "ambiguous_rows": int(manifest.ambiguous_mask.sum()),
        "classes": len(class_names),
        "dataset_signature": manifest.signature,
        "manifest_summary": manifest.summary,
        "effective_class_counts": effective_counts.tolist(),
        "config": config,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"v7_complete selected_epoch={selected_epoch} "
        f"checkpoint={output_dir / 'model_uncalibrated.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
