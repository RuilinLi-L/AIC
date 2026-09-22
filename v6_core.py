"""Core algorithms for the V6 single-model robust CLIP pipeline.

This module is intentionally independent from the V5 trainer.  It contains
the deterministic split, cross-fitted visual reliability, long-tail loss,
selective contrastive queue, gradient projection and evaluation primitives
used by ``train_v6.py`` and its tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PrototypeSignals:
    prototypes: np.ndarray
    fold_ids: np.ndarray
    label_similarity: np.ndarray
    label_margin: np.ndarray
    prototype_margin: np.ndarray
    prototype_label: np.ndarray
    prototype_confidence: np.ndarray
    view_agreement: np.ndarray


def capped_stratified_split(
    rows: Sequence[tuple[str, int, str]], seed: int
) -> tuple[list[int], list[int]]:
    """Create a deterministic, tail-safe validation split.

    Classes with one row remain entirely in training.  Classes with two or
    three rows contribute one validation row.  Larger classes contribute 8%,
    clamped to [2, 16], while retaining at least two training examples.
    """
    by_class: dict[int, list[int]] = {}
    for index, (_, label, _) in enumerate(rows):
        by_class.setdefault(int(label), []).append(index)
    rng = random.Random(seed)
    train, validation = [], []
    for label in sorted(by_class):
        indices = list(by_class[label])
        rng.shuffle(indices)
        count = len(indices)
        if count == 1:
            n_val = 0
        elif count < 4:
            n_val = 1
        else:
            n_val = min(16, max(2, int(round(0.08 * count))), count - 2)
        validation.extend(indices[:n_val])
        train.extend(indices[n_val:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def stratified_fold_ids(
    labels: np.ndarray,
    indices: Sequence[int],
    folds: int = 5,
    seed: int = 2026,
) -> np.ndarray:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    labels = np.asarray(labels, dtype=np.int64)
    result = np.full(len(labels), -1, dtype=np.int16)
    rng = np.random.default_rng(seed)
    candidates = np.asarray(indices, dtype=np.int64)
    for label in np.unique(labels[candidates]):
        class_indices = candidates[labels[candidates] == label].copy()
        rng.shuffle(class_indices)
        local_folds = min(folds, len(class_indices))
        result[class_indices] = np.arange(len(class_indices), dtype=np.int16) % local_folds
    return result


def _normalize(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def _trimmed_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    num_classes: int,
    keep_fraction: float,
    iterations: int,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    if not len(indices):
        if fallback is None:
            raise ValueError("cannot build prototypes from an empty index set")
        return np.asarray(fallback, dtype=np.float32).copy()
    selected = np.asarray(indices, dtype=np.int64)
    candidates = selected.copy()
    prototypes = np.zeros((num_classes, features.shape[1]), dtype=np.float32)
    for iteration in range(iterations):
        prototypes.fill(0.0)
        counts = np.zeros(num_classes, dtype=np.int64)
        np.add.at(prototypes, labels[selected], features[selected])
        np.add.at(counts, labels[selected], 1)
        if fallback is not None:
            missing = counts == 0
            prototypes[missing] = fallback[missing]
            counts[missing] = 1
        elif (counts == 0).any():
            missing = np.where(counts == 0)[0].tolist()
            raise ValueError(f"prototype candidates missing classes: {missing[:10]}")
        prototypes /= counts[:, None]
        prototypes = _normalize(prototypes)
        if iteration + 1 < iterations and keep_fraction < 1.0:
            retained: list[np.ndarray] = []
            similarities = np.sum(features[candidates] * prototypes[labels[candidates]], axis=1)
            for label in np.unique(labels[candidates]):
                local = candidates[labels[candidates] == label]
                local_scores = similarities[labels[candidates] == label]
                keep = max(1, int(math.ceil(len(local) * keep_fraction)))
                retained.append(local[np.argsort(local_scores)[::-1][:keep]])
            selected = np.concatenate(retained)
    return prototypes


def _batched_prototype_statistics(
    views: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    temperature: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_similarity = np.empty(len(indices), dtype=np.float32)
    label_margin = np.empty(len(indices), dtype=np.float32)
    prototype_margin = np.empty(len(indices), dtype=np.float32)
    pseudo = np.empty(len(indices), dtype=np.int64)
    confidence = np.empty(len(indices), dtype=np.float32)
    agreement = np.empty(len(indices), dtype=np.float32)
    averaged = _normalize(views[:, indices].mean(axis=0))
    for start in range(0, len(indices), chunk_size):
        stop = min(len(indices), start + chunk_size)
        local_indices = indices[start:stop]
        local_labels = labels[local_indices]
        scores = averaged[start:stop] @ prototypes.T
        chosen = scores[np.arange(len(scores)), local_labels]
        masked = scores.copy()
        masked[np.arange(len(scores)), local_labels] = -np.inf
        label_similarity[start:stop] = chosen
        label_margin[start:stop] = chosen - masked.max(axis=1)
        local_pseudo = scores.argmax(axis=1)
        pseudo[start:stop] = local_pseudo
        if scores.shape[1] > 1:
            top_two = np.partition(scores, -2, axis=1)[:, -2:]
            top_two.sort(axis=1)
            prototype_margin[start:stop] = top_two[:, 1] - top_two[:, 0]
        else:
            prototype_margin[start:stop] = np.inf
        scaled = scores / float(temperature)
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_scores = np.exp(scaled)
        probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        confidence[start:stop] = probabilities.max(axis=1)
        view_predictions = []
        for view_id in range(views.shape[0]):
            view_scores = np.asarray(views[view_id, local_indices], dtype=np.float32) @ prototypes.T
            view_predictions.append(view_scores.argmax(axis=1))
        stacked = np.stack(view_predictions, axis=1)
        agreement[start:stop] = (stacked == local_pseudo[:, None]).mean(axis=1)
    return label_similarity, label_margin, prototype_margin, pseudo, confidence, agreement


def cross_fitted_visual_signals(
    view_features: np.ndarray,
    labels: np.ndarray,
    candidate_indices: Sequence[int],
    num_classes: int,
    folds: int = 5,
    seed: int = 2026,
    keep_fraction: float = 0.70,
    iterations: int = 2,
    temperature: float = 0.07,
    chunk_size: int = 2048,
) -> PrototypeSignals:
    """Estimate prototype signals without scoring a row by a prototype containing it."""
    views = _normalize(np.asarray(view_features, dtype=np.float32))
    if views.ndim != 3:
        raise ValueError("view_features must have shape [views, rows, dim]")
    labels = np.asarray(labels, dtype=np.int64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if len(labels) != views.shape[1]:
        raise ValueError("labels and view feature rows do not match")
    averaged = _normalize(views.mean(axis=0))
    full_prototypes = _trimmed_prototypes(
        averaged, labels, candidates, num_classes, keep_fraction, iterations
    )
    fold_ids = stratified_fold_ids(labels, candidates, folds=folds, seed=seed)
    similarity = np.full(len(labels), np.nan, dtype=np.float32)
    margin = np.full(len(labels), np.nan, dtype=np.float32)
    prototype_margin = np.full(len(labels), np.nan, dtype=np.float32)
    pseudo = np.full(len(labels), -1, dtype=np.int64)
    confidence = np.zeros(len(labels), dtype=np.float32)
    agreement = np.zeros(len(labels), dtype=np.float32)
    for fold in range(folds):
        evaluation = candidates[fold_ids[candidates] == fold]
        if not len(evaluation):
            continue
        build = candidates[fold_ids[candidates] != fold]
        fold_prototypes = _trimmed_prototypes(
            averaged,
            labels,
            build,
            num_classes,
            keep_fraction,
            iterations,
            fallback=full_prototypes,
        )
        values = _batched_prototype_statistics(
            views, evaluation, labels, fold_prototypes, temperature, chunk_size
        )
        (
            similarity[evaluation],
            margin[evaluation],
            prototype_margin[evaluation],
            pseudo[evaluation],
            confidence[evaluation],
            agreement[evaluation],
        ) = values
    missing = candidates[np.isnan(margin[candidates])]
    if len(missing):
        raise RuntimeError(f"cross-fitted prototype scoring missed {len(missing)} rows")
    # A singleton class cannot be evaluated without self leakage.  It remains
    # trainable, but receives the lowest reliability rank downstream.
    counts = np.bincount(labels[candidates], minlength=num_classes)
    singleton_rows = candidates[counts[labels[candidates]] == 1]
    margin[singleton_rows] = -np.inf
    agreement[singleton_rows] = 0.0
    return PrototypeSignals(
        prototypes=full_prototypes,
        fold_ids=fold_ids,
        label_similarity=similarity,
        label_margin=margin,
        prototype_margin=prototype_margin,
        prototype_label=pseudo,
        prototype_confidence=confidence,
        view_agreement=agreement,
    )


def fixed_prototype_signals(
    view_features: np.ndarray,
    labels: np.ndarray,
    indices: Sequence[int],
    prototypes: np.ndarray,
    temperature: float = 0.07,
    chunk_size: int = 2048,
) -> PrototypeSignals:
    views = _normalize(np.asarray(view_features, dtype=np.float32))
    labels = np.asarray(labels, dtype=np.int64)
    indices_array = np.asarray(indices, dtype=np.int64)
    values = _batched_prototype_statistics(
        views, indices_array, labels, np.asarray(prototypes, dtype=np.float32), temperature, chunk_size
    )
    similarity = np.full(len(labels), np.nan, dtype=np.float32)
    margin = np.full(len(labels), np.nan, dtype=np.float32)
    prototype_margin = np.full(len(labels), np.nan, dtype=np.float32)
    pseudo = np.full(len(labels), -1, dtype=np.int64)
    confidence = np.zeros(len(labels), dtype=np.float32)
    agreement = np.zeros(len(labels), dtype=np.float32)
    (
        similarity[indices_array],
        margin[indices_array],
        prototype_margin[indices_array],
        pseudo[indices_array],
        confidence[indices_array],
        agreement[indices_array],
    ) = values
    return PrototypeSignals(
        prototypes=np.asarray(prototypes, dtype=np.float32),
        fold_ids=np.full(len(labels), -1, dtype=np.int16),
        label_similarity=similarity,
        label_margin=margin,
        prototype_margin=prototype_margin,
        prototype_label=pseudo,
        prototype_confidence=confidence,
        view_agreement=agreement,
    )


def classwise_percentile(
    labels: np.ndarray,
    values: np.ndarray,
    indices: Sequence[int],
    higher_is_better: bool = True,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    candidates = np.asarray(indices, dtype=np.int64)
    result = np.zeros(len(labels), dtype=np.float32)
    for label in np.unique(labels[candidates]):
        local = candidates[labels[candidates] == label]
        local_values = np.nan_to_num(values[local], nan=-np.inf if higher_is_better else np.inf)
        goodness = local_values if higher_is_better else -local_values
        order = np.argsort(goodness, kind="stable")
        scale = np.linspace(0.05, 1.0, len(local), dtype=np.float32)
        ranks = np.empty(len(local), dtype=np.float32)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and goodness[order[stop]] == goodness[order[start]]:
                stop += 1
            ranks[order[start:stop]] = float(scale[start:stop].mean())
            start = stop
        result[local] = ranks
    return result


def initial_reliability(
    labels: np.ndarray,
    indices: Sequence[int],
    prototype_margin: np.ndarray,
    view_agreement: np.ndarray,
) -> np.ndarray:
    q_proto = classwise_percentile(labels, prototype_margin, indices)
    q_aug = classwise_percentile(labels, view_agreement, indices)
    quality = np.sqrt(np.maximum(q_proto, 1e-4) ** 1.34 * np.maximum(q_aug, 1e-4) ** 0.66)
    result = np.zeros(len(labels), dtype=np.float32)
    candidate_array = np.asarray(indices, dtype=np.int64)
    result[candidate_array] = np.clip(quality[candidate_array], 0.05, 1.0)
    return result


def fused_reliability(
    labels: np.ndarray,
    indices: Sequence[int],
    prototype_margin: np.ndarray,
    view_agreement: np.ndarray,
    warmup_loss: np.ndarray,
    temporal_agreement: np.ndarray,
) -> np.ndarray:
    candidates = np.asarray(indices, dtype=np.int64)
    signals = (
        (classwise_percentile(labels, prototype_margin, candidates), 0.40),
        (classwise_percentile(labels, view_agreement, candidates), 0.20),
        (classwise_percentile(labels, warmup_loss, candidates, higher_is_better=False), 0.25),
        (classwise_percentile(labels, temporal_agreement, candidates), 0.15),
    )
    log_quality = np.zeros(len(labels), dtype=np.float32)
    for signal, exponent in signals:
        log_quality[candidates] += exponent * np.log(np.maximum(signal[candidates], 1e-4))
    result = np.zeros(len(labels), dtype=np.float32)
    result[candidates] = np.clip(np.exp(log_quality[candidates]), 0.05, 1.0)
    return result


def reliable_class_prior(
    labels: np.ndarray,
    quality: np.ndarray,
    indices: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    candidates = np.asarray(indices, dtype=np.int64)
    mass = np.bincount(
        np.asarray(labels, dtype=np.int64)[candidates],
        weights=np.asarray(quality, dtype=np.float64)[candidates],
        minlength=num_classes,
    )
    positive = mass[mass > 0]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    mass = np.where(mass > 0, mass, fallback)
    return (mass / max(float(mass.sum()), 1e-8)).astype(np.float32)


def balanced_softmax_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_prior: torch.Tensor,
) -> torch.Tensor:
    adjusted = logits + class_prior.clamp_min(1e-8).log()[None, :]
    return F.cross_entropy(adjusted, target, reduction="none")


def generalized_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    q: float = 0.7,
) -> torch.Tensor:
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    probability = logits.softmax(dim=-1).gather(1, target[:, None]).squeeze(1).clamp_min(1e-7)
    return (1.0 - probability.pow(q)) / q


def project_conflicting_gradients(
    trusted: Sequence[torch.Tensor | None],
    uncertain: Sequence[torch.Tensor | None],
    epsilon: float = 1e-12,
) -> tuple[list[torch.Tensor | None], bool]:
    if len(trusted) != len(uncertain):
        raise ValueError("gradient lists must have the same length")
    dot = None
    norm = None
    for clean_grad, noisy_grad in zip(trusted, uncertain):
        if clean_grad is None or noisy_grad is None:
            continue
        local_dot = (clean_grad.float() * noisy_grad.float()).sum()
        local_norm = clean_grad.float().square().sum()
        dot = local_dot if dot is None else dot + local_dot
        norm = local_norm if norm is None else norm + local_norm
    if dot is None or norm is None or float(dot.detach()) >= 0.0:
        return [None if value is None else value.clone() for value in uncertain], False
    coefficient = dot / norm.clamp_min(epsilon)
    projected: list[torch.Tensor | None] = []
    for clean_grad, noisy_grad in zip(trusted, uncertain):
        if noisy_grad is None:
            projected.append(None)
        elif clean_grad is None:
            projected.append(noisy_grad.clone())
        else:
            projected.append(noisy_grad - coefficient.to(noisy_grad.dtype) * clean_grad)
    return projected, True


class SelectiveFeatureQueue:
    """A fixed-size, per-class EMA feature queue kept on the training device."""

    def __init__(self, num_classes: int, capacity: int, dim: int, device: torch.device):
        self.num_classes = int(num_classes)
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.device = device
        self.features = torch.zeros(num_classes, capacity, dim, dtype=torch.float16, device=device)
        self.valid = torch.zeros(num_classes, capacity, dtype=torch.bool, device=device)
        self.pointer = torch.zeros(num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor, labels: torch.Tensor, trusted: torch.Tensor) -> None:
        normalized = F.normalize(features.detach().float(), dim=-1).to(torch.float16)
        for feature, label in zip(normalized[trusted], labels[trusted]):
            class_id = int(label)
            slot = int(self.pointer[class_id] % self.capacity)
            self.features[class_id, slot] = feature
            self.valid[class_id, slot] = True
            self.pointer[class_id] += 1

    def flattened(self) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.valid.nonzero(as_tuple=False)
        if not len(indices):
            return (
                torch.empty(0, self.dim, dtype=torch.float32, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )
        values = self.features[indices[:, 0], indices[:, 1]].float()
        return values, indices[:, 0]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "features": self.features.detach().cpu(),
            "valid": self.valid.detach().cpu(),
            "pointer": self.pointer.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.features.copy_(state["features"].to(self.device))
        self.valid.copy_(state["valid"].to(self.device))
        self.pointer.copy_(state["pointer"].to(self.device))


def selective_supervised_contrastive_loss(
    anchors: torch.Tensor,
    paired: torch.Tensor,
    labels: torch.Tensor,
    trusted: torch.Tensor,
    queue: SelectiveFeatureQueue,
    temperature: float = 0.07,
) -> torch.Tensor:
    if not trusted.any():
        return anchors.sum() * 0.0
    anchor = F.normalize(anchors[trusted].float(), dim=-1)
    paired_values = F.normalize(paired[trusted].float(), dim=-1)
    trusted_labels = labels[trusted]
    queued_features, queued_labels = queue.flattened()
    candidates = torch.cat([paired_values, queued_features], dim=0)
    candidate_labels = torch.cat([trusted_labels, queued_labels], dim=0)
    logits = anchor @ F.normalize(candidates, dim=-1).t() / float(temperature)
    positives = trusted_labels[:, None].eq(candidate_labels[None, :])
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    return -((log_prob * positives).sum(dim=1) / positive_count).mean()


def prototype_margin_thresholds(
    labels: np.ndarray,
    margins: np.ndarray,
    indices: Sequence[int],
    quantile: float = 0.75,
) -> np.ndarray:
    candidates = np.asarray(indices, dtype=np.int64)
    thresholds = np.full(int(labels.max()) + 1, np.inf, dtype=np.float32)
    for label in np.unique(labels[candidates]):
        local = margins[candidates[labels[candidates] == label]]
        thresholds[int(label)] = float(np.quantile(local, quantile))
    return thresholds


def build_repair_plan(
    labels: np.ndarray,
    indices: Sequence[int],
    quality: np.ndarray,
    prototype_label: np.ndarray,
    prototype_margin: np.ndarray,
    margin_thresholds: np.ndarray,
    temporal_probabilities: np.ndarray,
    previous_view_prediction: np.ndarray,
    confidence_threshold: float = 0.80,
    per_class_fraction: float = 0.15,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    candidates = np.asarray(indices, dtype=np.int64)
    temporal = np.asarray(temporal_probabilities, dtype=np.float32)
    temporal_sum = temporal.sum(axis=1, keepdims=True)
    normalized = temporal / np.maximum(temporal_sum, 1e-8)
    teacher_label = normalized.argmax(axis=1)
    teacher_confidence = normalized.max(axis=1)
    uncertain = (quality < 0.70) | (prototype_label != labels)
    eligible = (
        uncertain
        & (prototype_label != labels)
        & (prototype_label == teacher_label)
        & (prototype_label == previous_view_prediction)
        & (teacher_confidence >= confidence_threshold)
        & (prototype_margin >= margin_thresholds[labels])
    )
    plan = np.zeros(len(labels), dtype=np.bool_)
    for label in np.unique(labels[candidates]):
        local = candidates[(labels[candidates] == label) & eligible[candidates]]
        class_size = int((labels[candidates] == label).sum())
        keep = min(len(local), int(math.floor(class_size * per_class_fraction)))
        if keep:
            ranking = teacher_confidence[local] + prototype_margin[local]
            plan[local[np.argsort(ranking)[::-1][:keep]]] = True
    return plan


def macro_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    if not len(labels):
        return 0.0
    prediction = logits.argmax(dim=1)
    values = []
    for label in labels.unique(sorted=True):
        local = labels == label
        values.append((prediction[local] == labels[local]).float().mean())
    return float(torch.stack(values).mean())


def macro_nll(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    if not len(labels):
        return float("inf")
    losses = F.cross_entropy(logits, labels, reduction="none")
    values = [losses[labels == label].mean() for label in labels.unique(sorted=True)]
    return float(torch.stack(values).mean())


def head_mid_tail_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_counts: np.ndarray,
) -> dict[str, float]:
    counts = np.asarray(class_counts, dtype=np.float32)
    lower, upper = np.quantile(counts, [1.0 / 3.0, 2.0 / 3.0])
    prediction = logits.argmax(dim=1)
    result: dict[str, float] = {}
    for name, selected in {
        "tail": np.where(counts <= lower)[0],
        "mid": np.where((counts > lower) & (counts <= upper))[0],
        "head": np.where(counts > upper)[0],
    }.items():
        class_values = []
        for label in selected:
            local = labels == int(label)
            if local.any():
                class_values.append((prediction[local] == labels[local]).float().mean())
        result[name] = float(torch.stack(class_values).mean()) if class_values else 0.0
    return result
