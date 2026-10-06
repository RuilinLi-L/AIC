"""Out-of-fold frozen-CLIP neighborhood evidence for V8 training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_tta_v6 import strict_stratified_folds


NEIGHBORS = 32
TEMPERATURE = 0.07
FOLDS = 5
SEED = 2026


@dataclass
class NeighborEvidence:
    label_support: np.ndarray
    top_label: np.ndarray
    top_support: np.ndarray
    percentile: np.ndarray
    fold_ids: np.ndarray


def _embeddings(view_features: np.ndarray, device: torch.device) -> torch.Tensor:
    if view_features.ndim != 3 or view_features.shape[0] != 4:
        raise ValueError("V8 expects four frozen feature views")
    total = None
    for view in view_features:
        normalized = F.normalize(torch.as_tensor(np.asarray(view), device=device).float(), dim=1)
        total = normalized if total is None else total + normalized
    return F.normalize(total, dim=1)


def _query_pool(
    embeddings: torch.Tensor,
    labels: np.ndarray,
    hashes: Sequence[str],
    query_indices: np.ndarray,
    reference_indices: np.ndarray,
    num_classes: int,
    class_counts: np.ndarray,
    k: int,
    temperature: float,
    batch_size: int,
    label_support: np.ndarray,
    top_label: np.ndarray,
    top_support: np.ndarray,
) -> None:
    if len(reference_indices) == 0:
        raise ValueError("V8 neighborhood reference pool is empty")
    device = embeddings.device
    reference = embeddings[torch.as_tensor(reference_indices, device=device)]
    reference_labels = torch.as_tensor(labels[reference_indices], dtype=torch.long, device=device)
    class_weights = torch.as_tensor(
        1.0 / np.sqrt(np.maximum(class_counts, 5.0)), dtype=torch.float32, device=device
    )
    hash_positions: dict[str, list[int]] = {}
    for position, row in enumerate(reference_indices):
        hash_positions.setdefault(hashes[int(row)], []).append(position)
    for start in range(0, len(query_indices), batch_size):
        selected = query_indices[start : start + batch_size]
        query = embeddings[torch.as_tensor(selected, device=device)]
        # Half-precision products keep the largest query/reference matrix small;
        # voting and class normalization remain in float32.
        scores = (query.to(reference.dtype) @ reference.T).float()
        for local, row in enumerate(selected):
            excluded = hash_positions.get(hashes[int(row)], ())
            if excluded:
                scores[local, excluded] = -torch.inf
        finite = torch.isfinite(scores).sum(dim=1)
        if int(finite.min()) < 1:
            raise ValueError("a V8 query has no nonduplicate neighbors")
        take = min(k, int(finite.min()))
        similarities, positions = scores.topk(take, dim=1, sorted=True)
        neighbor_labels = reference_labels[positions]
        vote = torch.exp((similarities - similarities[:, :1]) / temperature)
        vote *= class_weights[neighbor_labels]
        posterior = torch.zeros((len(selected), num_classes), device=device)
        posterior.scatter_add_(1, neighbor_labels, vote)
        posterior /= posterior.sum(dim=1, keepdim=True).clamp_min(1e-12)
        query_labels = torch.as_tensor(labels[selected], dtype=torch.long, device=device)
        label_support[selected] = posterior.gather(1, query_labels[:, None]).squeeze(1).cpu().numpy()
        support, best = posterior.max(dim=1)
        top_label[selected] = best.cpu().numpy()
        top_support[selected] = support.cpu().numpy()


def _percentiles(
    label_support: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    result = np.zeros(len(labels), dtype=np.float32)
    for label in range(num_classes):
        members = train_indices[labels[train_indices] == label]
        if len(members) == 0:
            continue
        values = label_support[members]
        order = np.argsort(values, kind="stable")
        result[members[order]] = (np.arange(len(members)) + 0.5) / len(members)
        held_out = validation_indices[labels[validation_indices] == label]
        if len(held_out):
            sorted_values = np.sort(values)
            result[held_out] = np.searchsorted(
                sorted_values, label_support[held_out], side="right"
            ) / len(sorted_values)
    return result


def build_neighbor_evidence(
    view_features: np.ndarray,
    labels: np.ndarray,
    hashes: Sequence[str],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    num_classes: int,
    device: torch.device,
    k: int = NEIGHBORS,
    temperature: float = TEMPERATURE,
    batch_size: int = 256,
) -> NeighborEvidence:
    """Compute training scores out of fold, never using validation as reference."""
    labels = np.asarray(labels, dtype=np.int64)
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    if len(hashes) != len(labels) or view_features.shape[1] != len(labels):
        raise ValueError("V8 neighborhood rows do not match the manifest")
    if len(train) == 0 or len(set(train) & set(validation)):
        raise ValueError("V8 train/validation indices are empty or overlap")
    if k < 1 or temperature <= 0 or batch_size < 1:
        raise ValueError("invalid V8 neighborhood settings")
    folds = min(FOLDS, len(train))
    fold_ids = np.full(len(labels), -1, dtype=np.int16)
    fold_ids[train] = strict_stratified_folds(labels[train], folds=folds, seed=SEED)
    class_counts = np.bincount(labels[train], minlength=num_classes).astype(np.float32)
    label_support = np.zeros(len(labels), dtype=np.float32)
    top_label = np.full(len(labels), -1, dtype=np.int32)
    top_support = np.zeros(len(labels), dtype=np.float32)
    embeddings = _embeddings(view_features, device)
    if device.type == "cuda":
        embeddings = embeddings.half()
    for fold in range(folds):
        query = train[fold_ids[train] == fold]
        reference = train[fold_ids[train] != fold]
        _query_pool(
            embeddings, labels, hashes, query, reference, num_classes, class_counts,
            k, temperature, batch_size, label_support, top_label, top_support,
        )
        print(f"v8_neighbors fold={fold + 1}/{folds} queries={len(query)}", flush=True)
    if len(validation):
        _query_pool(
            embeddings, labels, hashes, validation, train, num_classes, class_counts,
            k, temperature, batch_size, label_support, top_label, top_support,
        )
    percentile = _percentiles(label_support, labels, train, validation, num_classes)
    return NeighborEvidence(label_support, top_label, top_support, percentile, fold_ids)


def load_or_create_neighbor_evidence(
    path: Path,
    signature: str,
    view_features: np.ndarray,
    labels: np.ndarray,
    hashes: Sequence[str],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    num_classes: int,
    device: torch.device,
) -> NeighborEvidence:
    metadata = {"signature": signature, "k": NEIGHBORS, "temperature": TEMPERATURE, "folds": FOLDS}
    if path.is_file():
        with np.load(path, allow_pickle=False) as cache:
            if json.loads(str(cache["metadata"])) != metadata:
                raise ValueError("V8 neighborhood cache does not match this dataset/configuration")
            result = NeighborEvidence(*(np.asarray(cache[name]).copy() for name in (
                "label_support", "top_label", "top_support", "percentile", "fold_ids"
            )))
        if len(result.label_support) != len(labels):
            raise ValueError("V8 neighborhood cache row count does not match")
        return result
    result = build_neighbor_evidence(
        view_features, labels, hashes, train_indices, validation_indices, num_classes, device
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        metadata=json.dumps(metadata, sort_keys=True),
        label_support=result.label_support,
        top_label=result.top_label,
        top_support=result.top_support,
        percentile=result.percentile,
        fold_ids=result.fold_ids,
    )
    temporary.replace(path)
    return result


def blend_quality(base: np.ndarray, evidence: NeighborEvidence, ambiguous: np.ndarray) -> np.ndarray:
    if len(base) != len(evidence.percentile) or len(base) != len(ambiguous):
        raise ValueError("V8 quality arrays have different lengths")
    quality = np.clip(0.75 * np.asarray(base) + 0.25 * evidence.percentile, 0.05, 1.0)
    quality[np.asarray(ambiguous, dtype=np.bool_)] = 0.25
    return quality.astype(np.float32)


def trusted_mask(
    quality: np.ndarray,
    prototype_label: np.ndarray,
    labels: np.ndarray,
    evidence: NeighborEvidence,
    ambiguous: np.ndarray,
) -> np.ndarray:
    neighbor_agrees = (evidence.top_label == labels) & (evidence.label_support >= 0.5)
    return (
        (quality >= 0.70)
        & ((prototype_label == labels) | neighbor_agrees)
        & ~np.asarray(ambiguous, dtype=np.bool_)
    )
