"""Deterministic dataset auditing and manifest loading for AIC V7."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps

from robust_clip import list_images
from v6_core import capped_stratified_split


MANIFEST_VERSION = 1
SEED = 2026


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_decode(path: Path) -> tuple[str, str | None]:
    """Return ``ok``, ``exif_fallback`` or ``unreadable`` without changing the file."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as image:
                value = ImageOps.exif_transpose(image).convert("RGB")
                value.load()
        return "ok", None
    except Exception as exif_exc:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with Image.open(path) as image:
                    value = image.convert("RGB")
                    value.load()
            return "exif_fallback", str(exif_exc)
        except Exception as decode_exc:
            return "unreadable", str(decode_exc)


def _manifest_digest(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        stable = {
            key: row[key]
            for key in (
                "relative_path",
                "label",
                "class_name",
                "sha256",
                "decode_status",
                "duplicate_group",
                "candidate_labels",
                "role",
                "split",
            )
        }
        digest.update(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(train_dir: str | Path, seed: int = SEED) -> dict[str, Any]:
    root = Path(train_dir).resolve()
    source_rows = list_images(root)
    records: list[dict[str, Any]] = []
    by_hash: dict[str, list[int]] = defaultdict(list)
    for path_text, label, class_name in source_rows:
        path = Path(path_text)
        sha256 = file_sha256(path)
        decode_status, decode_error = probe_decode(path)
        record = {
            "relative_path": path.relative_to(root).as_posix(),
            "label": int(label),
            "class_name": str(class_name),
            "sha256": sha256,
            "decode_status": decode_status,
            "decode_error": decode_error,
            "duplicate_group": None,
            "candidate_labels": [int(label)],
            "role": "clean",
            "split": None,
        }
        records.append(record)
        by_hash[sha256].append(len(records) - 1)

    duplicate_group = 0
    same_class_groups = 0
    same_class_extra = 0
    cross_class_groups = 0
    cross_class_files = 0
    for sha256 in sorted(by_hash):
        members = by_hash[sha256]
        if len(members) == 1:
            continue
        group_name = f"sha256:{duplicate_group:06d}"
        duplicate_group += 1
        members.sort(key=lambda index: records[index]["relative_path"])
        labels = sorted({int(records[index]["label"]) for index in members})
        for index in members:
            records[index]["duplicate_group"] = group_name
        if len(labels) == 1:
            same_class_groups += 1
            same_class_extra += len(members) - 1
            for index in members[1:]:
                records[index]["role"] = "drop_same_label_duplicate"
        else:
            cross_class_groups += 1
            cross_class_files += len(members)
            representative = members[0]
            records[representative]["role"] = "ambiguous"
            records[representative]["candidate_labels"] = labels
            for index in members[1:]:
                records[index]["role"] = "drop_cross_label_duplicate"
                records[index]["candidate_labels"] = labels

    for record in records:
        if record["decode_status"] == "unreadable":
            record["role"] = "unreadable"
            record["split"] = "excluded"

    clean_record_ids = [index for index, row in enumerate(records) if row["role"] == "clean"]
    clean_rows = [
        (str(root / records[index]["relative_path"]), int(records[index]["label"]), records[index]["class_name"])
        for index in clean_record_ids
    ]
    clean_train, clean_validation = capped_stratified_split(clean_rows, seed)
    for local_index in clean_train:
        records[clean_record_ids[local_index]]["split"] = "train"
    for local_index in clean_validation:
        records[clean_record_ids[local_index]]["split"] = "validation"
    for record in records:
        if record["role"] == "ambiguous":
            record["split"] = "train"
        elif record["split"] is None:
            record["split"] = "excluded"

    clean_counts = Counter(
        int(row["label"]) for row in records if row["role"] == "clean"
    )
    class_names = [name for _, name in sorted({(label, name) for _, label, name in source_rows})]
    summary = {
        "raw_files": len(records),
        "classes": len(class_names),
        "duplicate_groups": duplicate_group,
        "duplicate_extra_files": sum(len(value) - 1 for value in by_hash.values() if len(value) > 1),
        "same_class_groups": same_class_groups,
        "same_class_extra_files": same_class_extra,
        "cross_class_groups": cross_class_groups,
        "cross_class_files": cross_class_files,
        "clean_unique_files": sum(row["role"] == "clean" for row in records),
        "ambiguous_representatives": sum(row["role"] == "ambiguous" for row in records),
        "unreadable_files": sum(row["role"] == "unreadable" for row in records),
        "clean_train_files": sum(
            row["role"] == "clean" and row["split"] == "train" for row in records
        ),
        "clean_validation_files": sum(
            row["role"] == "clean" and row["split"] == "validation" for row in records
        ),
        "clean_class_count_min": min(clean_counts.values()) if clean_counts else 0,
        "clean_class_count_median": float(np.median(list(clean_counts.values()))) if clean_counts else 0.0,
        "clean_class_count_max": max(clean_counts.values()) if clean_counts else 0,
        "classes_without_clean_rows": sorted(set(range(len(class_names))) - set(clean_counts)),
    }
    payload = {
        "format_version": MANIFEST_VERSION,
        "seed": int(seed),
        "train_root": str(root),
        "class_names": class_names,
        "summary": summary,
        "rows": records,
    }
    payload["dataset_signature"] = _manifest_digest(records)
    return payload


def write_manifest(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def read_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != MANIFEST_VERSION:
        raise ValueError("unsupported V7 data manifest version")
    if payload.get("dataset_signature") != _manifest_digest(payload.get("rows", [])):
        raise ValueError("V7 data manifest signature does not match its rows")
    return payload


@dataclass
class ManifestDataset:
    rows: list[tuple[str, int, str]]
    class_names: list[str]
    clean_mask: np.ndarray
    ambiguous_mask: np.ndarray
    candidate_labels: list[tuple[int, ...]]
    train_indices: list[int]
    validation_indices: list[int]
    final_indices: list[int]
    signature: str
    summary: dict[str, Any]


def load_manifest_dataset(
    manifest_path: str | Path,
    train_dir: str | Path,
    conflict_policy: str,
) -> ManifestDataset:
    if conflict_policy not in {"drop", "partial"}:
        raise ValueError("conflict_policy must be 'drop' or 'partial'")
    payload = read_manifest(manifest_path)
    root = Path(train_dir).resolve()
    selected: list[dict[str, Any]] = []
    for row in payload["rows"]:
        if row["role"] == "clean" or (conflict_policy == "partial" and row["role"] == "ambiguous"):
            selected.append(row)
    selected.sort(key=lambda row: row["relative_path"])
    rows = [
        (str(root / row["relative_path"]), int(row["label"]), str(row["class_name"]))
        for row in selected
    ]
    for path, _, _ in rows:
        if not Path(path).is_file():
            raise FileNotFoundError(f"manifest row is missing from train-dir: {path}")
    clean_mask = np.asarray([row["role"] == "clean" for row in selected], dtype=np.bool_)
    ambiguous_mask = np.asarray([row["role"] == "ambiguous" for row in selected], dtype=np.bool_)
    candidates = [tuple(int(value) for value in row["candidate_labels"]) for row in selected]
    train_indices = [
        index
        for index, row in enumerate(selected)
        if row["split"] == "train" and (row["role"] == "clean" or conflict_policy == "partial")
    ]
    validation_indices = [
        index for index, row in enumerate(selected) if row["role"] == "clean" and row["split"] == "validation"
    ]
    final_indices = [index for index, row in enumerate(selected) if row["role"] in {"clean", "ambiguous"}]
    digest = hashlib.sha256()
    digest.update(str(payload["dataset_signature"]).encode("ascii"))
    digest.update(b"\0")
    digest.update(conflict_policy.encode("ascii"))
    return ManifestDataset(
        rows=rows,
        class_names=[str(value) for value in payload["class_names"]],
        clean_mask=clean_mask,
        ambiguous_mask=ambiguous_mask,
        candidate_labels=candidates,
        train_indices=train_indices,
        validation_indices=validation_indices,
        final_indices=final_indices,
        signature=digest.hexdigest(),
        summary=dict(payload["summary"]),
    )


def assert_no_validation_hash_overlap(manifest: dict[str, Any]) -> None:
    validation = {
        row["sha256"] for row in manifest["rows"] if row["role"] == "clean" and row["split"] == "validation"
    }
    training = {
        row["sha256"] for row in manifest["rows"] if row["split"] == "train" and row["role"] in {"clean", "ambiguous"}
    }
    overlap = validation & training
    if overlap:
        raise ValueError(f"training and validation share {len(overlap)} content hashes")


def effective_repeat_factors(
    labels: np.ndarray,
    clean_mask: np.ndarray,
    indices: Iterable[int],
    cap: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    clean_mask = np.asarray(clean_mask, dtype=np.bool_)
    selected = np.asarray(list(indices), dtype=np.int64)
    clean = selected[clean_mask[selected]]
    class_count = int(labels.max()) + 1
    counts = np.bincount(labels[clean], minlength=class_count).astype(np.float64)
    positive = counts[counts > 0]
    median = float(np.median(positive)) if len(positive) else 1.0
    class_factors = np.ones(class_count, dtype=np.float64)
    present = counts > 0
    class_factors[present] = np.minimum(cap, np.sqrt(median / counts[present]))
    class_factors = np.maximum(class_factors, 1.0)
    row_factors = np.ones(len(labels), dtype=np.float64)
    row_factors[clean] = class_factors[labels[clean]]
    effective_counts = counts * class_factors
    return row_factors.astype(np.float64), effective_counts.astype(np.float64)
