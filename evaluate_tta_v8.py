"""Nested-fold TTA calibration of the selected V8 validation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_tta_v7 import collect_four_view_logits
from robust_clip import resolve_device
from train_v5 import atomic_torch_save
from train_v8 import FORMAT_VERSION
from v6_core import head_mid_tail_accuracy
from v7_data import load_manifest_dataset
from v7_views import VIEW_NAMES
from v8_calibration import nested_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--model-dir", default="clip-ViT-B-32")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("checkpoint is not V8")
    if checkpoint.get("training_stage") != "validation":
        raise ValueError("V8 calibration requires the selected validation checkpoint")
    policy = checkpoint["config"]["conflict_policy"]
    manifest = load_manifest_dataset(args.data_manifest, args.train_dir, policy)
    if checkpoint.get("dataset_signature") != manifest.signature:
        raise ValueError("V8 checkpoint and dataset manifest differ")
    indices = np.asarray(checkpoint["validation"]["indices"], dtype=np.int64)
    if indices.tolist() != manifest.validation_indices:
        raise ValueError("V8 checkpoint validation split differs from manifest")
    expected_labels = np.asarray([manifest.rows[index][1] for index in indices], dtype=np.int64)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.train_dir).resolve()
    row_keys = [Path(manifest.rows[index][0]).resolve().relative_to(root).as_posix() for index in indices]
    (output_dir / "val_row_keys.json").write_text(
        json.dumps(row_keys, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    identity = {"checkpoint_sha256": _sha256(checkpoint_path), "dataset_signature": manifest.signature}
    cache_meta = output_dir / "val_logits_cache.json"
    cache_paths = {name: output_dir / f"val_{name}_logits.npy" for name in VIEW_NAMES}
    label_path = output_dir / "val_labels.npy"
    valid_cache = (
        not args.force and cache_meta.is_file() and label_path.is_file()
        and all(path.is_file() for path in cache_paths.values())
        and json.loads(cache_meta.read_text(encoding="utf-8")) == identity
    )
    if valid_cache:
        views = {name: np.load(path) for name, path in cache_paths.items()}
        labels = np.load(label_path)
    else:
        views, labels = collect_four_view_logits(
            checkpoint, manifest.rows, indices, args.model_dir, resolve_device(args.device),
            args.batch_size, args.workers,
        )
        for name, values in views.items():
            np.save(cache_paths[name], values)
        np.save(label_path, labels)
        cache_meta.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    if not np.array_equal(labels, expected_labels):
        raise ValueError("V8 validation logits have incorrect row labels/order")
    expected_shape = (len(indices), len(checkpoint["class_names"]))
    for name, values in views.items():
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"V8 {name} logits are invalid")
    result = nested_calibration(views, labels, resolve_device(args.device))
    np.save(output_dir / "val_oof_logits.npy", result.oof_logits)
    np.save(output_dir / "val_selected_logits.npy", result.full_logits)
    clean_train = np.asarray(
        [index for index in manifest.train_indices if manifest.clean_mask[index]], dtype=np.int64
    )
    row_labels = np.asarray([row[1] for row in manifest.rows], dtype=np.int64)
    counts = np.bincount(row_labels[clean_train], minlength=len(manifest.class_names))
    oof_groups = head_mid_tail_accuracy(
        torch.as_tensor(result.oof_logits), torch.as_tensor(labels), counts
    )
    result.summary["selected_outer_metrics"].update(
        {f"{key}_accuracy": value for key, value in oof_groups.items()}
    )
    full_groups = head_mid_tail_accuracy(
        torch.as_tensor(result.full_logits), torch.as_tensor(labels), counts
    )
    result.summary["selected_full_validation"].update(
        {f"{key}_accuracy": value for key, value in full_groups.items()}
    )
    result.summary.update({
        "validation_rows": len(indices),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "dataset_signature": manifest.signature,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "config": checkpoint["config"],
    })
    (output_dir / "calibration.json").write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )
    checkpoint["class_bias"] = torch.from_numpy(result.class_bias)
    checkpoint["calibration"] = {
        "method": result.summary["method"],
        "selected_family": result.summary["selected_family"],
        "view_weights": result.summary["selected"]["view_weights"],
        "alpha": result.summary["selected"]["alpha"],
        "outer_macro_accuracy": result.summary["selected_outer_metrics"]["macro_accuracy"],
        "selected_epoch": int(checkpoint["selected_epoch"]),
    }
    output_checkpoint = Path(args.output_checkpoint or output_dir / "model.pt")
    atomic_torch_save(checkpoint, output_checkpoint)
    print(json.dumps({
        "selected_family": result.summary["selected_family"],
        "outer_metrics": result.summary["selected_outer_metrics"],
        "full_metrics": result.summary["selected_full_validation"],
        "checkpoint": str(output_checkpoint.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
