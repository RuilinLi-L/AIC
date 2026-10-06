"""Strict 5-fold validation for a V9/V13 run; calibrate one selected checkpoint.

Epoch selection and calibration use only native validation images. Fixed image
quality stresses are scored with the native-fitted parameters unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from v13_model import (
    RESOLUTION_VERSION, build_classifier_from_checkpoint, build_zoom_transform, checkpoint_resolution,
)
from robust_clip import _clip_image_transform, load_classifier_state, resolve_device, safe_open_image
from train_v5 import atomic_torch_save
from v10_evaluation import CONDITIONS, StrictEvaluation, strict_nested_evaluation
from v7_data import load_manifest_dataset
from v7_views import VIEW_NAMES
from v13_runtime import autocast_context, loader_options, precision_name, Progress
from v13_core import model_identity


STRESS_VERSION = "v10_resize384_bicubic_no_upscale_jpeg75_rgb_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="V9 selected run or V13 training directory")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--model-dir", default="clip-ViT-B-32")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stress_image(image: Image.Image, condition: str) -> Image.Image:
    """Apply one deterministic, validation-only image quality condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown validation stress: {condition}")
    if condition in ("resize384", "combined"):
        width, height = image.size
        longest = max(width, height)
        if longest > 384:
            scale = 384.0 / longest
            target = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(target, resample=Image.Resampling.BICUBIC)
    if condition in ("jpeg75", "combined"):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            image = decoded.convert("RGB")
            image.load()
    return image


class StressFourViewDataset(Dataset):
    def __init__(
        self, rows: Sequence[tuple[str, int, str]], indices: Sequence[int], processor,
        condition: str, *, four: bool,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown validation stress: {condition}")
        self.rows = rows
        self.indices = list(indices)
        self.condition = condition
        self.four = four
        self.center_transform = _clip_image_transform(processor, "none")
        self.zoom_transform = build_zoom_transform(processor) if four else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        path, label, _ = self.rows[self.indices[item]]
        image = stress_image(safe_open_image(path), self.condition)
        result = {
            "center": self.center_transform(image),
            "label": torch.tensor(label, dtype=torch.long),
        }
        if self.zoom_transform is not None:
            result["zoom256"] = self.zoom_transform(image)
        return result


@torch.no_grad()
def collect_view_logits(
    classifier, processor, rows, indices: Sequence[int], condition: str,
    *, four: bool, device: torch.device, batch_size: int, workers: int, prefetch_factor: int = 2,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    dataset = StressFourViewDataset(rows, indices, processor, condition, four=four)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=False,
        **loader_options(workers, prefetch_factor),
    )
    names = VIEW_NAMES if four else VIEW_NAMES[:2]
    collected: dict[str, list[np.ndarray]] = {name: [] for name in names}
    labels: list[np.ndarray] = []
    classifier.eval()
    progress = Progress(f"eval_{condition}", device, len(loader))
    for batch_id, batch in enumerate(loader, 1):
        center = batch["center"].to(device, non_blocking=True)
        tensors = {"center": center, "hflip": torch.flip(center, dims=[3])}
        if four:
            zoom = batch["zoom256"].to(device, non_blocking=True)
            tensors.update({"zoom256": zoom, "zoom256_hflip": torch.flip(zoom, dims=[3])})
        for name in names:
            with autocast_context(device):
                logits = classifier(tensors[name], None)[0]
            collected[name].append(logits.float().cpu().numpy())
        labels.append(batch["label"].numpy())
        progress.update(batch_id, len(center))
    return {name: np.concatenate(parts) for name, parts in collected.items()}, np.concatenate(labels)


def validate_epoch_checkpoints(
    run_dir: Path, manifest, expected_labels: np.ndarray,
) -> tuple[dict[int, Path], dict[int, dict[str, Any]]]:
    first = torch.load(run_dir / "validation_epochs" / "epoch_01.pt", map_location="cpu", weights_only=False)
    count = int(first["config"]["epochs"])
    if first.get("format_version") == 13 and (first["config"].get("stage") != "validate" or count != 24):
        raise ValueError("V13 evaluation requires a complete 24-epoch validation run")
    paths = {epoch: run_dir / "validation_epochs" / f"epoch_{epoch:02d}.pt" for epoch in range(1, count + 1)}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V13 strict evaluation requires all configured epochs; missing {missing}")
    checkpoints = {
        epoch: torch.load(path, map_location="cpu", weights_only=False)
        for epoch, path in paths.items()
    }
    reference = checkpoints[1]
    reference_size = checkpoint_resolution(reference)
    reference_indices = list(reference["validation"]["indices"])
    if reference_indices != manifest.validation_indices:
        raise ValueError("validation split differs from manifest")
    if not np.array_equal(
        expected_labels,
        np.asarray([manifest.rows[index][1] for index in reference_indices], dtype=np.int64),
    ):
        raise ValueError("validation labels differ from manifest")
    for epoch, checkpoint in checkpoints.items():
        if checkpoint["config"] != reference["config"]:
            raise ValueError("mixed training configurations across epoch checkpoints")
        if checkpoint_resolution(checkpoint) != reference_size:
            raise ValueError("mixed input resolutions across epoch checkpoints")
        if checkpoint.get("format_version") not in (9, 11, 12, 13):
            raise ValueError(f"epoch {epoch} checkpoint is not a supported validation model")
        if checkpoint.get("format_version") != reference.get("format_version"):
            raise ValueError("mixed V9/V13 epoch checkpoints")
        if checkpoint.get("training_stage") != "validation":
            raise ValueError(f"epoch {epoch} checkpoint is not validation-stage")
        if int(checkpoint.get("selected_epoch", -1)) != epoch:
            raise ValueError(f"epoch {epoch} checkpoint has wrong epoch metadata")
        if checkpoint.get("dataset_signature") != manifest.signature:
            raise ValueError(f"epoch {epoch} checkpoint dataset signature differs")
        if list(checkpoint["validation"]["indices"]) != reference_indices:
            raise ValueError(f"epoch {epoch} checkpoint validation split differs")
        if list(checkpoint["class_names"]) != list(manifest.class_names):
            raise ValueError(f"epoch {epoch} checkpoint classes differ")
        if checkpoint["config"]["conflict_policy"] != reference["config"]["conflict_policy"]:
            raise ValueError(f"epoch {epoch} checkpoint conflict policy differs")
    return paths, checkpoints


class LogitCache:
    def __init__(
        self, *, output_dir: Path, paths: Mapping[int, Path],
        checkpoints: Mapping[int, dict[str, Any]], manifest, indices: Sequence[int],
        labels: np.ndarray, model_dir: str, device: torch.device,
        batch_size: int, workers: int, force: bool, prefetch_factor: int = 2,
    ) -> None:
        self.cache_dir = output_dir / "logit_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths = paths
        self.checkpoints = checkpoints
        self.manifest = manifest
        self.indices = list(indices)
        self.labels = labels
        self.device = device
        self.batch_size = batch_size
        self.workers = workers
        self.prefetch_factor = prefetch_factor
        self.force = force
        self.model_dir = str(Path(model_dir).resolve())
        self.hashes = {epoch: sha256_file(path) for epoch, path in paths.items()}
        self.row_digest = hashlib.sha256(
            np.asarray(self.indices, dtype="<i8").tobytes()
        ).hexdigest()
        self.classifier = None
        self.processor = None
        self.loaded_epoch: int | None = None
        self.memory: dict[tuple[int, str, bool], dict[str, np.ndarray]] = {}

    def _identity(self, epoch: int, condition: str, four: bool) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.hashes[epoch],
            "format_version": self.checkpoints[epoch]["format_version"],
            "precision": precision_name(self.device),
            "dataset_signature": self.manifest.signature,
            "validation_indices_sha256": self.row_digest,
            "model_dir": self.model_dir,
            "stress_version": STRESS_VERSION,
            "preprocessing_version": RESOLUTION_VERSION,
            "image_size": checkpoint_resolution(self.checkpoints[epoch]),
            "condition": condition,
            "views": list(VIEW_NAMES if four else VIEW_NAMES[:2]),
        }

    def _path(self, epoch: int, condition: str, four: bool) -> Path:
        suffix = "four" if four else "two"
        return self.cache_dir / f"epoch_{epoch:02d}_{self.hashes[epoch][:12]}_{condition}_{suffix}.npz"

    def _validate(self, views: Mapping[str, np.ndarray], four: bool) -> None:
        names = VIEW_NAMES if four else VIEW_NAMES[:2]
        if set(views) != set(names):
            raise ValueError("cached view names differ")
        shape = (len(self.labels), len(self.manifest.class_names))
        for name in names:
            if views[name].shape != shape or not np.isfinite(views[name]).all():
                raise ValueError(f"invalid cached {name} logits")

    def get_views(self, epoch: int, condition: str, four: bool) -> dict[str, np.ndarray]:
        key = (epoch, condition, four)
        if key in self.memory:
            return self.memory[key]
        identity = self._identity(epoch, condition, four)
        path = self._path(epoch, condition, four)
        if path.is_file() and not self.force:
            try:
                with np.load(path, allow_pickle=False) as cache:
                    metadata = json.loads(str(cache["metadata"].item()))
                    if metadata == identity:
                        names = VIEW_NAMES if four else VIEW_NAMES[:2]
                        views = {name: np.asarray(cache[name], dtype=np.float32) for name in names}
                        if np.array_equal(cache["labels"], self.labels):
                            self._validate(views, four)
                            self.memory[key] = views
                            return views
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                pass
        if (
            condition == "native" and not four and not self.force
            and self.checkpoints[epoch].get("format_version") == 13
        ):
            training_logits = self.paths[epoch].with_name(f"epoch_{epoch:02d}_logits.npz")
            if (
                training_logits.is_file()
                and training_logits.stat().st_mtime_ns >= self.paths[epoch].stat().st_mtime_ns
            ):
                with np.load(training_logits, allow_pickle=False) as saved:
                    if (
                        "precision" in saved and str(saved["precision"].item()) == precision_name(self.device)
                        and "checkpoint_sha256" in saved and str(saved["checkpoint_sha256"].item()) == self.hashes[epoch]
                        and "format_version" in saved and int(saved["format_version"].item()) == 13
                        and np.array_equal(saved["labels"], self.labels)
                        and np.array_equal(saved["row_indices"], self.indices)
                    ):
                        views = {
                            "center": np.asarray(saved["center"], dtype=np.float32),
                            "hflip": np.asarray(saved["hflip"], dtype=np.float32),
                        }
                        self._validate(views, False)
                        self.memory[key] = views
                        print(f"reused_training_epoch_logits={epoch}", flush=True)
                        return views
        if self.classifier is None:
            self.classifier, self.processor = build_classifier_from_checkpoint(
                self.checkpoints[epoch], self.model_dir, self.device,
            )
            self.loaded_epoch = epoch
        elif self.loaded_epoch != epoch:
            load_classifier_state(self.classifier, self.checkpoints[epoch]["model"])
            self.loaded_epoch = epoch
        views, labels = collect_view_logits(
            self.classifier, self.processor, self.manifest.rows, self.indices, condition,
            four=four, device=self.device, batch_size=self.batch_size, workers=self.workers,
            prefetch_factor=self.prefetch_factor,
        )
        if not np.array_equal(labels, self.labels):
            raise ValueError("collected validation logits have wrong row labels/order")
        self._validate(views, four)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, metadata=json.dumps(identity, sort_keys=True), labels=labels, **views)
        os.replace(temporary, path)
        self.memory[key] = views
        print(f"cached_epoch={epoch} condition={condition} views={'four' if four else 'two'}", flush=True)
        return views


def write_result(
    result: StrictEvaluation, *, output_dir: Path, output_checkpoint: Path,
    paths: Mapping[int, Path], checkpoints: Mapping[int, dict[str, Any]],
    cache: LogitCache, row_keys: list[str], labels: np.ndarray,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "val_row_keys.json").write_text(
        json.dumps(row_keys, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(output_dir / "val_labels.npy", labels)
    for condition, logits in result.oof_by_condition.items():
        np.save(output_dir / f"val_oof_{condition}.npy", logits)
    np.save(output_dir / "val_selected_native.npy", result.full_native_logits)
    epoch = int(result.summary["selected"]["epoch"])
    selected_checkpoint = dict(checkpoints[epoch])
    selected_checkpoint["class_bias"] = torch.from_numpy(result.class_bias)
    selected_checkpoint["calibration"] = {
        "method": result.summary["method"],
        "selected_family": result.summary["selected"]["family"],
        "view_weights": result.summary["selected"]["view_weights"],
        "alpha": result.summary["selected"]["alpha"],
        "outer_macro_accuracy": result.summary["conditions"]["native"]["outer_metrics"]["macro_accuracy"],
        "selected_epoch": epoch,
        "image_size": checkpoint_resolution(selected_checkpoint),
        "precision": precision_name(cache.device),
    }
    atomic_torch_save(selected_checkpoint, output_checkpoint)
    result.summary["evaluation_scope"] = "noisy holdout; folds select epoch/calibration, not independently trained models"
    result.summary.update({
        "format_version": int(selected_checkpoint["format_version"]),
        "calibrated_checkpoint_sha256": sha256_file(output_checkpoint),
        "recipe": selected_checkpoint["config"].get("recipe", "v12_fallback"),
        "dataset_signature": cache.manifest.signature,
        "validation_rows": int(len(labels)),
        "checkpoint_sha256": cache.hashes[epoch],
        "calibrated_checkpoint": str(output_checkpoint.resolve()),
        "epoch_checkpoint_sha256": {str(number): digest for number, digest in cache.hashes.items()},
        "stress_version": STRESS_VERSION,
        "preprocessing_version": RESOLUTION_VERSION,
        "image_size": checkpoint_resolution(selected_checkpoint),
        "precision": precision_name(cache.device),
    })
    (output_dir / "strict_eval.json").write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )
    return result.summary


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers nonnegative")
    loader_options(args.workers, args.prefetch_factor)
    run_dir = Path(args.run_dir)
    first_path = run_dir / "validation_epochs" / "epoch_01.pt"
    if not first_path.is_file():
        raise FileNotFoundError(f"first epoch checkpoint is missing: {first_path}")
    first_checkpoint = torch.load(first_path, map_location="cpu", weights_only=False)
    if first_checkpoint.get("format_version") == 13 and (
        model_identity(args.model_dir) != first_checkpoint["config"]["base_model_identity"]
    ):
        raise ValueError("base weights changed since training; cached logits cannot be reused")
    if first_checkpoint.get("format_version") not in (9, 11, 12, 13):
        raise ValueError("V13 strict evaluation accepts V9, V11 or V13 validation checkpoints")
    manifest = load_manifest_dataset(
        args.data_manifest, args.train_dir, first_checkpoint["config"]["conflict_policy"]
    )
    indices = np.asarray(first_checkpoint["validation"]["indices"], dtype=np.int64)
    labels = np.asarray([manifest.rows[index][1] for index in indices], dtype=np.int64)
    paths, checkpoints = validate_epoch_checkpoints(run_dir, manifest, labels)
    root = Path(args.train_dir).resolve()
    row_keys = [
        Path(manifest.rows[index][0]).resolve().relative_to(root).as_posix()
        for index in indices
    ]
    clean_train = np.asarray(
        [index for index in manifest.train_indices if manifest.clean_mask[index]], dtype=np.int64
    )
    row_labels = np.asarray([row[1] for row in manifest.rows], dtype=np.int64)
    class_counts = np.bincount(row_labels[clean_train], minlength=len(manifest.class_names))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = LogitCache(
        output_dir=output_dir, paths=paths, checkpoints=checkpoints,
        manifest=manifest, indices=indices, labels=labels, model_dir=args.model_dir,
        device=resolve_device(args.device), batch_size=args.batch_size,
        workers=args.workers, force=args.force, prefetch_factor=args.prefetch_factor,
    )
    result = strict_nested_evaluation(
        cache.get_views, list(paths), labels, class_counts, device=cache.device,
    )
    output_checkpoint = Path(args.output_checkpoint or output_dir / "model.pt")
    summary = write_result(
        result, output_dir=output_dir, output_checkpoint=output_checkpoint,
        paths=paths, checkpoints=checkpoints, cache=cache, row_keys=row_keys, labels=labels,
    )
    print(json.dumps({
        "selected": summary["selected"],
        "conditions": {name: value["outer_metrics"] for name, value in summary["conditions"].items()},
        "checkpoint": str(output_checkpoint.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
