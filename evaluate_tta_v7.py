"""Four-view greedy macro-CV calibration for one V7 single-model run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibrate import class_bias_from_logits
from evaluate_tta_v6 import _grid, _search_mix_macro, macro_metrics, strict_stratified_folds
from robust_clip import (
    RobustCLIPClassifier,
    inject_lora_from_config,
    load_classifier_state,
    load_clip,
    resolve_device,
)
from train_v5 import atomic_torch_save
from train_v6 import validate_clip_vit_b32
from train_v7 import FORMAT_VERSION, SEED
from v6_core import head_mid_tail_accuracy
from v7_data import load_manifest_dataset
from v7_views import FourViewFolderDataset, VIEW_NAMES, normalize_view_weights


MIN_VIEW_GAIN = 0.0005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-checkpoint", required=True)
    parser.add_argument("--final-checkpoint", required=True)
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


def build_classifier_from_checkpoint(
    checkpoint: dict[str, Any], model_dir: str, device: torch.device
) -> tuple[RobustCLIPClassifier, Any]:
    model, processor = load_clip(model_dir, device)
    validate_clip_vit_b32(model)
    config = checkpoint.get("config", {})
    class_names = list(checkpoint["class_names"])
    classifier = RobustCLIPClassifier(
        model,
        classifier_init=torch.randn(
            len(class_names), int(model.config.projection_dim), device=device
        ),
        bottleneck=int(config.get("bottleneck", 128)),
        initial_logit_scale=float(config.get("initial_logit_scale", 30.0)),
        max_logit_scale=float(config.get("max_logit_scale", 50.0)),
        learn_logit_scale=bool(config.get("learn_logit_scale", False)),
    ).to(device)
    replaced = inject_lora_from_config(classifier.clip, config)
    if len(replaced) != 8:
        raise ValueError(f"V7 checkpoint must recreate 8 Q/V LoRA modules, got {len(replaced)}")
    load_classifier_state(classifier, checkpoint["model"])
    classifier.eval()
    return classifier, processor


@torch.no_grad()
def collect_four_view_logits(
    checkpoint: dict[str, Any],
    rows,
    validation_indices: Sequence[int],
    model_dir: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    classifier, processor = build_classifier_from_checkpoint(checkpoint, model_dir, device)
    loader = DataLoader(
        FourViewFolderDataset(rows, validation_indices, processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_NAMES}
    labels: list[np.ndarray] = []
    for batch_id, batch in enumerate(loader, 1):
        center = batch["center"].to(device, non_blocking=True)
        zoom = batch["zoom256"].to(device, non_blocking=True)
        tensors = {
            "center": center,
            "hflip": torch.flip(center, dims=[3]),
            "zoom256": zoom,
            "zoom256_hflip": torch.flip(zoom, dims=[3]),
        }
        for name, pixels in tensors.items():
            collected[name].append(classifier(pixels, None)[0].float().cpu().numpy())
        labels.append(batch["label"].numpy())
        if batch_id % 40 == 0:
            print(f"v7_tta_batches={batch_id}/{len(loader)}", flush=True)
    return (
        {name: np.concatenate(parts, axis=0) for name, parts in collected.items()},
        np.concatenate(labels),
    )


def greedy_view_search(
    logits_by_view: dict[str, np.ndarray],
    labels: np.ndarray,
    fold_ids: np.ndarray,
    device: torch.device,
    minimum_gain: float = MIN_VIEW_GAIN,
) -> dict[str, Any]:
    """Greedily add non-negative views; each pairwise mix uses a 0.05 grid."""
    if set(logits_by_view) != set(VIEW_NAMES) or len(logits_by_view) != len(VIEW_NAMES):
        missing = sorted(set(VIEW_NAMES) - set(logits_by_view))
        extra = sorted(set(logits_by_view) - set(VIEW_NAMES))
        raise ValueError(f"invalid V7 TTA views; missing={missing}, extra={extra}")
    shapes = {tuple(value.shape) for value in logits_by_view.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] != len(labels):
        raise ValueError("V7 TTA logits have inconsistent shapes")
    weights_grid = _grid(1.0, 0.05)
    alphas = _grid(2.0, 0.05)
    current = np.asarray(logits_by_view["center"], dtype=np.float32)
    weights = {"center": 1.0}
    selected = _search_mix_macro(
        current, current, labels, fold_ids, np.asarray([1.0], dtype=np.float32), alphas, device
    )
    history: list[dict[str, Any]] = [
        {
            "added_view": "center",
            "view_weights": dict(weights),
            "cv_macro_accuracy": selected["cv_macro_accuracy"],
            "cv_macro_nll": selected["cv_macro_nll"],
            "alpha": selected["alpha"],
        }
    ]
    remaining = list(VIEW_NAMES[1:])
    while remaining:
        best: tuple[tuple[float, float, int], str, dict[str, float]] | None = None
        for order, name in enumerate(remaining):
            candidate = _search_mix_macro(
                current,
                logits_by_view[name],
                labels,
                fold_ids,
                weights_grid,
                alphas,
                device,
            )
            key = (
                float(candidate["cv_macro_accuracy"]),
                -float(candidate["cv_macro_nll"]),
                -order,
            )
            if best is None or key > best[0]:
                best = (key, name, candidate)
        assert best is not None
        _, name, candidate = best
        gain = float(candidate["cv_macro_accuracy"]) - float(selected["cv_macro_accuracy"])
        if gain + 1e-12 < float(minimum_gain):
            break
        old_weight = float(candidate["original_weight"])
        current = old_weight * current + (1.0 - old_weight) * logits_by_view[name]
        weights = {key: value * old_weight for key, value in weights.items()}
        weights[name] = 1.0 - old_weight
        weights = normalize_view_weights(weights)
        selected = candidate
        history.append(
            {
                "added_view": name,
                "gain": gain,
                "pairwise_existing_weight": old_weight,
                "view_weights": dict(weights),
                "cv_macro_accuracy": selected["cv_macro_accuracy"],
                "cv_macro_nll": selected["cv_macro_nll"],
                "alpha": selected["alpha"],
            }
        )
        remaining.remove(name)
    return {
        "view_weights": normalize_view_weights(weights),
        "alpha": float(selected["alpha"]),
        "cv_macro_accuracy": float(selected["cv_macro_accuracy"]),
        "cv_macro_nll": float(selected["cv_macro_nll"]),
        "history": history,
        "mixed_logits": current,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    device = resolve_device(args.device)
    validation_checkpoint = torch.load(args.validation_checkpoint, map_location="cpu")
    final_checkpoint = torch.load(args.final_checkpoint, map_location="cpu")
    for name, checkpoint, stage in (
        ("validation", validation_checkpoint, "validation"),
        ("final", final_checkpoint, "final"),
    ):
        if checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{name} checkpoint is not V7")
        if checkpoint.get("training_stage") != stage:
            raise ValueError(f"{name} checkpoint has the wrong training stage")
    if validation_checkpoint["class_names"] != final_checkpoint["class_names"]:
        raise ValueError("validation and final checkpoints have different classes")
    config = validation_checkpoint.get("config", {})
    policy = str(config.get("conflict_policy", "drop"))
    manifest = load_manifest_dataset(args.data_manifest, args.train_dir, policy)
    for checkpoint in (validation_checkpoint, final_checkpoint):
        if checkpoint.get("dataset_signature") != manifest.signature:
            raise ValueError("manifest/train-dir does not match the V7 checkpoint")
    validation_indices = [int(value) for value in validation_checkpoint["validation"]["indices"]]
    expected_labels = np.asarray(
        [manifest.rows[index][1] for index in validation_indices], dtype=np.int64
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_root = Path(args.train_dir).resolve()
    validation_row_keys = [
        Path(manifest.rows[index][0]).resolve().relative_to(train_root).as_posix()
        for index in validation_indices
    ]
    (output_dir / "val_row_keys.json").write_text(
        json.dumps(validation_row_keys, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cache_paths = {name: output_dir / f"val_{name}_logits.npy" for name in VIEW_NAMES}
    label_path = output_dir / "val_labels.npy"
    use_cache = not args.force and label_path.is_file() and all(path.is_file() for path in cache_paths.values())
    if use_cache:
        logits_by_view = {name: np.load(path) for name, path in cache_paths.items()}
        labels = np.load(label_path)
    else:
        logits_by_view, labels = collect_four_view_logits(
            validation_checkpoint,
            manifest.rows,
            validation_indices,
            args.model_dir,
            device,
            args.batch_size,
            args.workers,
        )
        for name, values in logits_by_view.items():
            np.save(cache_paths[name], values)
        np.save(label_path, labels)
    expected_shape = (len(validation_indices), len(validation_checkpoint["class_names"]))
    if not np.array_equal(labels, expected_labels):
        raise ValueError("cached validation labels do not match the V7 manifest")
    for name, values in logits_by_view.items():
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"{name} validation logits are invalid: {values.shape}")

    fold_ids = strict_stratified_folds(labels, folds=5, seed=SEED)
    selection = greedy_view_search(logits_by_view, labels, fold_ids, device)
    mixed = selection.pop("mixed_logits")
    raw_bias = class_bias_from_logits(torch.as_tensor(mixed, dtype=torch.float32)).numpy()
    class_bias = selection["alpha"] * raw_bias
    calibrated = mixed + class_bias[None, :]
    np.save(output_dir / "val_selected_logits.npy", calibrated.astype(np.float32))

    clean_train = [
        index for index in manifest.train_indices if bool(manifest.clean_mask[index])
    ]
    train_labels = np.asarray([row[1] for row in manifest.rows], dtype=np.int64)
    class_counts = np.bincount(
        train_labels[np.asarray(clean_train)], minlength=len(manifest.class_names)
    )
    hmt = head_mid_tail_accuracy(
        torch.as_tensor(calibrated), torch.as_tensor(labels), class_counts
    )
    trusted_global = set(validation_checkpoint["validation"].get("trusted_indices", []))
    trusted_local = np.asarray([index in trusted_global for index in validation_indices])
    result: dict[str, Any] = {
        "method": "v7_greedy_four_view_strict_5fold_macro_and_marginal_bias",
        "minimum_view_gain": MIN_VIEW_GAIN,
        "selected": selection,
        "per_view": {name: macro_metrics(values, labels) for name, values in logits_by_view.items()},
        "selected_full_validation": {
            **macro_metrics(calibrated, labels),
            **{f"{name}_accuracy": value for name, value in hmt.items()},
        },
        "trusted_selected_full_validation": (
            macro_metrics(calibrated[trusted_local], labels[trusted_local])
            if trusted_local.any()
            else None
        ),
        "grid": {
            "folds": 5,
            "pairwise_view_weight_step": 0.05,
            "class_bias_alpha": {"minimum": 0.0, "maximum": 2.0, "step": 0.05},
        },
        "validation_rows": len(validation_indices),
        "trusted_validation_rows": int(trusted_local.sum()),
        "selected_epoch": int(validation_checkpoint["selected_epoch"]),
        "fold_ids": fold_ids.tolist(),
        "dataset_signature": manifest.signature,
        "config": config,
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    final_checkpoint["class_bias"] = torch.from_numpy(class_bias.astype(np.float32))
    final_checkpoint["calibration"] = {
        "method": result["method"],
        "view_weights": selection["view_weights"],
        "alpha": selection["alpha"],
        "folds": 5,
        "cv_macro_accuracy": selection["cv_macro_accuracy"],
        "cv_macro_nll": selection["cv_macro_nll"],
        "minimum_view_gain": MIN_VIEW_GAIN,
        "selected_epoch": int(validation_checkpoint["selected_epoch"]),
    }
    output_checkpoint = Path(args.output_checkpoint or (output_dir / "model.pt"))
    atomic_torch_save(final_checkpoint, output_checkpoint)
    print(json.dumps(result, indent=2), flush=True)
    print(f"calibrated_v7_checkpoint={output_checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
