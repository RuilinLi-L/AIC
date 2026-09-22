"""Strict macro-CV calibration for the V6 single-model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from calibrate import class_bias_from_logits
from robust_clip import (
    FolderImageDataset,
    RobustCLIPClassifier,
    inject_lora_from_config,
    list_images,
    load_classifier_state,
    load_clip,
    resolve_device,
)
from train_v5 import atomic_torch_save
from train_v6 import FORMAT_VERSION, SEED, dataset_signature, validate_clip_vit_b32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-checkpoint", required=True)
    parser.add_argument("--final-checkpoint", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--model-dir", default="clip-ViT-B-32")
    parser.add_argument("--original-val-logits", required=True)
    parser.add_argument("--hflip-val-logits", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def strict_stratified_folds(labels: np.ndarray, folds: int = 5, seed: int = SEED) -> np.ndarray:
    """Assign every row to one of exactly ``folds`` class-stratified folds.

    Tail classes with fewer than five validation rows are spread with a
    deterministic random class offset instead of being rejected.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if folds < 2 or len(labels) < folds:
        raise ValueError("strict CV requires at least one row per fold")
    rng = np.random.default_rng(seed)
    result = np.full(len(labels), -1, dtype=np.int16)
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        offset = int(rng.integers(0, folds))
        result[indices] = (np.arange(len(indices)) + offset) % folds
    if set(result.tolist()) != set(range(folds)):
        # This only occurs on very small, extremely sparse validation sets.
        # A deterministic global round-robin keeps the requested five folds.
        indices = np.arange(len(labels))
        rng.shuffle(indices)
        result[indices] = np.arange(len(indices)) % folds
    return result


def _grid(maximum: float, step: float) -> np.ndarray:
    if maximum < 0.0 or step <= 0.0:
        raise ValueError("invalid calibration grid")
    return np.linspace(0.0, maximum, int(round(maximum / step)) + 1, dtype=np.float32)


@torch.no_grad()
def collect_validation_logits(
    checkpoint: dict[str, Any],
    rows: list[tuple[str, int, str]],
    validation_indices: Sequence[int],
    model_dir: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    model, processor = load_clip(model_dir, device)
    validate_clip_vit_b32(model)
    class_names = list(checkpoint["class_names"])
    config = checkpoint.get("config", {})
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
        raise ValueError(f"V6 checkpoint must recreate 8 Q/V LoRA modules, got {len(replaced)}")
    print(f"loaded_lora_modules={len(replaced)}", flush=True)
    load_classifier_state(classifier, checkpoint["model"])
    classifier.eval()
    loader = DataLoader(
        FolderImageDataset(rows, validation_indices, processor, augment="none"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    original, hflip = [], []
    for batch_id, batch in enumerate(loader, 1):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        original.append(classifier(pixels, None)[0].float().cpu().numpy())
        hflip.append(classifier(torch.flip(pixels, dims=[3]), None)[0].float().cpu().numpy())
        if batch_id % 40 == 0:
            print(f"calibration_batches={batch_id}/{len(loader)}", flush=True)
    return np.concatenate(original), np.concatenate(hflip)


def _search_mix_macro(
    original: np.ndarray,
    hflip: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    weights: np.ndarray,
    alphas: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Search macro accuracy, then macro NLL, alpha and original weight."""
    original_t = torch.as_tensor(original, dtype=torch.float32, device=device)
    hflip_t = torch.as_tensor(hflip, dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    folds_t = torch.as_tensor(fold_ids, dtype=torch.long, device=device)
    class_count = int(original.shape[1])
    present = torch.bincount(labels_t, minlength=class_count) > 0
    row_counts = torch.bincount(labels_t, minlength=class_count).double().clamp_min(1.0)
    accuracy_grid = torch.zeros((len(weights), len(alphas)), dtype=torch.float64, device=device)
    nll_grid = torch.zeros_like(accuracy_grid)

    for weight_id, weight in enumerate(weights):
        mixed = float(weight) * original_t + (1.0 - float(weight)) * hflip_t
        correct_by_class = torch.zeros((len(alphas), class_count), dtype=torch.float64, device=device)
        nll_by_class = torch.zeros_like(correct_by_class)
        for fold in range(int(fold_ids.max()) + 1):
            calibration = folds_t != fold
            evaluation = ~calibration
            if not calibration.any() or not evaluation.any():
                raise ValueError(f"fold {fold} has an empty calibration or evaluation partition")
            bias = class_bias_from_logits(mixed[calibration])
            eval_logits = mixed[evaluation]
            eval_labels = labels_t[evaluation]
            one_hot = F.one_hot(eval_labels, num_classes=class_count).double()
            for start in range(0, len(alphas), 8):
                stop = min(len(alphas), start + 8)
                alpha = torch.as_tensor(alphas[start:stop], dtype=torch.float32, device=device)
                adjusted = eval_logits[None, :, :] + alpha[:, None, None] * bias[None, None, :]
                predictions = adjusted.argmax(dim=2)
                matches = predictions.eq(eval_labels[None, :]).double()
                losses = F.cross_entropy(
                    adjusted.permute(0, 2, 1),
                    eval_labels[None, :].expand(stop - start, -1),
                    reduction="none",
                ).double()
                correct_by_class[start:stop] += matches @ one_hot
                nll_by_class[start:stop] += losses @ one_hot
        accuracy_grid[weight_id] = (correct_by_class[:, present] / row_counts[present]).mean(dim=1)
        nll_grid[weight_id] = (nll_by_class[:, present] / row_counts[present]).mean(dim=1)

    best_key: tuple[float, float, float, float] | None = None
    best_ids = (0, 0)
    for weight_id, weight in enumerate(weights):
        for alpha_id, alpha in enumerate(alphas):
            key = (
                float(accuracy_grid[weight_id, alpha_id]),
                -float(nll_grid[weight_id, alpha_id]),
                -float(alpha),
                float(weight),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_ids = weight_id, alpha_id
    assert best_key is not None
    weight_id, alpha_id = best_ids
    return {
        "original_weight": float(weights[weight_id]),
        "hflip_weight": float(1.0 - weights[weight_id]),
        "alpha": float(alphas[alpha_id]),
        "cv_macro_accuracy": float(accuracy_grid[weight_id, alpha_id]),
        "cv_macro_nll": float(nll_grid[weight_id, alpha_id]),
    }


def macro_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    predictions = logits_t.argmax(dim=1)
    losses = F.cross_entropy(logits_t, labels_t, reduction="none")
    accuracies, nlls = [], []
    for label in labels_t.unique(sorted=True):
        selected = labels_t == label
        accuracies.append(predictions[selected].eq(labels_t[selected]).float().mean())
        nlls.append(losses[selected].mean())
    return {
        "macro_accuracy": float(torch.stack(accuracies).mean()),
        "macro_nll": float(torch.stack(nlls).mean()),
        "accuracy": float(predictions.eq(labels_t).float().mean()),
    }


def main() -> None:
    args = parse_args()
    if args.folds != 5:
        raise ValueError("V6 calibration is locked to strict 5-fold selection")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    device = resolve_device(args.device)
    validation_checkpoint = torch.load(args.validation_checkpoint, map_location="cpu")
    final_checkpoint = torch.load(args.final_checkpoint, map_location="cpu")
    if validation_checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("validation checkpoint is not V6")
    if final_checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("final checkpoint is not V6")
    if validation_checkpoint.get("training_stage") != "validation":
        raise ValueError("--validation-checkpoint must be best_model.pt from the validation stage")
    if final_checkpoint.get("training_stage") != "final":
        raise ValueError("--final-checkpoint must be model.pt from the full-data stage")
    if validation_checkpoint["class_names"] != final_checkpoint["class_names"]:
        raise ValueError("validation and final checkpoints have different classes")

    rows = list_images(args.train_dir)
    digest = dataset_signature(rows)
    if validation_checkpoint.get("dataset_signature") != digest:
        raise ValueError("train-dir does not match the validation checkpoint data signature")
    if final_checkpoint.get("dataset_signature") != digest:
        raise ValueError("train-dir does not match the final checkpoint data signature")
    validation_indices = [int(value) for value in validation_checkpoint["validation"]["indices"]]
    all_labels = np.asarray([label for _, label, _ in rows], dtype=np.int64)
    labels = all_labels[np.asarray(validation_indices, dtype=np.int64)]
    expected = (len(validation_indices), len(validation_checkpoint["class_names"]))
    original_path = Path(args.original_val_logits)
    hflip_path = Path(args.hflip_val_logits)
    original = None if args.force or not original_path.is_file() else np.load(original_path)
    hflip = None if args.force or not hflip_path.is_file() else np.load(hflip_path)
    if original is None or hflip is None:
        original, hflip = collect_validation_logits(
            validation_checkpoint,
            rows,
            validation_indices,
            args.model_dir,
            device,
            args.batch_size,
            args.workers,
        )
        original_path.parent.mkdir(parents=True, exist_ok=True)
        hflip_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(original_path, original)
        np.save(hflip_path, hflip)
    if original.shape != expected or hflip.shape != expected:
        raise ValueError(
            f"validation logits must both have shape {expected}; got {original.shape} and {hflip.shape}"
        )
    if not np.isfinite(original).all() or not np.isfinite(hflip).all():
        raise ValueError("validation logits contain non-finite values")

    fold_ids = strict_stratified_folds(labels, folds=5, seed=SEED)
    selected = _search_mix_macro(
        original,
        hflip,
        labels,
        fold_ids,
        _grid(1.0, 0.05),
        _grid(2.0, 0.05),
        device,
    )
    weight = selected["original_weight"]
    mixed = weight * original + (1.0 - weight) * hflip
    raw_bias = class_bias_from_logits(torch.as_tensor(mixed, dtype=torch.float32)).numpy()
    calibrated = mixed + selected["alpha"] * raw_bias[None, :]
    trusted_global = set(validation_checkpoint["validation"].get("trusted_indices", []))
    trusted_local = np.asarray([index in trusted_global for index in validation_indices])
    result: dict[str, Any] = {
        "method": "strict_5fold_macro_tta_and_marginal_bias",
        "selection_order": [
            "macro_accuracy_desc",
            "macro_nll_asc",
            "class_bias_alpha_asc",
            "original_weight_desc",
        ],
        "selected": selected,
        "original": macro_metrics(original, labels),
        "hflip": macro_metrics(hflip, labels),
        "selected_full_validation": macro_metrics(calibrated, labels),
        "trusted_selected_full_validation": (
            macro_metrics(calibrated[trusted_local], labels[trusted_local])
            if trusted_local.any()
            else None
        ),
        "grid": {
            "folds": 5,
            "original_weight": {"minimum": 0.0, "maximum": 1.0, "step": 0.05},
            "class_bias_alpha": {"minimum": 0.0, "maximum": 2.0, "step": 0.05},
        },
        "validation_rows": len(validation_indices),
        "trusted_validation_rows": int(trusted_local.sum()),
        "fold_ids": fold_ids.tolist(),
    }
    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")

    final_checkpoint["class_bias"] = torch.from_numpy(
        (selected["alpha"] * raw_bias).astype(np.float32)
    )
    final_checkpoint["calibration"] = {
        "method": result["method"],
        "tta": "hflip",
        "original_weight": selected["original_weight"],
        "hflip_weight": selected["hflip_weight"],
        "view_weights": {
            "original": selected["original_weight"],
            "hflip": selected["hflip_weight"],
        },
        "alpha": selected["alpha"],
        "folds": 5,
        "cv_macro_accuracy": selected["cv_macro_accuracy"],
        "cv_macro_nll": selected["cv_macro_nll"],
    }
    output_checkpoint = Path(args.output_checkpoint)
    atomic_torch_save(final_checkpoint, output_checkpoint)

    training_metrics = output_checkpoint.parent / "metrics.json"
    if training_metrics.is_file():
        payload = json.loads(training_metrics.read_text(encoding="utf-8"))
        payload["calibration"] = result
        training_metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"calibrated_checkpoint={output_checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
