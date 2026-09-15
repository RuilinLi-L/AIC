"""Select a weighted original/HFlip TTA mix and class-bias calibrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from calibrate import class_bias_from_logits, stratified_folds
from robust_clip import (
    FolderImageDataset,
    RobustCLIPClassifier,
    inject_lora_from_config,
    list_images,
    load_classifier_state,
    load_clip,
    resolve_device,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--original-val-logits", required=True)
    parser.add_argument("--hflip-val-logits", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_grid(step: float, maximum: float, name: str) -> np.ndarray:
    if step <= 0.0 or maximum < 0.0:
        raise ValueError(f"{name} grid is invalid")
    count = int(round(maximum / step))
    return np.linspace(0.0, maximum, count + 1, dtype=np.float32)


def _collect_logits(
    checkpoint: dict,
    rows: list[tuple[str, int, str]],
    val_indices: list[int],
    model_dir: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    original: np.ndarray | None,
    hflip: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if original is not None and hflip is not None:
        return original, hflip
    model, processor = load_clip(model_dir, device)
    class_names = checkpoint["class_names"]
    config = checkpoint.get("config", {})
    classifier = RobustCLIPClassifier(
        model,
        classifier_init=torch.randn(len(class_names), int(model.config.projection_dim), device=device),
        bottleneck=int(config.get("bottleneck", 128)),
        initial_logit_scale=float(config.get("initial_logit_scale", 10.0)),
        max_logit_scale=float(config.get("max_logit_scale", 100.0)),
        learn_logit_scale=False,
    ).to(device)
    lora_modules = inject_lora_from_config(classifier.clip, config)
    if lora_modules:
        print(f"loaded_lora_modules={len(lora_modules)}", flush=True)
    load_classifier_state(classifier, checkpoint["model"])
    classifier.eval()
    loader = DataLoader(
        FolderImageDataset(rows, val_indices, processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    original_batches: list[np.ndarray] = []
    hflip_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader, 1):
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            if original is None:
                original_batches.append(classifier(pixels, None)[0].float().cpu().numpy())
            if hflip is None:
                hflip_batches.append(classifier(torch.flip(pixels, dims=[3]), None)[0].float().cpu().numpy())
            if batch_id % 40 == 0:
                print(f"validation_batches={batch_id}/{len(loader)}", flush=True)
    if original is None:
        original = np.concatenate(original_batches, axis=0)
    if hflip is None:
        hflip = np.concatenate(hflip_batches, axis=0)
    return original, hflip


def _search_mix(
    original: np.ndarray,
    hflip: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    weights: np.ndarray,
    alphas: np.ndarray,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Jointly select view weight and bias alpha by stratified CV."""
    original_t = torch.as_tensor(original, dtype=torch.float32, device=device)
    hflip_t = torch.as_tensor(hflip, dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    alpha_t = torch.as_tensor(alphas, dtype=torch.float32, device=device)
    fold_t = torch.as_tensor(fold_ids, dtype=torch.long, device=device)
    correct = torch.zeros((len(weights), len(alphas)), dtype=torch.long, device=device)
    losses = torch.zeros((len(weights), len(alphas)), dtype=torch.float64, device=device)
    for weight_id, weight in enumerate(weights):
        mixed = float(weight) * original_t + (1.0 - float(weight)) * hflip_t
        for fold in range(int(fold_ids.max()) + 1):
            calibration = fold_t != fold
            evaluation = ~calibration
            bias = class_bias_from_logits(mixed[calibration])
            eval_logits = mixed[evaluation]
            eval_labels = labels_t[evaluation]
            adjusted = eval_logits.unsqueeze(0) + alpha_t[:, None, None] * bias[None, None, :]
            predictions = adjusted.argmax(dim=-1)
            correct[weight_id] += (predictions == eval_labels.unsqueeze(0)).sum(dim=1)
            losses[weight_id] += (
                torch.logsumexp(adjusted.double(), dim=-1)
                - adjusted.double().gather(2, eval_labels[None, :, None].expand(len(alphas), -1, 1)).squeeze(2)
            ).sum(dim=1)
    accuracy = correct.double() / float(len(labels))
    loss = losses / float(len(labels))
    best = None
    for weight_id, weight in enumerate(weights):
        for alpha_id, alpha in enumerate(alphas):
            candidate = (
                float(accuracy[weight_id, alpha_id].item()),
                float(loss[weight_id, alpha_id].item()),
                -float(alpha),
                float(weight),
            )
            if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1:] < best[1:]):
                best = candidate
    assert best is not None
    return best[3], -best[2], best[0], best[1]


def _metrics(logits: np.ndarray, labels: np.ndarray, weight: float, alpha: float, bias: np.ndarray) -> dict:
    adjusted = logits + alpha * bias[None, :]
    shifted = adjusted - adjusted.max(axis=1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return {
        "accuracy": float((adjusted.argmax(axis=1) == labels).mean()),
        "loss": float(-log_prob[np.arange(len(labels)), labels].mean()),
        "original_weight": float(weight),
        "hflip_weight": float(1.0 - weight),
        "calibration_alpha": float(alpha),
    }


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    rows = list_images(args.train_dir)
    _, val_indices = stratified_split(rows, float(config.get("val_ratio", 0.05)), int(config.get("seed", 2026)))
    labels = np.asarray([label for _, label, _ in rows], dtype=np.int64)[val_indices]
    expected_shape = (len(val_indices), len(checkpoint["class_names"]))
    original_path = Path(args.original_val_logits)
    hflip_path = Path(args.hflip_val_logits)
    original = None if args.force or not original_path.is_file() else np.load(original_path)
    hflip = None if args.force or not hflip_path.is_file() else np.load(hflip_path)
    if original is not None and original.shape != expected_shape:
        raise ValueError(f"Unexpected original logits shape: {original.shape}, expected {expected_shape}")
    if hflip is not None and hflip.shape != expected_shape:
        raise ValueError(f"Unexpected hflip logits shape: {hflip.shape}, expected {expected_shape}")
    original, hflip = _collect_logits(
        checkpoint, rows, val_indices, args.model_dir, device, args.batch_size, args.workers, original, hflip
    )
    original_path.parent.mkdir(parents=True, exist_ok=True)
    hflip_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(original_path, original)
    np.save(hflip_path, hflip)
    print(f"validation_logits original={original.shape} hflip={hflip.shape}", flush=True)

    fold_ids = stratified_folds(labels, args.folds, int(config.get("seed", 2026)))
    weights = _validate_grid(args.weight_step, 1.0, "view-weight")
    alphas = _validate_grid(args.alpha_step, args.alpha_max, "alpha")
    best_weight, best_alpha, cv_accuracy, cv_loss = _search_mix(
        original, hflip, labels, fold_ids, weights, alphas, device
    )
    mixed = best_weight * original + (1.0 - best_weight) * hflip
    mixed_tensor = torch.as_tensor(mixed, dtype=torch.float32, device=device)
    raw_bias = class_bias_from_logits(mixed_tensor).cpu().numpy()
    full_bias = best_alpha * raw_bias
    result = {
        "selected": {
            "original_weight": best_weight,
            "hflip_weight": 1.0 - best_weight,
            "calibration_alpha": best_alpha,
            "cv_accuracy": cv_accuracy,
            "cv_loss": cv_loss,
        },
        "original": _metrics(original, labels, 1.0, 0.0, np.zeros(original.shape[1], dtype=np.float32)),
        "hflip": _metrics(hflip, labels, 0.0, 0.0, np.zeros(hflip.shape[1], dtype=np.float32)),
        "selected_full": _metrics(mixed, labels, best_weight, best_alpha, raw_bias),
        "grid": {
            "weight_step": args.weight_step,
            "alpha_step": args.alpha_step,
            "alpha_max": args.alpha_max,
            "folds": args.folds,
        },
    }
    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

    if args.output_checkpoint:
        checkpoint["class_bias"] = torch.from_numpy(full_bias)
        checkpoint["calibration"] = {
            "method": "validation_marginal_log_bias",
            "tta": "hflip",
            "original_weight": best_weight,
            "hflip_weight": 1.0 - best_weight,
            "view_weights": {"original": best_weight, "hflip": 1.0 - best_weight},
            "alpha": best_alpha,
            "folds": args.folds,
            "cv_accuracy": cv_accuracy,
            "cv_loss": cv_loss,
            "full_val_accuracy": result["selected_full"]["accuracy"],
            "full_val_loss": result["selected_full"]["loss"],
        }
        output_checkpoint = Path(args.output_checkpoint)
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_checkpoint)
        print(f"saved_checkpoint={output_checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
