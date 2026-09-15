"""Evaluate horizontal-flip TTA on the held-out training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from calibrate import choose_alpha, class_bias_from_logits, stratified_folds
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def evaluate_logits(logits: np.ndarray, labels: np.ndarray, folds: int, seed: int) -> tuple[dict, np.ndarray]:
    tensor = torch.tensor(logits, dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.long)
    predictions = tensor.argmax(dim=1)
    counts = torch.bincount(predictions, minlength=tensor.shape[1]).float()
    fold_ids = stratified_folds(labels, folds, seed)
    alpha, cv_accuracy, cv_loss = choose_alpha(
        tensor, target, fold_ids, np.linspace(0.0, 3.0, 61)
    )
    bias = alpha * class_bias_from_logits(tensor)
    adjusted = tensor + bias
    metrics = {
        "accuracy": float((predictions == target).float().mean()),
        "loss": float(F.cross_entropy(tensor, target)),
        "prediction_count_cv": float(counts.std(unbiased=False) / counts.mean()),
        "calibration_alpha": alpha,
        "calibration_cv_accuracy": cv_accuracy,
        "calibration_cv_loss": cv_loss,
        "calibrated_full_accuracy": float((adjusted.argmax(dim=1) == target).float().mean()),
        "calibrated_full_loss": float(F.cross_entropy(adjusted, target)),
    }
    return metrics, bias.numpy()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    rows = list_images(args.train_dir)
    _, val_indices = stratified_split(
        rows, float(config.get("val_ratio", 0.1)), int(config.get("seed", 2026))
    )
    labels = np.array([label for _, label, _ in rows], dtype=np.int64)[val_indices]
    expected_shape = (len(val_indices), len(checkpoint["class_names"]))
    original_path = Path(args.original_val_logits)
    hflip_path = Path(args.hflip_val_logits)
    original = None
    hflip = None
    if original_path.is_file() and not args.force:
        original = np.load(original_path)
        if original.shape != expected_shape:
            raise ValueError(f"Unexpected original logits shape: {original.shape}, expected {expected_shape}")
        print(f"loaded_original_logits={original_path.resolve()} shape={original.shape}", flush=True)
    if hflip_path.is_file() and not args.force:
        hflip = np.load(hflip_path)
        if hflip.shape != expected_shape:
            raise ValueError(f"Unexpected hflip logits shape: {hflip.shape}")
        print(f"loaded_hflip_logits={hflip_path.resolve()} shape={hflip.shape}", flush=True)

    if original is None or hflip is None:
        model, processor = load_clip(args.model_dir, device)
        classifier = RobustCLIPClassifier(
            model,
            classifier_init=torch.randn(
                len(checkpoint["class_names"]), int(model.config.projection_dim), device=device
            ),
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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        original_batches = []
        hflip_batches = []
        with torch.no_grad():
            for batch_id, batch in enumerate(loader, 1):
                pixels = batch["pixel_values"].to(device, non_blocking=True)
                if original is None:
                    original_batches.append(classifier(pixels, None)[0].cpu().numpy())
                if hflip is None:
                    hflip_batches.append(classifier(torch.flip(pixels, dims=[3]), None)[0].cpu().numpy())
                if batch_id % 40 == 0:
                    print(f"validation_batches={batch_id}/{len(loader)}", flush=True)
        if original is None:
            original = np.concatenate(original_batches, axis=0)
            original_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(original_path, original)
            print(f"saved_original_logits={original_path.resolve()} shape={original.shape}", flush=True)
        if hflip is None:
            hflip = np.concatenate(hflip_batches, axis=0)
            hflip_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(hflip_path, hflip)
            print(f"saved_hflip_logits={hflip_path.resolve()} shape={hflip.shape}", flush=True)

    seed = int(config.get("seed", 2026))
    average = 0.5 * (original + hflip)
    original_metrics, _ = evaluate_logits(original, labels, args.folds, seed)
    hflip_metrics, _ = evaluate_logits(hflip, labels, args.folds, seed)
    average_metrics, average_bias = evaluate_logits(average, labels, args.folds, seed)
    result = {
        "original": original_metrics,
        "hflip": hflip_metrics,
        "average": average_metrics,
    }
    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

    if args.output_checkpoint:
        checkpoint["class_bias"] = torch.from_numpy(average_bias)
        checkpoint["calibration"] = {
            "method": "validation_marginal_log_bias",
            "tta": "hflip",
            "alpha": average_metrics["calibration_alpha"],
            "folds": args.folds,
            "cv_accuracy": average_metrics["calibration_cv_accuracy"],
            "cv_loss": average_metrics["calibration_cv_loss"],
            "full_val_accuracy": average_metrics["calibrated_full_accuracy"],
            "full_val_loss": average_metrics["calibrated_full_loss"],
        }
        output_checkpoint = Path(args.output_checkpoint)
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_checkpoint)
        print(f"saved_checkpoint={output_checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
