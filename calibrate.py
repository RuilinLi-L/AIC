"""Fit a train-validation class-bias calibrator and optionally rewrite cached test logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robust_clip import (
    RobustCLIPClassifier,
    list_images,
    load_classifier_state,
    resolve_device,
    stratified_split,
)


class FeatureOnlyBackbone(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.config = SimpleNamespace(projection_dim=dim)

    def get_image_features(self, pixel_values):
        return pixel_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--val-logits", default=None, help="Optional reusable validation logits .npy")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-steps", type=int, default=61)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--test-logits", default=None)
    parser.add_argument("--base-predictions", default=None)
    parser.add_argument("--output-predictions", default=None)
    return parser.parse_args()


def stratified_folds(labels: np.ndarray, folds: int, seed: int) -> np.ndarray:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    result = np.empty(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) < folds:
            raise ValueError(f"Class {label} has only {len(indices)} validation rows for {folds} folds")
        rng.shuffle(indices)
        result[indices] = np.arange(len(indices)) % folds
    return result


@torch.no_grad()
def collect_val_logits(
    checkpoint: dict,
    features: np.ndarray,
    val_indices: list[int],
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    config = checkpoint.get("config", {})
    num_classes = len(checkpoint["class_names"])
    dim = int(features.shape[1])
    classifier = RobustCLIPClassifier(
        FeatureOnlyBackbone(dim),
        classifier_init=torch.randn(num_classes, dim),
        bottleneck=int(config.get("bottleneck", 128)),
        initial_logit_scale=float(config.get("initial_logit_scale", 10.0)),
        max_logit_scale=float(config.get("max_logit_scale", 100.0)),
        learn_logit_scale=False,
    ).to(device)
    load_classifier_state(classifier, checkpoint["model"])
    classifier.eval()
    rows = []
    for start in range(0, len(val_indices), batch_size):
        indices = val_indices[start : start + batch_size]
        batch = torch.tensor(np.asarray(features[indices]), dtype=torch.float32, device=device)
        rows.append(classifier.forward_features(batch, None)[0].cpu().numpy())
    return np.concatenate(rows, axis=0)


def class_bias_from_logits(logits: torch.Tensor) -> torch.Tensor:
    num_classes = logits.shape[1]
    marginal = logits.softmax(dim=1).mean(dim=0).clamp_min(1e-6)
    return -torch.log(marginal * num_classes)


def choose_alpha(
    logits: torch.Tensor,
    labels: torch.Tensor,
    fold_ids: np.ndarray,
    alphas: np.ndarray,
) -> tuple[float, float, float]:
    correct = np.zeros(len(alphas), dtype=np.int64)
    losses = np.zeros(len(alphas), dtype=np.float64)
    fold_tensor = torch.as_tensor(fold_ids, device=logits.device)
    for fold in np.unique(fold_ids):
        calibration = fold_tensor != int(fold)
        evaluation = ~calibration
        bias = class_bias_from_logits(logits[calibration])
        for index, alpha in enumerate(alphas):
            adjusted = logits[evaluation] + float(alpha) * bias
            correct[index] += int((adjusted.argmax(dim=1) == labels[evaluation]).sum())
            losses[index] += float(F.cross_entropy(adjusted, labels[evaluation], reduction="sum"))
    accuracy = correct / len(labels)
    loss = losses / len(labels)
    # Accuracy is primary; loss and then the smaller correction break ties.
    order = np.lexsort((alphas, loss, -accuracy))
    best = int(order[0])
    return float(alphas[best]), float(accuracy[best]), float(loss[best])


def write_predictions(
    base_predictions: Path,
    output_predictions: Path,
    adjusted_logits: np.ndarray,
    class_names: list[str],
) -> None:
    lines = base_predictions.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(adjusted_logits):
        raise ValueError("base prediction row count does not match test logits")
    names = [line.rsplit(",", 1)[0].strip() for line in lines]
    predicted = adjusted_logits.argmax(axis=1)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with output_predictions.open("w", encoding="utf-8", newline="") as handle:
        for name, index in zip(names, predicted):
            class_name = class_names[int(index)]
            try:
                numeric_id = int(class_name)
            except ValueError:
                numeric_id = int(index)
            handle.write(f"{name}, {numeric_id:04d}\n")
    counts = np.bincount(predicted, minlength=len(class_names))
    print(
        f"calibrated_prediction_counts min={int(counts.min())} median={float(np.median(counts)):.1f} "
        f"max={int(counts.max())} cv={float(counts.std() / counts.mean()):.4f}"
    )
    print(f"wrote={output_predictions.resolve()} rows={len(predicted)}")


def main() -> None:
    args = parse_args()
    if args.alpha_max < 0 or args.alpha_steps < 2:
        raise ValueError("alpha range is invalid")
    optional = [args.test_logits, args.base_predictions, args.output_predictions]
    if any(optional) and not all(optional):
        raise ValueError("test-logits, base-predictions, and output-predictions must be supplied together")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    rows = list_images(args.train_dir)
    class_names = list(checkpoint["class_names"])
    observed_names = [name for _, name in sorted({(label, name) for _, label, name in rows})]
    if observed_names != class_names:
        raise ValueError("Checkpoint classes do not match train-dir")
    features = np.load(args.feature_cache, mmap_mode="r")
    expected_dim = int(checkpoint["model"]["classifier"].shape[1])
    if features.shape != (len(rows), expected_dim):
        raise ValueError(f"Unexpected feature cache shape: {features.shape}")
    config = checkpoint.get("config", {})
    _, val_indices = stratified_split(rows, float(config.get("val_ratio", 0.1)), int(config.get("seed", 2026)))
    labels = np.array([label for _, label, _ in rows], dtype=np.int64)[val_indices]

    val_logits_path = Path(args.val_logits) if args.val_logits else None
    if val_logits_path and val_logits_path.is_file():
        val_logits = np.load(val_logits_path)
        if val_logits.shape != (len(val_indices), len(class_names)):
            raise ValueError(f"Unexpected validation logits shape: {val_logits.shape}")
        print(f"loaded_val_logits={val_logits_path.resolve()} shape={val_logits.shape}")
    else:
        val_logits = collect_val_logits(checkpoint, features, val_indices, device)
        if val_logits_path:
            val_logits_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(val_logits_path, val_logits)
            print(f"saved_val_logits={val_logits_path.resolve()} shape={val_logits.shape}")

    logits_tensor = torch.tensor(val_logits, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    fold_ids = stratified_folds(labels, args.folds, int(config.get("seed", 2026)))
    alphas = np.linspace(0.0, args.alpha_max, args.alpha_steps)
    alpha, cv_accuracy, cv_loss = choose_alpha(logits_tensor, labels_tensor, fold_ids, alphas)
    full_bias = alpha * class_bias_from_logits(logits_tensor)
    adjusted = logits_tensor + full_bias
    baseline_accuracy = float((logits_tensor.argmax(dim=1) == labels_tensor).float().mean())
    adjusted_accuracy = float((adjusted.argmax(dim=1) == labels_tensor).float().mean())
    adjusted_loss = float(F.cross_entropy(adjusted, labels_tensor))

    checkpoint["class_bias"] = full_bias.cpu()
    checkpoint["calibration"] = {
        "method": "validation_marginal_log_bias",
        "alpha": alpha,
        "folds": args.folds,
        "cv_accuracy": cv_accuracy,
        "cv_loss": cv_loss,
        "baseline_val_accuracy": baseline_accuracy,
        "full_val_accuracy": adjusted_accuracy,
        "full_val_loss": adjusted_loss,
    }
    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_checkpoint)
    (output_checkpoint.parent / "calibration.json").write_text(
        json.dumps(checkpoint["calibration"], indent=2), encoding="utf-8"
    )
    print(json.dumps(checkpoint["calibration"], indent=2))
    print(f"saved_checkpoint={output_checkpoint.resolve()}")

    if all(optional):
        test_logits = np.load(args.test_logits, mmap_mode="r")
        if test_logits.shape[1] != len(class_names):
            raise ValueError(f"Unexpected test logits shape: {test_logits.shape}")
        adjusted_test = np.asarray(test_logits) + full_bias.cpu().numpy()[None, :]
        write_predictions(Path(args.base_predictions), Path(args.output_predictions), adjusted_test, class_names)


if __name__ == "__main__":
    main()
