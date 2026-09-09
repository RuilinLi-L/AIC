r"""Train the first robust CLIP baseline for the AIC noisy-label challenge.

Example (PowerShell):
  python train.py --train-dir .\data\preliminary\train --model-dir .\clip-ViT-B-32

The training directory must contain one subdirectory per class.  The script
creates its own deterministic validation split from the noisy training data;
the official test set is never opened here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from robust_clip import (
    FolderImageDataset,
    RobustCLIPClassifier,
    TrainConfig,
    _feature_tensor,
    classwise_quality,
    encode_class_text,
    list_images,
    load_clip,
    resolve_device,
    save_checkpoint,
    seed_everything,
    stratified_split,
    train_round,
)


def parse_fraction_schedule(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("clean-fraction-schedule cannot be empty")
    schedule = tuple(float(part) for part in parts)
    for fraction in schedule:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("each clean-fraction-schedule value must be in (0, 1]")
    return schedule


def clean_fraction_for_round(config: TrainConfig, round_id: int) -> float:
    if config.clean_fraction_schedule:
        index = min(round_id, len(config.clean_fraction_schedule) - 1)
        return config.clean_fraction_schedule[index]
    return config.clean_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, help="Folder-organized noisy training data")
    parser.add_argument("--model-dir", default="clip-ViT-B-32", help="Local OpenAI CLIP ViT-B/32 directory")
    parser.add_argument("--output-dir", default="outputs/robust_clip_v1")
    parser.add_argument("--resume", default=None, help="Optional existing model.pt to continue from")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--clean-fraction", type=float, default=0.8)
    parser.add_argument(
        "--clean-fraction-schedule",
        default=None,
        help="Comma-separated per-round clean fractions, e.g. 0.90,0.85,0.80",
    )
    parser.add_argument("--score-momentum", type=float, default=0.7, help="EMA momentum for sample scores")
    parser.add_argument("--weight-floor", type=float, default=0.25, help="Minimum per-sample weight")
    parser.add_argument("--weight-cap", type=float, default=2.0, help="Upper clip for class-balanced weights")
    parser.add_argument("--class-weight-beta", type=float, default=0.9999, help="Effective-number beta for class weights")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_fraction_schedule = parse_fraction_schedule(args.clean_fraction_schedule)
    if not 0.0 <= args.score_momentum < 1.0:
        raise ValueError("--score-momentum must be in [0, 1)")
    if not 0.0 <= args.weight_floor < 1.0:
        raise ValueError("--weight-floor must be in [0, 1)")
    if args.weight_cap <= 0:
        raise ValueError("--weight-cap must be positive")
    if not 0.0 <= args.class_weight_beta < 1.0:
        raise ValueError("--class-weight-beta must be in [0, 1)")
    config = TrainConfig(
        model_dir=args.model_dir,
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        rounds=args.rounds,
        clean_fraction=args.clean_fraction,
        clean_fraction_schedule=clean_fraction_schedule,
        score_momentum=args.score_momentum,
        weight_floor=args.weight_floor,
        weight_cap=args.weight_cap,
        class_weight_beta=args.class_weight_beta,
        val_ratio=args.val_ratio,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    seed_everything(config.seed)
    device = resolve_device(config.device)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")

    rows = list_images(config.train_dir)
    # list_images emits one class name per row; recover the sorted unique map.
    class_names = [name for _, name in sorted({(label, name) for _, label, name in rows})]
    train_idx, val_idx = stratified_split(rows, config.val_ratio, config.seed)
    model, processor = load_clip(config.model_dir, device)
    text_features = encode_class_text(model, processor, class_names, device)
    classifier = RobustCLIPClassifier(model, text_features, bottleneck=config.bottleneck).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        previous_classes = list(checkpoint.get("class_names", []))
        if previous_classes != class_names:
            raise ValueError("Checkpoint class_names do not match the current training directory")
        classifier.load_state_dict(checkpoint["model"], strict=True)
        print(f"resumed_from={Path(args.resume).resolve()}")

    train_ds = FolderImageDataset(rows, train_idx, processor)
    val_ds = FolderImageDataset(rows, val_idx, processor)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )

    # Initial quality comes from frozen CLIP's image-text agreement.  A later
    # round blends it with the adapted classifier confidence.
    full_loader = DataLoader(
        FolderImageDataset(rows, list(range(len(rows))), processor),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )
    # The above dataset is intentionally full-data only for screening.  Use a
    # small temporary adapter-free classifier so its trainable weights cannot
    # influence the first screening round.
    quality_scores = np.zeros(len(rows), dtype=np.float32)
    with torch.no_grad():
        for batch in full_loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            idx = batch["row_index"].numpy()
            y = batch["label"].to(device)
            base = torch.nn.functional.normalize(_feature_tensor(model.get_image_features(pixel_values=pixels)).float(), dim=-1)
            probs = (base @ text_features.t()).softmax(dim=-1)
            quality_scores[idx] = probs.gather(1, y[:, None]).squeeze(1).cpu().numpy()
    first_fraction = clean_fraction_for_round(config, 0)
    quality, clean = classwise_quality(rows, quality_scores, first_fraction, config.weight_floor)
    print(
        f"initial clean_fraction={first_fraction:.3f} clean={int(clean.sum())}/{len(clean)} "
        f"quality_mean={quality.mean():.4f}"
    )

    class_counts = np.bincount(np.array([label for _, label, _ in rows]), minlength=len(class_names))
    for round_id in range(config.rounds):
        stats = train_round(classifier, train_loader, text_features, quality, class_counts, config, device)
        print(
            f"round={round_id + 1}/{config.rounds} loss={stats['loss']:.5f} "
            f"noisy_label_acc={stats['noisy_label_accuracy']:.4f}"
        )

        # Re-score all training images after each round.  The validation split
        # remains untouched by the optimizer and is reported only as a noisy-
        # label diagnostic, never as clean test evidence.
        if round_id + 1 < config.rounds:
            model_scores = np.zeros(len(rows), dtype=np.float32)
            classifier.eval()
            with torch.no_grad():
                for batch in full_loader:
                    pixels = batch["pixel_values"].to(device, non_blocking=True)
                    idx = batch["row_index"].numpy()
                    y = batch["label"].to(device)
                    logits, _, _, _ = classifier(pixels, text_features)
                    model_scores[idx] = logits.softmax(dim=-1).gather(1, y[:, None]).squeeze(1).cpu().numpy()
            # EMA keeps the frozen CLIP prior while letting adapted confidence refine the ranking.
            quality_scores = config.score_momentum * quality_scores + (1.0 - config.score_momentum) * model_scores
            next_fraction = clean_fraction_for_round(config, round_id + 1)
            quality, clean = classwise_quality(rows, quality_scores, next_fraction, config.weight_floor)
            print(
                f"rescreen round={round_id + 1} clean_fraction={next_fraction:.3f} "
                f"clean={int(clean.sum())}/{len(clean)} quality_mean={quality.mean():.4f}"
            )

    # Report the noisy validation diagnostic, but do not select checkpoints by it.
    classifier.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            logits, _, _, _ = classifier(batch["pixel_values"].to(device), text_features)
            val_correct += int((logits.argmax(dim=-1).cpu() == batch["label"]).sum())
            val_total += len(batch["label"])
    print(f"noisy_validation_label_accuracy={val_correct / max(val_total, 1):.4f} (diagnostic only)")

    save_checkpoint(out / "model.pt", classifier, class_names, config, quality)
    (out / "classes.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out / "sample_quality.npy", quality)
    print(f"saved={out / 'model.pt'}")


if __name__ == "__main__":
    main()
