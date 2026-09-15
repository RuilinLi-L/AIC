r"""Train the first robust CLIP baseline for the AIC noisy-label challenge.

Example (PowerShell):
  python train.py --train-dir .\data\preliminary\train --model-dir .\clip-ViT-B-32

The training directory must contain one subdirectory per class.  The script
creates its own deterministic validation split from the noisy training data;
the official test set is never opened here.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from robust_clip import (
    FeatureDataset,
    FeatureOnlyBackbone,
    FolderImageDataset,
    RobustCLIPClassifier,
    TrainConfig,
    build_optimizer,
    classwise_rank_scores,
    classwise_quality,
    collect_frozen_features,
    evaluate_classifier,
    load_classifier_state,
    list_images,
    load_clip,
    resolve_device,
    robust_visual_prototypes_and_scores,
    save_checkpoint,
    seed_everything,
    stratified_split,
    trainable_state_dict,
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


def load_or_create_feature_cache(model, processor, rows, config, device, cache_path: Path):
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = np.load(cache_path, mmap_mode="r")
        if (
            metadata.get("rows") == len(rows)
            and tuple(cached.shape) == (len(rows), int(model.config.projection_dim))
        ):
            print(f"loaded_feature_cache={cache_path.resolve()} shape={cached.shape}")
            labels = np.array([label for _, label, _ in rows], dtype=np.int64)
            return cached, labels
        print("feature_cache_mismatch=recomputing")

    loader = DataLoader(
        FolderImageDataset(rows, list(range(len(rows))), processor),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )
    features, labels = collect_frozen_features(model, loader, len(rows), device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    metadata_path.write_text(
        json.dumps({"rows": len(rows), "dim": features.shape[1]}, indent=2),
        encoding="utf-8",
    )
    print(f"saved_feature_cache={cache_path.resolve()} shape={features.shape}")
    return np.load(cache_path, mmap_mode="r"), labels


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
    parser.add_argument("--low-confidence-multiplier", type=float, default=1.0, help="Extra multiplier below the clean cutoff")
    parser.add_argument("--weight-cap", type=float, default=2.0, help="Upper clip for class-balanced weights")
    parser.add_argument("--class-weight-beta", type=float, default=0.9999, help="Effective-number beta for class weights")
    parser.add_argument("--loss-mode", choices=["soft_ce", "hard_gce", "hard_mae"], default="soft_ce")
    parser.add_argument("--gce-q", type=float, default=0.7, help="GCE exponent used by hard_gce")
    parser.add_argument("--distill-weight", type=float, default=0.0, help="Optional text distillation weight; keep 0 with numeric class IDs")
    parser.add_argument("--drift-weight", type=float, default=0.0, help="Feature drift penalty toward frozen CLIP features")
    parser.add_argument("--augmentation", choices=["none", "light", "strong"], default="none")
    parser.add_argument("--no-augment", action="store_true", help="Deprecated alias that forces --augmentation none")
    parser.add_argument("--prototype-keep-fraction", type=float, default=0.7)
    parser.add_argument("--prototype-iterations", type=int, default=2)
    parser.add_argument("--initial-logit-scale", type=float, default=30.0)
    parser.add_argument("--max-logit-scale", type=float, default=50.0)
    parser.add_argument("--learn-logit-scale", action="store_true")
    parser.add_argument("--feature-cache", default=None, help="Reusable frozen feature .npy; defaults inside output-dir")
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
    if not 0.0 <= args.low_confidence_multiplier <= 1.0:
        raise ValueError("--low-confidence-multiplier must be in [0, 1]")
    if args.weight_cap <= 0:
        raise ValueError("--weight-cap must be positive")
    if not 0.0 <= args.class_weight_beta < 1.0:
        raise ValueError("--class-weight-beta must be in [0, 1)")
    if not 0.0 < args.gce_q <= 1.0:
        raise ValueError("--gce-q must be in (0, 1]")
    if args.distill_weight != 0.0:
        raise ValueError("--distill-weight must be 0 because numeric class IDs have no text semantics")
    if args.distill_weight < 0.0 or args.drift_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if not 0.0 < args.prototype_keep_fraction <= 1.0:
        raise ValueError("--prototype-keep-fraction must be in (0, 1]")
    if args.prototype_iterations < 1:
        raise ValueError("--prototype-iterations must be at least 1")
    if args.max_logit_scale <= 1.0:
        raise ValueError("--max-logit-scale must be greater than 1")
    if not 1.0 <= args.initial_logit_scale <= args.max_logit_scale:
        raise ValueError("--initial-logit-scale must be in [1, max-logit-scale]")
    augmentation = "none" if args.no_augment else args.augmentation
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
        low_confidence_multiplier=args.low_confidence_multiplier,
        weight_cap=args.weight_cap,
        class_weight_beta=args.class_weight_beta,
        loss_mode=args.loss_mode,
        gce_q=args.gce_q,
        distill_weight=args.distill_weight,
        drift_weight=args.drift_weight,
        augmentation=augmentation,
        prototype_keep_fraction=args.prototype_keep_fraction,
        prototype_iterations=args.prototype_iterations,
        initial_logit_scale=args.initial_logit_scale,
        max_logit_scale=args.max_logit_scale,
        learn_logit_scale=args.learn_logit_scale,
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

    cache_path = Path(args.feature_cache) if args.feature_cache else out / "frozen_features.npy"
    features, labels = load_or_create_feature_cache(model, processor, rows, config, device, cache_path)

    # Prototypes and screening use train_idx only. The validation split is now
    # a genuine holdout used solely for checkpoint selection.
    visual_prototypes, prototype_scores = robust_visual_prototypes_and_scores(
        features,
        labels,
        len(class_names),
        candidate_indices=train_idx,
        keep_fraction=config.prototype_keep_fraction,
        iterations=config.prototype_iterations,
    )
    quality_scores = classwise_rank_scores(labels, prototype_scores, train_idx)
    classifier_backbone = model
    if config.augmentation == "none":
        projection_dim = int(model.config.projection_dim)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        classifier_backbone = FeatureOnlyBackbone(projection_dim)
    classifier = RobustCLIPClassifier(
        classifier_backbone,
        text_features=None,
        bottleneck=config.bottleneck,
        classifier_init=torch.from_numpy(visual_prototypes).to(device),
        initial_logit_scale=config.initial_logit_scale,
        max_logit_scale=config.max_logit_scale,
        learn_logit_scale=config.learn_logit_scale,
    ).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        previous_classes = list(checkpoint.get("class_names", []))
        if previous_classes != class_names:
            raise ValueError("Checkpoint class_names do not match the current training directory")
        load_classifier_state(classifier, checkpoint["model"])
        print(f"resumed_from={Path(args.resume).resolve()}")

    if config.augmentation == "none":
        train_ds = FeatureDataset(rows, train_idx, features)
        train_workers = 0
    else:
        train_ds = FolderImageDataset(rows, train_idx, processor, augment=config.augmentation)
        train_workers = config.workers
    val_ds = FeatureDataset(rows, val_idx, features)
    screen_ds = FeatureDataset(rows, train_idx, features)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=train_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    screen_loader = DataLoader(
        screen_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    first_fraction = clean_fraction_for_round(config, 0)
    quality, clean = classwise_quality(
        rows,
        quality_scores,
        first_fraction,
        config.weight_floor,
        candidate_indices=train_idx,
        low_confidence_multiplier=config.low_confidence_multiplier,
    )
    print(
        f"initial clean_fraction={first_fraction:.3f} clean={int(clean.sum())}/{len(train_idx)} "
        f"train_quality_mean={quality[train_idx].mean():.4f}"
    )

    class_counts = np.bincount(labels[np.asarray(train_idx)], minlength=len(class_names))
    optimizer = build_optimizer(classifier, config)
    history = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = trainable_state_dict(classifier)
    global_epoch = 0
    for round_id in range(config.rounds):
        round_best_accuracy = -1.0
        round_best_loss = float("inf")
        round_best_state = trainable_state_dict(classifier)
        round_best_optimizer = copy.deepcopy(optimizer.state_dict())
        for epoch_id in range(config.epochs):
            global_epoch += 1
            stats = train_round(
                classifier,
                train_loader,
                None,
                quality,
                clean,
                class_counts,
                config,
                device,
                optimizer=optimizer,
                epochs=1,
            )
            val_stats = evaluate_classifier(classifier, val_loader, device)
            effective_scale = min(float(classifier.logit_scale.detach().exp().cpu()), config.max_logit_scale)
            record = {
                "global_epoch": global_epoch,
                "round": round_id + 1,
                "epoch": epoch_id + 1,
                "train_loss": stats["loss"],
                "train_noisy_accuracy": stats["noisy_label_accuracy"],
                "val_loss": val_stats["loss"],
                "val_noisy_accuracy": val_stats["accuracy"],
                "logit_scale": effective_scale,
            }
            history.append(record)
            print(
                f"round={round_id + 1}/{config.rounds} epoch={epoch_id + 1}/{config.epochs} "
                f"train_loss={stats['loss']:.5f} train_acc={stats['noisy_label_accuracy']:.4f} "
                f"val_loss={val_stats['loss']:.5f} val_acc={val_stats['accuracy']:.4f} "
                f"scale={effective_scale:.2f}"
            )
            improved_round = val_stats["accuracy"] > round_best_accuracy or (
                val_stats["accuracy"] == round_best_accuracy and val_stats["loss"] < round_best_loss
            )
            if improved_round:
                round_best_accuracy = val_stats["accuracy"]
                round_best_loss = val_stats["loss"]
                round_best_state = trainable_state_dict(classifier)
                round_best_optimizer = copy.deepcopy(optimizer.state_dict())
            improved_global = val_stats["accuracy"] > best_accuracy or (
                val_stats["accuracy"] == best_accuracy and val_stats["loss"] < best_loss
            )
            if improved_global:
                best_accuracy = val_stats["accuracy"]
                best_loss = val_stats["loss"]
                best_epoch = global_epoch
                best_state = trainable_state_dict(classifier)
                save_checkpoint(out / "best_model.pt", classifier, class_names, config, quality)
                (out / "metrics_progress.json").write_text(
                    json.dumps(
                        {
                            "selected_epoch": best_epoch,
                            "selected_val_loss": best_loss,
                            "selected_val_noisy_accuracy": best_accuracy,
                            "history": history,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        # Continue the next screening round from the best epoch of this round,
        # including matching AdamW momentum state.
        load_classifier_state(classifier, round_best_state)
        optimizer.load_state_dict(round_best_optimizer)

        if round_id + 1 < config.rounds:
            model_scores = np.full(len(rows), np.nan, dtype=np.float32)
            classifier.eval()
            with torch.no_grad():
                class_weights = torch.nn.functional.normalize(classifier.classifier, dim=-1)
                for batch in screen_loader:
                    idx = batch["row_index"].numpy()
                    y = batch["label"].to(device)
                    _, _, _, adapted = classifier.forward_features(
                        batch["features"].to(device, non_blocking=True), None
                    )
                    model_scores[idx] = (adapted * class_weights[y]).sum(dim=-1).cpu().numpy()
            model_rank_scores = classwise_rank_scores(labels, model_scores, train_idx)
            quality_scores[train_idx] = (
                config.score_momentum * quality_scores[train_idx]
                + (1.0 - config.score_momentum) * model_rank_scores[train_idx]
            )
            next_fraction = clean_fraction_for_round(config, round_id + 1)
            quality, clean = classwise_quality(
                rows,
                quality_scores,
                next_fraction,
                config.weight_floor,
                candidate_indices=train_idx,
                low_confidence_multiplier=config.low_confidence_multiplier,
            )
            print(
                f"rescreen round={round_id + 1} clean_fraction={next_fraction:.3f} "
                f"clean={int(clean.sum())}/{len(train_idx)} train_quality_mean={quality[train_idx].mean():.4f}"
            )

    load_classifier_state(classifier, best_state)
    selected_stats = evaluate_classifier(classifier, val_loader, device)
    print(
        f"selected_epoch={best_epoch} val_loss={selected_stats['loss']:.5f} "
        f"val_acc={selected_stats['accuracy']:.4f}"
    )

    save_checkpoint(out / "model.pt", classifier, class_names, config, quality)
    (out / "classes.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "selected_epoch": best_epoch,
                "selected_val_loss": selected_stats["loss"],
                "selected_val_noisy_accuracy": selected_stats["accuracy"],
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(out / "sample_quality.npy", quality)
    print(f"saved={out / 'model.pt'}")


if __name__ == "__main__":
    main()
