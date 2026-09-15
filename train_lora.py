"""Aggressive single-model V4 training: visual LoRA + EMA + label refurbishment.

Only the official training split is used.  Frozen CLIP features initialize the
classifier, estimate label quality, and anchor the LoRA-tuned representation.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from robust_clip import (
    FolderImageDataset,
    RobustCLIPClassifier,
    TrainConfig,
    classwise_quality,
    classwise_rank_scores,
    effective_number_class_weights,
    inject_visual_lora,
    list_images,
    load_classifier_state,
    load_clip,
    resolve_device,
    robust_visual_prototypes_and_scores,
    save_checkpoint,
    seed_everything,
    stratified_split,
)
from train import load_or_create_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--model-dir", default="clip-ViT-B-32")
    parser.add_argument("--output-dir", default="outputs/robust_visual_v4_lora")
    parser.add_argument("--feature-cache", default="outputs/frozen_clip_train.npy")
    parser.add_argument("--head-init-checkpoint", default=None, help="Optional V3 head/adapter warm start")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--augmentation", choices=["none", "light", "strong"], default="light")
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--head-lr", type=float, default=8e-5)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument("--tune-layernorm", action="store_true")
    parser.add_argument("--tune-visual-projection", action="store_true")
    parser.add_argument("--clean-fraction", type=float, default=0.80)
    parser.add_argument("--weight-floor", type=float, default=0.20)
    parser.add_argument("--class-weight-beta", type=float, default=0.9999)
    parser.add_argument("--weight-cap", type=float, default=2.0)
    parser.add_argument("--prototype-keep-fraction", type=float, default=0.70)
    parser.add_argument("--prototype-iterations", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--anchor-weight", type=float, default=0.10)
    parser.add_argument("--pseudo-start-epoch", type=int, default=2)
    parser.add_argument("--pseudo-threshold", type=float, default=0.70)
    parser.add_argument("--pseudo-weight", type=float, default=0.60)
    parser.add_argument("--temporal-decay", type=float, default=0.80)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--trusted-val-fraction", type=float, default=0.70)
    parser.add_argument("--initial-logit-scale", type=float, default=30.0)
    parser.add_argument("--max-logit-scale", type=float, default=50.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    targets = tuple(part.strip() for part in args.lora_targets.split(",") if part.strip())
    if not targets:
        raise ValueError("--lora-targets cannot be empty")
    if args.batch_size < 1 or args.epochs < 1 or args.gradient_accumulation < 1:
        raise ValueError("batch-size, epochs and gradient-accumulation must be positive")
    if args.lora_rank < 1 or args.lora_layers < 1:
        raise ValueError("LoRA rank and layer count must be positive")
    for name in ("clean_fraction", "prototype_keep_fraction", "pseudo_threshold", "trusted_val_fraction"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    for name in ("label_smoothing", "pseudo_weight", "temporal_decay", "ema_decay", "warmup_ratio"):
        value = float(getattr(args, name))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    return targets


def enable_extra_visual_parameters(classifier: RobustCLIPClassifier, last_n: int, layernorm: bool, projection: bool) -> list[str]:
    names: list[str] = []
    layers = classifier.clip.vision_model.encoder.layers
    if layernorm:
        for layer_id in range(len(layers) - last_n, len(layers)):
            for norm_name in ("layer_norm1", "layer_norm2"):
                for parameter_name, parameter in getattr(layers[layer_id], norm_name).named_parameters():
                    parameter.requires_grad_(True)
                    names.append(f"vision_model.encoder.layers.{layer_id}.{norm_name}.{parameter_name}")
    if projection:
        for parameter_name, parameter in classifier.clip.visual_projection.named_parameters():
            parameter.requires_grad_(True)
            names.append(f"visual_projection.{parameter_name}")
    return names


def make_config(args: argparse.Namespace, targets: tuple[str, ...]) -> TrainConfig:
    return TrainConfig(
        model_dir=args.model_dir,
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        rounds=1,
        lr=args.head_lr,
        weight_decay=args.weight_decay,
        bottleneck=args.bottleneck,
        clean_fraction=args.clean_fraction,
        weight_floor=args.weight_floor,
        class_weight_beta=args.class_weight_beta,
        weight_cap=args.weight_cap,
        augmentation=args.augmentation,
        prototype_keep_fraction=args.prototype_keep_fraction,
        prototype_iterations=args.prototype_iterations,
        initial_logit_scale=args.initial_logit_scale,
        max_logit_scale=args.max_logit_scale,
        learn_logit_scale=False,
        device=args.device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        lora_targets=targets,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        anchor_weight=args.anchor_weight,
        label_smoothing=args.label_smoothing,
        pseudo_start_epoch=args.pseudo_start_epoch,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_weight=args.pseudo_weight,
        temporal_decay=args.temporal_decay,
        ema_decay=args.ema_decay,
        gradient_accumulation=args.gradient_accumulation,
        tune_layernorm=args.tune_layernorm,
        tune_visual_projection=args.tune_visual_projection,
        trusted_val_fraction=args.trusted_val_fraction,
    )


def trainable_parameters(classifier: RobustCLIPClassifier) -> dict[str, torch.nn.Parameter]:
    return {name: value for name, value in classifier.named_parameters() if value.requires_grad}


def clone_trainable(classifier: RobustCLIPClassifier) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in trainable_parameters(classifier).items()}


def update_ema(shadow: dict[str, torch.Tensor], classifier: RobustCLIPClassifier, decay: float, step: int) -> None:
    effective_decay = min(decay, (1.0 + step) / (10.0 + step))
    for name, parameter in trainable_parameters(classifier).items():
        shadow[name].mul_(effective_decay).add_(parameter.detach(), alpha=1.0 - effective_decay)


@contextmanager
def swapped_trainable(classifier: RobustCLIPClassifier, state: dict[str, torch.Tensor]):
    parameters = trainable_parameters(classifier)
    backup = {name: value.detach().clone() for name, value in parameters.items()}
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(state[name])
    try:
        yield
    finally:
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(backup[name])


def build_optimizer(classifier: RobustCLIPClassifier, args: argparse.Namespace):
    visual, head = [], []
    for name, parameter in classifier.named_parameters():
        if not parameter.requires_grad:
            continue
        (visual if name.startswith("clip.") else head).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": visual, "lr": args.lora_lr},
            {"params": head, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    return optimizer, visual, head


def trusted_validation_mask(
    features: np.ndarray,
    labels: np.ndarray,
    val_indices: list[int],
    prototypes: np.ndarray,
    fraction: float,
) -> np.ndarray:
    indices = np.asarray(val_indices, dtype=np.int64)
    scores = np.sum(np.asarray(features[indices], dtype=np.float32) * prototypes[labels[indices]], axis=1)
    trusted = np.zeros(len(indices), dtype=np.bool_)
    val_labels = labels[indices]
    for label in np.unique(val_labels):
        local = np.where(val_labels == label)[0]
        keep = max(1, int(math.ceil(len(local) * fraction)))
        trusted[local[np.argsort(scores[local])[::-1][:keep]]] = True
    return trusted


@torch.no_grad()
def evaluate(
    classifier: RobustCLIPClassifier,
    loader: DataLoader,
    device: torch.device,
    trusted_mask: np.ndarray,
    save_logits: Path | None = None,
) -> dict[str, float]:
    classifier.eval()
    logits_batches, labels_batches = [], []
    amp = device.type == "cuda"
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = classifier(pixels, None)[0]
        logits_batches.append(logits.float().cpu())
        labels_batches.append(batch["label"])
    logits = torch.cat(logits_batches)
    labels = torch.cat(labels_batches)
    trusted = torch.as_tensor(trusted_mask, dtype=torch.bool)
    result = {
        "loss": float(F.cross_entropy(logits, labels)),
        "accuracy": float((logits.argmax(dim=1) == labels).float().mean()),
        "trusted_accuracy": float((logits[trusted].argmax(dim=1) == labels[trusted]).float().mean()),
        "trusted_rows": int(trusted.sum()),
    }
    if save_logits is not None:
        save_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_logits, logits.numpy())
    return result


def main() -> None:
    args = parse_args()
    targets = validate_args(args)
    config = make_config(args, targets)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = list_images(args.train_dir)
    labels = np.asarray([label for _, label, _ in rows], dtype=np.int64)
    class_names = [name for _, name in sorted({(label, name) for _, label, name in rows})]
    train_indices, val_indices = stratified_split(rows, args.val_ratio, args.seed)
    model, processor = load_clip(args.model_dir, device)
    features, cached_labels = load_or_create_feature_cache(
        model, processor, rows, config, device, Path(args.feature_cache)
    )
    if not np.array_equal(labels, cached_labels):
        raise ValueError("Feature-cache labels do not match the training folders")

    prototypes, prototype_scores = robust_visual_prototypes_and_scores(
        features,
        labels,
        len(class_names),
        candidate_indices=train_indices,
        keep_fraction=args.prototype_keep_fraction,
        iterations=args.prototype_iterations,
    )
    ranked_scores = classwise_rank_scores(labels, prototype_scores, train_indices)
    quality, clean = classwise_quality(
        rows,
        ranked_scores,
        args.clean_fraction,
        args.weight_floor,
        candidate_indices=train_indices,
    )
    trusted_mask = trusted_validation_mask(
        features, labels, val_indices, prototypes, args.trusted_val_fraction
    )

    classifier = RobustCLIPClassifier(
        model,
        classifier_init=torch.from_numpy(prototypes).to(device),
        bottleneck=args.bottleneck,
        initial_logit_scale=args.initial_logit_scale,
        max_logit_scale=args.max_logit_scale,
        learn_logit_scale=False,
    ).to(device)
    if args.head_init_checkpoint:
        head_path = Path(args.head_init_checkpoint)
        if head_path.is_file():
            head_checkpoint = torch.load(head_path, map_location="cpu")
            if list(head_checkpoint.get("class_names", [])) != class_names:
                raise ValueError("Head warm-start classes do not match train-dir")
            load_classifier_state(classifier, head_checkpoint["model"])
            print(f"head_warm_start={head_path.resolve()}", flush=True)
        else:
            print(
                f"head_warm_start_missing={head_path}; using robust visual prototypes",
                flush=True,
            )

    replaced = inject_visual_lora(
        classifier.clip,
        last_n_layers=args.lora_layers,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=targets,
    )
    extra = enable_extra_visual_parameters(
        classifier, args.lora_layers, args.tune_layernorm, args.tune_visual_projection
    )
    classifier.train()
    optimizer, visual_params, head_params = build_optimizer(classifier, args)
    steps_per_epoch = math.ceil(len(train_indices) / args.batch_size / args.gradient_accumulation)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-3, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    train_loader = DataLoader(
        FolderImageDataset(rows, train_indices, processor, augment=args.augmentation),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    val_loader = DataLoader(
        FolderImageDataset(rows, val_indices, processor, augment="none"),
        batch_size=max(args.batch_size, 32),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    class_counts = np.bincount(labels[np.asarray(train_indices)], minlength=len(class_names))
    class_weights = torch.tensor(
        effective_number_class_weights(class_counts, args.class_weight_beta, args.weight_cap),
        dtype=torch.float32,
        device=device,
    )
    temporal = np.zeros((len(rows), len(class_names)), dtype=np.float16)
    temporal_seen = np.zeros(len(rows), dtype=np.bool_)
    ema = clone_trainable(classifier)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    best_state = clone_trainable(classifier)
    best_score = -1.0
    best_full_accuracy = -1.0
    best_epoch = 0
    optimizer_step = 0

    print(
        f"device={device} train={len(train_indices)} val={len(val_indices)} trusted_val={int(trusted_mask.sum())} "
        f"lora_modules={len(replaced)} extra_visual_tensors={len(extra)} "
        f"trainable_visual={sum(p.numel() for p in visual_params):,} trainable_head={sum(p.numel() for p in head_params):,}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = running_correct = running_rows = pseudo_rows = 0.0
        for batch_id, batch in enumerate(train_loader, 1):
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["label"].to(device)
            idx = batch["row_index"].numpy()
            weights = torch.as_tensor(quality[idx], dtype=torch.float32, device=device) * class_weights[y]
            frozen = torch.tensor(np.asarray(features[idx]), dtype=torch.float32, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _, base, adapted = classifier(pixels, None)
                hard_loss = F.cross_entropy(
                    logits, y, reduction="none", label_smoothing=args.label_smoothing
                )
                sample_loss = hard_loss
                if epoch > args.pseudo_start_epoch:
                    old_seen = torch.as_tensor(temporal_seen[idx], dtype=torch.bool, device=device)
                    teacher = torch.tensor(np.asarray(temporal[idx]), dtype=torch.float32, device=device)
                    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    confidence, pseudo_label = teacher.max(dim=1)
                    eligible = (
                        old_seen
                        & torch.as_tensor(~clean[idx], dtype=torch.bool, device=device)
                        & (confidence >= args.pseudo_threshold)
                        & pseudo_label.ne(y)
                    )
                    soft_loss = -(teacher * F.log_softmax(logits, dim=1)).sum(dim=1)
                    repaired = (1.0 - args.pseudo_weight) * hard_loss + args.pseudo_weight * soft_loss
                    sample_loss = torch.where(eligible, repaired, hard_loss)
                    pseudo_rows += float(eligible.sum())
                loss_cls = (weights * sample_loss).sum() / weights.sum().clamp_min(1e-6)
                loss_anchor = (1.0 - (base * F.normalize(frozen, dim=1)).sum(dim=1)).mean()
                loss = loss_cls + args.anchor_weight * loss_anchor

            current_probs = logits.detach().softmax(dim=1).float().cpu().numpy()
            seen = temporal_seen[idx]
            if seen.any():
                temporal[idx[seen]] = (
                    args.temporal_decay * temporal[idx[seen]].astype(np.float32)
                    + (1.0 - args.temporal_decay) * current_probs[seen]
                ).astype(np.float16)
            if (~seen).any():
                temporal[idx[~seen]] = current_probs[~seen].astype(np.float16)
            temporal_seen[idx] = True

            scaler.scale(loss / args.gradient_accumulation).backward()
            should_step = batch_id % args.gradient_accumulation == 0 or batch_id == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(visual_params + head_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
                update_ema(ema, classifier, args.ema_decay, optimizer_step)

            running_loss += float(loss.detach()) * len(y)
            running_correct += float((logits.argmax(dim=1) == y).sum())
            running_rows += len(y)

        with swapped_trainable(classifier, ema):
            val_stats = evaluate(classifier, val_loader, device, trusted_mask)
        selection_score = 0.6 * val_stats["trusted_accuracy"] + 0.4 * val_stats["accuracy"]
        record = {
            "epoch": epoch,
            "train_loss": running_loss / running_rows,
            "train_noisy_accuracy": running_correct / running_rows,
            "pseudo_repaired_rows": int(pseudo_rows),
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
            "trusted_val_accuracy": val_stats["trusted_accuracy"],
            "selection_score": selection_score,
            "visual_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        print(
            f"epoch={epoch}/{args.epochs} train_loss={record['train_loss']:.5f} "
            f"train_acc={record['train_noisy_accuracy']:.4f} pseudo={int(pseudo_rows)} "
            f"val_acc={val_stats['accuracy']:.4f} trusted_acc={val_stats['trusted_accuracy']:.4f} "
            f"score={selection_score:.4f}",
            flush=True,
        )
        improved = selection_score > best_score or (
            selection_score == best_score and val_stats["accuracy"] > best_full_accuracy
        )
        if improved:
            best_score = selection_score
            best_full_accuracy = val_stats["accuracy"]
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in ema.items()}
            with swapped_trainable(classifier, best_state):
                save_checkpoint(out / "best_model.pt", classifier, class_names, config, quality)
            (out / "metrics_progress.json").write_text(
                json.dumps({"selected_epoch": best_epoch, "history": history}, indent=2), encoding="utf-8"
            )

    with torch.no_grad():
        for name, parameter in trainable_parameters(classifier).items():
            parameter.copy_(best_state[name])
    selected = evaluate(classifier, val_loader, device, trusted_mask, out / "val_logits.npy")
    save_checkpoint(out / "model.pt", classifier, class_names, config, quality)
    (out / "classes.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "selected_epoch": best_epoch,
                "selected": selected,
                "selection_score": best_score,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(out / "sample_quality.npy", quality)
    print(
        f"selected_epoch={best_epoch} val_acc={selected['accuracy']:.4f} "
        f"trusted_acc={selected['trusted_accuracy']:.4f} saved={out / 'model.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
