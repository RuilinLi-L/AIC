"""V13 recipes and training-only, periodically refreshed label repair."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


RECIPES = ("strength", "dynamic", "expanded")


def recipe_config(recipe):
    if recipe not in (*RECIPES, "v12_fallback"):
        raise ValueError(f"unknown V13 recipe: {recipe}")
    fallback = recipe == "v12_fallback"
    dynamic = recipe in ("dynamic", "expanded")
    return {
        "recipe": recipe, "epochs": 12 if fallback else 24,
        "schedule_epochs": 12 if fallback else 24,
        "lora_lr": 1e-5 if fallback else 1e-4,
        "head_lr": 5e-5 if fallback else 5e-4,
        "lora_layers": 12 if recipe == "expanded" else 4,
        "lora_rank": 16, "lora_alpha": 32.0, "lora_dropout": 0.0,
        "lora_targets": ("q_proj", "k_proj", "v_proj", "out_proj"),
        "layer_lr_decay": 0.8 if recipe == "expanded" else 1.0,
        "dynamic_noise": dynamic, "quality_floor": 0.05 if dynamic else 0.25,
        "repair_confidence": 0.70 if dynamic else 0.80,
        "repair_fraction": 0.15, "repair_start_epoch": 5,
        "repair_ramp_end_epoch": 8 if dynamic else (12 if fallback else 24),
        "repair_max_weight": 0.5, "neighbor_support_threshold": 0.5,
        "noise_audit": "epochs_1_2_then_every_2" if dynamic else "epochs_1_2",
    }


def build_optimizer(classifier, config):
    groups, visual, head = {}, [], []
    for name, parameter in classifier.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("clip."):
            match = re.search(r"encoder\.layers\.(\d+)\.", name)
            if match is None or not name.endswith(("lora_a", "lora_b")):
                raise ValueError(f"unexpected unfrozen backbone parameter: {name}")
            layer = int(match.group(1))
            lr = config["lora_lr"] * config["layer_lr_decay"] ** (11 - layer)
            groups.setdefault(layer, {"params": [], "lr": lr, "layer": layer})["params"].append(parameter)
            visual.append(parameter)
        else:
            head.append(parameter)
    if not visual or not head:
        raise ValueError("V13 requires trainable LoRA and classifier parameters")
    optimizer = torch.optim.AdamW(
        [groups[layer] for layer in sorted(groups)] + [{"params": head, "lr": config["head_lr"]}],
        weight_decay=1e-4,
    )
    return optimizer, visual, head


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(model_dir):
    root = Path(model_dir).resolve()
    if (root / "0_CLIPModel").is_dir():
        root = root / "0_CLIPModel"
    weights = sorted([*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")])
    if not weights or not (root / "config.json").is_file():
        raise FileNotFoundError("V13 requires local official CLIP config and weights")
    files = [root / "config.json", *weights]
    return {p.name: file_sha256(p) for p in files}


def pool_signature(dataset_signature, train_indices, validation_indices):
    payload = {"dataset": dataset_signature, "train": list(map(int, train_indices)),
               "validation": list(map(int, validation_indices)), "version": 13}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def stage_indices(manifest, stage):
    if stage not in ("validate", "refit"):
        raise ValueError("stage must be validate or refit")
    train = list(manifest.final_indices if stage == "refit" else manifest.train_indices)
    validation = [] if stage == "refit" else list(manifest.validation_indices)
    if set(train) & set(validation) or len(train) != len(set(train)):
        raise ValueError("training/validation overlap or duplicate training indices")
    supervised = [i for i in train if manifest.clean_mask[i]]
    return train, supervised, validation


def should_audit(epoch, dynamic):
    return epoch in (1, 2) or (dynamic and epoch % 2 == 0)


def repair_weight(epoch):
    return 0.5 * min(1.0, max(0.0, (epoch - 4) / 4.0))


def dynamic_repair_plan(labels, indices, prototype_labels, neighbor, previous_prob,
                        current_prob, view_consensus, *, confidence=0.70, fraction=0.15):
    """Only audited training rows can be corrected; retain original labels on disk."""
    labels = np.asarray(labels)
    ids = np.asarray(indices, dtype=np.int64)
    current = np.asarray(current_prob, dtype=np.float32)
    previous = np.asarray(previous_prob, dtype=np.float32)
    if current.shape != previous.shape or current.shape[0] != len(labels):
        raise ValueError("audit probability shapes differ")
    if not np.isfinite(current).all() or not np.isfinite(previous).all():
        raise ValueError("nonfinite audit probabilities")
    p = current / current.sum(1, keepdims=True).clip(1e-8)
    predicted, conf = p.argmax(1), p.max(1)
    evidence = (predicted == prototype_labels) | (
        (predicted == neighbor.top_label) & (neighbor.top_support >= 0.5))
    eligible = ((predicted != labels) & (previous.sum(1) > 0)
                & (previous.argmax(1) == predicted) & (view_consensus == predicted)
                & (conf >= confidence) & evidence)
    plan = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels[ids]):
        members = ids[labels[ids] == label]
        candidates = members[eligible[members]]
        cap = max(1, math.floor(len(members) * fraction))
        # Stable confidence ranking makes ties reproducible across resume.
        order = np.argsort(-conf[candidates], kind="stable")
        plan[candidates[order[:cap]]] = True
    return plan


def repaired_soft_loss(logits_a, logits_b, probabilities, mask, weight):
    """Detached targets; old-label losses are masked by the caller."""
    target = probabilities.float().detach()
    target = target / target.sum(1, keepdim=True).clamp_min(1e-8)
    ce = -0.5 * ((target * F.log_softmax(logits_a.float(), dim=1)).sum(1)
                 + (target * F.log_softmax(logits_b.float(), dim=1)).sum(1))
    return float(weight) * ce * mask.float()


def repair_diagnostics(labels, indices, previous_plan, plan, old_quality, quality,
                       probabilities, classes):
    ids = np.asarray(indices, dtype=np.int64)
    prediction = np.asarray(probabilities).argmax(1)
    audited = np.asarray(probabilities).sum(1) > 0
    count = lambda mask: np.bincount(labels[ids][mask[ids]], minlength=classes).tolist()
    return {"repaired_by_class": count(plan),
            "added_by_class": count(plan & ~previous_plan),
            "withdrawn_by_class": count(previous_plan & ~plan),
            "teacher_disagreement_by_class": count(audited & (prediction != labels)),
            "quality_mean": float(quality[ids].mean()),
            "quality_mean_delta": float((quality[ids] - old_quality[ids]).mean())}
