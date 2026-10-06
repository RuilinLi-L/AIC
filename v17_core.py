"""Independent V17 supervision recovery and training-only dynamic prototype repair."""

import math
import re

import numpy as np
import torch

from v15_core import (
    dynamic_repair_plan, file_sha256, model_identity, pool_signature,
    repair_diagnostics, repair_weight, repaired_soft_loss, should_audit, stage_indices,
    ROBUST_AUGMENTATION, MLP_CONFIG_KEYS,
)
from v15_core import recipe_config as baseline_recipe


RECIPES = ("agreement_recovery", "dynamic_prototype")
V17_CONFIG_KEYS = (
    "agreement_recovery", "recovery_confidence", "recovery_start_epoch",
    "recovery_ramp_end_epoch", "recovery_max_fraction", "recovery_static_support",
    "dynamic_prototype", "dynamic_prototype_folds", "dynamic_reference_confidence",
    "dynamic_min_references", "dynamic_repair_confidence", "dynamic_prototype_margin",
    "dynamic_feature_source",
)


def recipe_config(recipe):
    if recipe not in RECIPES:
        raise ValueError(f"unknown V17 recipe: {recipe}")
    return {
        **baseline_recipe("expanded_mlp"), "recipe": recipe,
        "agreement_recovery": recipe == "agreement_recovery",
        "recovery_confidence": 0.90, "recovery_start_epoch": 5,
        "recovery_ramp_end_epoch": 8, "recovery_max_fraction": 0.50,
        "recovery_static_support": True,
        "dynamic_prototype": recipe == "dynamic_prototype",
        "dynamic_prototype_folds": 5, "dynamic_reference_confidence": 0.80,
        "dynamic_min_references": 3, "dynamic_repair_confidence": 0.70,
        "dynamic_prototype_margin": 0.03,
        "dynamic_feature_source": "ema_normalized_center_hflip_adapted_mean",
    }


def validate_recipe_augmentation(config):
    expected = recipe_config(config["recipe"])
    if config.get("augmentation") != expected["augmentation"]:
        raise ValueError("V17 recipe conflicts with augmentation")
    return expected


def validate_recipe_config(config):
    expected = validate_recipe_augmentation(config)
    for key, value in expected.items():
        actual = config.get(key)
        if key in ("lora_targets", "mlp_lora_targets") and actual is not None:
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"V17 checkpoint recipe conflicts with {key}")
    if config.get("layernorm_layers", 0) != 0:
        raise ValueError("V17 freezes every backbone LayerNorm")
    return expected


def build_optimizer(classifier, config):
    validate_recipe_config(config)
    groups, visual, head = {}, [], []
    for name, parameter in classifier.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("clip."):
            match = re.fullmatch(
                r"clip\.vision_model\.encoder\.layers\.(\d+)\.(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.(?:fc1|fc2))\.lora_[ab]", name)
            if match is None or int(match.group(1)) not in range(12):
                raise ValueError(f"unexpected unfrozen backbone parameter: {name}")
            layer = int(match.group(1))
            lr = config["lora_lr"] * config["layer_lr_decay"] ** (11 - layer)
            groups.setdefault(layer, {"params": [], "lr": lr, "layer": layer})["params"].append(parameter)
            visual.append(parameter)
        else:
            head.append(parameter)
    if len(groups) != 12 or not head:
        raise ValueError("V17 requires all twelve LoRA layers and classifier parameters")
    parameters = [groups[layer] for layer in sorted(groups)] + [{"params": head, "lr": config["head_lr"]}]
    return torch.optim.AdamW(parameters, weight_decay=1e-4), visual, head


def _audit_inputs(labels, indices, previous_prob, current_prob, consensus):
    labels = np.asarray(labels, dtype=np.int64)
    ids = np.asarray(indices, dtype=np.int64)
    previous = np.asarray(previous_prob, dtype=np.float32)
    current = np.asarray(current_prob, dtype=np.float32)
    consensus = np.asarray(consensus, dtype=np.int64)
    if (previous.shape != current.shape or current.ndim != 2 or current.shape[0] != len(labels)
            or consensus.shape != labels.shape or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= len(labels))):
        raise ValueError("V17 audit shapes or training indices are invalid")
    if (not np.isfinite(current).all() or not np.isfinite(previous).all()
            or np.any(current < 0) or np.any(previous < 0)
            or np.any(labels < 0) or np.any(labels >= current.shape[1])):
        raise ValueError("V17 audit contains invalid probabilities or labels")
    previous = previous / previous.sum(1, keepdims=True).clip(1e-8)
    current = current / current.sum(1, keepdims=True).clip(1e-8)
    return labels, ids, previous, current, consensus


def agreement_recovery_quality(labels, indices, base_quality, repair_plan, prototype_labels,
                               neighbor, previous_prob, current_prob, view_consensus, *,
                               epoch, confidence=0.90, max_fraction=0.50,
                               start_epoch=5, ramp_end_epoch=8):
    """Restore only original-label supervision, never trust/regularizer quality."""
    labels, ids, previous, current, consensus = _audit_inputs(
        labels, indices, previous_prob, current_prob, view_consensus)
    quality = np.asarray(base_quality, dtype=np.float32).copy()
    repairs = np.asarray(repair_plan, dtype=np.bool_)
    if (quality.shape != labels.shape or repairs.shape != labels.shape
            or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1))):
        raise ValueError("V17 recovery quality or repair mask is invalid")
    recovered = np.zeros(len(labels), dtype=np.bool_)
    if epoch < start_epoch:
        return quality, recovered
    original = np.arange(len(labels)), labels
    static = (np.asarray(prototype_labels) == labels) | (
        (neighbor.top_label == labels) & (neighbor.label_support >= 0.50))
    eligible = ((previous.argmax(1) == labels) & (current.argmax(1) == labels)
                & (previous[original] >= confidence) & (current[original] >= confidence)
                & (consensus == labels) & static & ~repairs)
    recovered[ids] = eligible[ids]
    fraction = max_fraction * min(1.0, max(0.0, (epoch - start_epoch + 1) / max(1, ramp_end_epoch - start_epoch + 1)))
    quality[recovered] += float(fraction) * (1.0 - quality[recovered])
    return quality, recovered


def cross_fitted_dynamic_prototypes(features, labels, indices, fold_ids, trusted,
                                    previous_prob, current_prob, *, num_classes,
                                    folds=5, confidence=0.80, min_references=3,
                                    chunk_size=2048):
    """References exclude every query's fold; missing classes supply no evidence."""
    labels, ids, previous, current, _ = _audit_inputs(
        labels, indices, previous_prob, current_prob, np.full(len(labels), -1))
    features = np.asarray(features, dtype=np.float32)
    fold_ids = np.asarray(fold_ids)
    trusted = np.asarray(trusted, dtype=np.bool_)
    if (features.ndim != 2 or features.shape[0] != len(labels) or not np.isfinite(features).all()
            or fold_ids.shape != labels.shape or trusted.shape != labels.shape
            or current.shape[1] != num_classes or min_references < 1 or folds < 2
            or np.any((fold_ids[ids] < 0) | (fold_ids[ids] >= folds))):
        raise ValueError("V17 dynamic prototype inputs are invalid")
    features = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-8)
    original = np.arange(len(labels)), labels
    reference_mask = np.zeros(len(labels), dtype=np.bool_)
    reference_mask[ids] = (trusted[ids] & (previous.argmax(1)[ids] == labels[ids])
                           & (current.argmax(1)[ids] == labels[ids])
                           & (previous[original][ids] >= confidence)
                           & (current[original][ids] >= confidence)
                           & (np.linalg.norm(features[ids], axis=1) > 0))
    prototypes = np.zeros((folds, num_classes, features.shape[1]), dtype=np.float32)
    counts = np.zeros((folds, num_classes), dtype=np.int64)
    prediction = np.full(len(labels), -1, dtype=np.int64)
    margins = np.zeros(len(labels), dtype=np.float32)
    valid = np.zeros(len(labels), dtype=np.bool_)
    for fold in range(folds):
        refs = ids[reference_mask[ids] & (fold_ids[ids] != fold)]
        np.add.at(prototypes[fold], labels[refs], features[refs])
        np.add.at(counts[fold], labels[refs], 1)
        norms = np.linalg.norm(prototypes[fold], axis=1)
        available = (counts[fold] >= min_references) & (norms > 1e-8)
        prototypes[fold] /= norms[:, None].clip(1e-8)
        prototypes[fold, ~available] = 0
        # A sole available class cannot establish a meaningful top-two margin.
        if available.sum() < 2:
            continue
        query = ids[fold_ids[ids] == fold]
        for start in range(0, len(query), chunk_size):
            local = query[start:start + chunk_size]
            scores = features[local] @ prototypes[fold].T
            scores[:, ~available] = -np.inf
            top = scores.argmax(1)
            top_two = np.partition(scores, -2, axis=1)[:, -2:]
            margin = top_two.max(1) - top_two.min(1)
            nonzero = np.linalg.norm(features[local], axis=1) > 0
            prediction[local[nonzero]] = top[nonzero]
            margins[local[nonzero]] = margin[nonzero]
            valid[local[nonzero]] = True
    return {"dynamic_prototypes": prototypes, "dynamic_reference_counts": counts,
            "dynamic_reference_mask": reference_mask, "dynamic_prototype_label": prediction,
            "dynamic_prototype_margin": margins, "dynamic_prototype_valid": valid}


def dynamic_prototype_repair_plan(labels, indices, prototype_labels, neighbor,
                                  previous_prob, current_prob, view_consensus,
                                  dynamic_labels, dynamic_margin, dynamic_valid, *,
                                  confidence=0.70, margin_threshold=0.03, fraction=0.15):
    """Union static/new eligibility, then apply exactly one original-class cap."""
    labels, ids, previous, current, consensus = _audit_inputs(
        labels, indices, previous_prob, current_prob, view_consensus)
    dynamic_labels = np.asarray(dynamic_labels)
    dynamic_margin = np.asarray(dynamic_margin)
    dynamic_valid = np.asarray(dynamic_valid, dtype=np.bool_)
    if (dynamic_labels.shape != labels.shape or dynamic_margin.shape != labels.shape
            or dynamic_valid.shape != labels.shape or not np.isfinite(dynamic_margin).all()
            or np.any(dynamic_valid & ((dynamic_labels < 0) | (dynamic_labels >= current.shape[1])))):
        raise ValueError("V17 dynamic repair evidence is invalid")
    predicted, conf = current.argmax(1), current.max(1)
    temporal = ((predicted != labels) & (previous.sum(1) > 0)
                & (previous.argmax(1) == predicted) & (consensus == predicted))
    static_support = (predicted == prototype_labels) | (
        (predicted == neighbor.top_label) & (neighbor.top_support >= 0.50))
    static = temporal & (conf >= confidence) & static_support
    dynamic = (temporal & (conf >= confidence) & (previous.max(1) >= confidence)
               & dynamic_valid & (predicted == dynamic_labels)
               & (dynamic_margin >= margin_threshold))
    eligible = static | dynamic
    plan = np.zeros(len(labels), dtype=np.bool_)
    for label in np.unique(labels[ids]):
        members = ids[labels[ids] == label]
        candidates = members[eligible[members]]
        # Preserve static evidence on confidence ties, then use manifest row order.
        order = np.lexsort((candidates, ~static[candidates], -conf[candidates]))
        # B's approved cap is literal; classes below seven rows get no repair.
        plan[candidates[order[:math.floor(len(members) * fraction)]]] = True
    dynamic_only = plan & dynamic & ~static
    selected_dynamic_ids = ids[dynamic_only[ids]]
    return plan, {"static_eligible": int(static[ids].sum()),
                  "dynamic_eligible": int(dynamic[ids].sum()),
                  "dynamic_only_eligible": int((dynamic & ~static)[ids].sum()),
                  "dynamic_only_selected": int(dynamic_only[ids].sum()),
                  "dynamic_only_selected_by_class": np.bincount(
                      labels[selected_dynamic_ids], minlength=current.shape[1]).tolist(),
                  "dynamic_only_selected_by_target_class": np.bincount(
                      predicted[selected_dynamic_ids], minlength=current.shape[1]).tolist()}
