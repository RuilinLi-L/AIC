"""V20's isolated DoRA and LoRA+ recipes and stable optimizer contract."""

import re
from copy import deepcopy

import torch

from v15_core import (
    dynamic_repair_plan, file_sha256, model_identity, pool_signature,
    repair_diagnostics, repair_weight, repaired_soft_loss, should_audit, stage_indices,
    ROBUST_AUGMENTATION, MLP_CONFIG_KEYS,
)
from v15_core import recipe_config as baseline_recipe


RECIPES = ("dora_rank32", "loraplus_rank32")
V20_CONFIG_KEYS = (
    "image_size", "zoom_shortest_edge", "interpolate_pos_encoding",
    "frozen_feature_image_size", "method", "lora_lr_a", "lora_lr_b", "magnitude_lr",
)

# These values are implemented in the training losses and schedule. Recording a
# different value must fail, rather than silently claiming a changed experiment.
FROZEN_TRAINING_CONFIG = {
    "seed": 2026, "batch_size": 256, "gradient_accumulation": 1,
    "warmup_epochs": 2, "bottleneck": 128, "weight_decay": 1e-4,
    "warmup_ratio": 0.05, "initial_logit_scale": 30.0,
    "max_logit_scale": 50.0, "learn_logit_scale": False,
    "ema_decay": 0.999, "consistency_weight": 0.05,
    "anchor_weight": 0.10, "contrastive_weight": 0.10,
    "contrastive_temperature": 0.07, "gce_q": 0.7,
    "quality_threshold": 0.70, "prototype_temperature": 0.07,
    "prototype_keep_fraction": 0.70, "prototype_iterations": 2,
    "prototype_folds": 5, "queue_per_class": 16,
    "temporal_decay": 0.85, "consistency_temperature": 2.0,
    "gradient_projection": "uncertain_orthogonal_to_trusted_on_negative_dot",
    "repair_margin_quantile": 0.75,
    "fixed_feature_views": ["center", "hflip", "light_seed_2", "light_seed_3"],
    "partial_weight": 0.25, "partial_prototype_mix": 0.75,
    "partial_temperature": 0.07, "partial_max_probability": 0.80,
    "repeat_factor": "min(4,sqrt(median_count/class_count))",
    "neighbor_method": "four_view_cosine_32_crossfit_5fold",
    "neighbor_quality_mix": [0.75, 0.25], "soft_teacher": True,
    "soft_teacher_cutoff": 0.55, "soft_teacher_weight": 0.10,
    "conflict_policy": "partial", "sampler": "repeat-factor",
    "loss_precision": "fp32",
    "layernorm_layers": 0, "agreement_recovery": False,
    "dynamic_prototype": False, "activation_checkpointing": False,
}


def recipe_config(recipe):
    if recipe not in RECIPES:
        raise ValueError(f"unknown V20 recipe: {recipe}")
    method = "dora" if recipe == "dora_rank32" else "loraplus"
    return {
        **deepcopy(FROZEN_TRAINING_CONFIG), **baseline_recipe("expanded_mlp"), "recipe": recipe,
        "method": method,
        "image_size": 320, "zoom_shortest_edge": 366,
        "interpolate_pos_encoding": True, "frozen_feature_image_size": 224,
        "lora_rank": 32, "lora_alpha": 64.0, "lora_dropout": 0.0,
        "mlp_lora_rank": 32, "mlp_lora_alpha": 64.0, "mlp_lora_dropout": 0.0,
        # Retain the historical reference LR; actual A/B rates are explicit.
        "lora_lr": 1e-4, "lora_lr_a": 5e-5,
        "lora_lr_b": 5e-5 if method == "dora" else 2e-4,
        "magnitude_lr": 5e-5 if method == "dora" else 0.0,
        "head_lr": 5e-4, "layer_lr_decay": 0.8,
        "epochs": 24, "schedule_epochs": 24,
    }


def validate_recipe_augmentation(config):
    expected = recipe_config(config["recipe"])
    if config.get("augmentation") != expected["augmentation"]:
        raise ValueError("V20 recipe conflicts with augmentation")
    return expected


def validate_recipe_config(config):
    """Validate every recorded training and model setting of either recipe."""
    expected = validate_recipe_augmentation(config)
    for key, value in expected.items():
        actual = config.get(key)
        if key in ("lora_targets", "mlp_lora_targets") and actual is not None:
            actual = tuple(actual)
        if (actual != value or (isinstance(value, bool) and actual is not value)
                or (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and isinstance(actual, bool))):
            raise ValueError(f"V20 checkpoint recipe conflicts with {key}")
    if config.get("layernorm_layers", 0) != 0:
        raise ValueError("V20 freezes every backbone LayerNorm")
    for key in ("agreement_recovery", "dynamic_prototype", "activation_checkpointing"):
        if config.get(key, False) is not False:
            raise ValueError(f"V20 does not enable {key}")
    if "stage" in config:
        stage = config["stage"]
        if stage not in ("validate", "refit"):
            raise ValueError("V20 stage must be validate or refit")
        stage_expected = {
            "validation_split": "none_refit_all_manifest_rows" if stage == "refit" else "manifest_clean_tail_safe_capped_8_percent",
            "epoch_selection": "frozen_selection" if stage == "refit" else "joint_epoch_tta_bias_nested_cv",
        }
        for key, value in stage_expected.items():
            if key in config and config[key] != value:
                raise ValueError(f"V20 checkpoint stage conflicts with {key}")
    return expected


def build_optimizer(classifier, config):
    """Assign every trainable tensor exactly once, with reproducible group order."""
    validate_recipe_config(config)
    groups, visual, head = {}, [], []
    expected_kinds = ("a", "b", "m") if config["method"] == "dora" else ("a", "b")
    expected_modules = {
        f"clip.vision_model.encoder.layers.{layer}.{section}.{target}"
        for layer in range(12)
        for section, targets in (("self_attn", ("q_proj", "k_proj", "v_proj", "out_proj")),
                                 ("mlp", ("fc1", "fc2")))
        for target in targets
    }
    expected_names = {f"{module}.lora_{kind}" for module in expected_modules for kind in expected_kinds}
    visual_names, head_names = [], []
    for name, parameter in sorted(classifier.named_parameters()):
        if not parameter.requires_grad:
            continue
        if name.startswith("clip."):
            if name not in expected_names:
                raise ValueError(f"unexpected unfrozen backbone parameter: {name}")
            match = re.fullmatch(r"clip\.vision_model\.encoder\.layers\.(\d+)\..*\.lora_([abm])", name)
            layer, kind = int(match.group(1)), match.group(2)
            base_lr = config["magnitude_lr"] if kind == "m" else config[f"lora_lr_{kind}"]
            lr = base_lr * config["layer_lr_decay"] ** (11 - layer)
            group = groups.setdefault((layer, kind), {
                "params": [], "param_names": [], "lr": lr, "initial_lr": lr,
                "layer": layer, "kind": kind, "group_name": f"visual.{layer:02d}.{kind}",
                "weight_decay": 0.0 if kind == "m" else 1e-4,
            })
            group["params"].append(parameter)
            group["param_names"].append(name)
            visual.append(parameter)
            visual_names.append(name)
        else:
            head.append(parameter)
            head_names.append(name)
    expected_head = {"classifier", "adapter.scale", "adapter.net.0.weight", "adapter.net.0.bias",
                     "adapter.net.3.weight", "adapter.net.3.bias"}
    if set(visual_names) != expected_names or set(head_names) != expected_head:
        raise ValueError("V20 requires all 72 adapter modules and the complete classifier head")
    parameters = [groups[key] for key in sorted(groups)] + [{
        "params": head, "param_names": head_names, "lr": config["head_lr"],
        "initial_lr": config["head_lr"], "weight_decay": 1e-4,
        "layer": None, "kind": "head", "group_name": "head",
    }]
    all_parameters = [parameter for group in parameters for parameter in group["params"]]
    if len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
        raise ValueError("V20 optimizer groups contain duplicate parameters")
    optimizer = torch.optim.AdamW(parameters, weight_decay=1e-4)
    return optimizer, visual, head


def optimizer_group_metadata(optimizer):
    """Record stable topology and initial rates, independent of scheduler progress."""
    result, assigned = [], set()
    for order, group in enumerate(optimizer.param_groups):
        names = list(group.get("param_names", ()))
        if (not names or len(names) != len(group["params"]) or names != sorted(names)
                or len(names) != len(set(names)) or assigned.intersection(names)):
            raise ValueError("V20 optimizer param_names are missing, duplicated or out of order")
        if "initial_lr" not in group:
            raise ValueError("V20 optimizer group is missing initial_lr")
        assigned.update(names)
        result.append({
            "order": order, "group_name": group["group_name"],
            "layer": group["layer"], "kind": group["kind"], "param_names": names,
            "initial_lr": float(group["initial_lr"]), "weight_decay": float(group["weight_decay"]),
        })
    return result
