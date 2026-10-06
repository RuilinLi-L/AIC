"""V19's fixed rank-32 resolution and training-dropout candidates."""

import re
from copy import deepcopy

import torch

from v15_core import (
    dynamic_repair_plan, file_sha256, model_identity, pool_signature,
    repair_diagnostics, repair_weight, repaired_soft_loss, should_audit, stage_indices,
    ROBUST_AUGMENTATION, MLP_CONFIG_KEYS,
)
from v15_core import recipe_config as baseline_recipe


RECIPES = ("resolution384_rank32", "rank32_dropout", "rank32_control")
V19_CONFIG_KEYS = (
    "image_size", "zoom_shortest_edge", "interpolate_pos_encoding",
    "frozen_feature_image_size",
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
        raise ValueError(f"unknown V19 recipe: {recipe}")
    dropout = 0.05 if recipe == "rank32_dropout" else 0.0
    return {
        **deepcopy(FROZEN_TRAINING_CONFIG), **baseline_recipe("expanded_mlp"), "recipe": recipe,
        "image_size": 384 if recipe == "resolution384_rank32" else 320,
        "zoom_shortest_edge": 439 if recipe == "resolution384_rank32" else 366,
        "interpolate_pos_encoding": True, "frozen_feature_image_size": 224,
        "lora_rank": 32, "lora_alpha": 64.0, "lora_dropout": dropout,
        "mlp_lora_rank": 32, "mlp_lora_alpha": 64.0, "mlp_lora_dropout": dropout,
    }


def validate_recipe_augmentation(config):
    expected = recipe_config(config["recipe"])
    if config.get("augmentation") != expected["augmentation"]:
        raise ValueError("V19 recipe conflicts with augmentation")
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
            raise ValueError(f"V19 checkpoint recipe conflicts with {key}")
    if config.get("layernorm_layers", 0) != 0:
        raise ValueError("V19 freezes every backbone LayerNorm")
    for key in ("agreement_recovery", "dynamic_prototype", "activation_checkpointing"):
        if config.get(key, False) is not False:
            raise ValueError(f"V19 does not enable {key}")
    if "stage" in config:
        stage = config["stage"]
        if stage not in ("validate", "refit"):
            raise ValueError("V19 stage must be validate or refit")
        stage_expected = {
            "validation_split": "none_refit_all_manifest_rows" if stage == "refit" else "manifest_clean_tail_safe_capped_8_percent",
            "epoch_selection": "frozen_selection" if stage == "refit" else "fixed_equal_center_hflip_macro",
        }
        for key, value in stage_expected.items():
            if key in config and config[key] != value:
                raise ValueError(f"V19 checkpoint stage conflicts with {key}")
    return expected


def build_optimizer(classifier, config):
    """Keep V15's optimizer and layer rates, checking the 72-module boundary."""
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
    if len(groups) != 12 or len(visual) != 144 or not head:
        raise ValueError("V19 requires all twelve LoRA layers and classifier parameters")
    parameters = [groups[layer] for layer in sorted(groups)] + [{"params": head, "lr": config["head_lr"]}]
    return torch.optim.AdamW(parameters, weight_decay=1e-4), visual, head
