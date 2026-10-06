"""V18's independent resolution and LoRA-rank ablations of the V15 recipe."""

import re

import torch

from v15_core import (
    dynamic_repair_plan, file_sha256, model_identity, pool_signature,
    repair_diagnostics, repair_weight, repaired_soft_loss, should_audit, stage_indices,
    ROBUST_AUGMENTATION, MLP_CONFIG_KEYS,
)
from v15_core import recipe_config as baseline_recipe


RECIPES = ("resolution384", "rank32", "matched_control")
V18_CONFIG_KEYS = (
    "image_size", "zoom_shortest_edge", "interpolate_pos_encoding",
    "frozen_feature_image_size",
)


def recipe_config(recipe):
    if recipe not in RECIPES:
        raise ValueError(f"unknown V18 recipe: {recipe}")
    rank = 32 if recipe == "rank32" else 16
    return {
        **baseline_recipe("expanded_mlp"), "recipe": recipe,
        "image_size": 384 if recipe == "resolution384" else 320,
        "zoom_shortest_edge": 439 if recipe == "resolution384" else 366,
        "interpolate_pos_encoding": True, "frozen_feature_image_size": 224,
        "lora_rank": rank, "lora_alpha": float(2 * rank),
        "mlp_lora_rank": rank, "mlp_lora_alpha": float(2 * rank),
    }


def validate_recipe_augmentation(config):
    expected = recipe_config(config["recipe"])
    if config.get("augmentation") != expected["augmentation"]:
        raise ValueError("V18 recipe conflicts with augmentation")
    return expected


def validate_recipe_config(config):
    """Freeze recipe science; runtime freezes the recorded actual microbatch."""
    expected = validate_recipe_augmentation(config)
    for key, value in expected.items():
        actual = config.get(key)
        if key in ("lora_targets", "mlp_lora_targets") and actual is not None:
            actual = tuple(actual)
        if actual != value or (isinstance(value, bool) and actual is not value):
            raise ValueError(f"V18 checkpoint recipe conflicts with {key}")
    if config.get("layernorm_layers", 0) != 0:
        raise ValueError("V18 freezes every backbone LayerNorm")
    for key in ("agreement_recovery", "dynamic_prototype", "activation_checkpointing"):
        if config.get(key, False) is not False:
            raise ValueError(f"V18 does not enable {key}")
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
        raise ValueError("V18 requires all twelve LoRA layers and classifier parameters")
    parameters = [groups[layer] for layer in sorted(groups)] + [{"params": head, "lr": config["head_lr"]}]
    return torch.optim.AdamW(parameters, weight_decay=1e-4), visual, head
