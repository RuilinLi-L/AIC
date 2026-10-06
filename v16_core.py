"""Two controlled V16 recipes: light views, optionally last-four visual norms."""

import re

import torch

from v15_core import (
    dynamic_repair_plan, file_sha256, model_identity, pool_signature,
    repair_diagnostics, repair_weight, repaired_soft_loss, should_audit, stage_indices,
)
from v15_core import recipe_config as v15_recipe_config


RECIPES = ("mlp_light", "mlp_light_ln")
PLAIN_AUGMENTATION = "light_320_both_views"
LAYER_NORM_CONFIG_KEYS = ("layernorm_layers", "layernorm_lr", "layernorm_weight_decay")
MLP_CONFIG_KEYS = ("mlp_lora_targets", "mlp_lora_layers", "mlp_lora_rank",
                   "mlp_lora_alpha", "mlp_lora_dropout")


def recipe_config(recipe):
    if recipe not in RECIPES:
        raise ValueError(f"unknown V16 recipe: {recipe}")
    return {
        **v15_recipe_config("expanded_mlp"),
        "recipe": recipe,
        "augmentation": PLAIN_AUGMENTATION,
        "layernorm_layers": 4 if recipe == "mlp_light_ln" else 0,
        "layernorm_lr": 1e-5,
        "layernorm_weight_decay": 0.0,
    }


def validate_recipe_augmentation(config):
    expected = recipe_config(config["recipe"])
    if config.get("augmentation") != expected["augmentation"]:
        raise ValueError("V16 recipe conflicts with augmentation")
    return expected


def validate_recipe_config(config):
    expected = validate_recipe_augmentation(config)
    for key, value in expected.items():
        actual = config.get(key)
        if key in ("lora_targets", "mlp_lora_targets") and actual is not None:
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"V16 checkpoint recipe conflicts with {key}")
    return expected


def layernorm_parameter_names(config):
    """Exact official-model names; never include pre/post norms or text norms."""
    validate_recipe_config(config)
    return {
        f"clip.vision_model.encoder.layers.{layer}.layer_norm{norm}.{parameter}"
        for layer in range(12 - config["layernorm_layers"], 12)
        for norm in (1, 2) for parameter in ("weight", "bias")
    }


def build_optimizer(classifier, config):
    validate_recipe_config(config)
    allowed_norms = layernorm_parameter_names(config)
    groups, visual, head, norms = {}, [], [], []
    found_norms = set()
    for name, parameter in classifier.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in allowed_norms:
            norms.append(parameter)
            visual.append(parameter)
            found_norms.add(name)
        elif name.startswith("clip."):
            match = re.fullmatch(
                r"clip\.vision_model\.encoder\.layers\.(\d+)\.(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.(?:fc1|fc2))\.lora_[ab]",
                name,
            )
            if match is None or int(match.group(1)) not in range(12):
                raise ValueError(f"unexpected unfrozen backbone parameter: {name}")
            layer = int(match.group(1))
            lr = config["lora_lr"] * config["layer_lr_decay"] ** (11 - layer)
            groups.setdefault(layer, {"params": [], "lr": lr, "layer": layer})["params"].append(parameter)
            visual.append(parameter)
        else:
            head.append(parameter)
    if found_norms != allowed_norms:
        raise ValueError("V16 trainable LayerNorm parameters differ from recipe")
    if len(groups) != 12 or not head:
        raise ValueError("V16 requires all twelve LoRA layers and classifier parameters")
    parameter_groups = [groups[layer] for layer in sorted(groups)]
    if norms:
        parameter_groups.append({"params": norms, "lr": config["layernorm_lr"],
                                 "weight_decay": config["layernorm_weight_decay"],
                                 "kind": "layernorm"})
    parameter_groups.append({"params": head, "lr": config["head_lr"]})
    parameters = [parameter for group in parameter_groups for parameter in group["params"]]
    expected = [parameter for parameter in classifier.parameters() if parameter.requires_grad]
    if len({id(parameter) for parameter in parameters}) != len(parameters) or {
            id(parameter) for parameter in parameters} != {id(parameter) for parameter in expected}:
        raise ValueError("V16 optimizer must include every trainable parameter exactly once")
    return torch.optim.AdamW(parameter_groups, weight_decay=1e-4), visual, head
