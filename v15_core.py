"""V15 adds MLP LoRA while preserving V14's robust expanded recipe."""

from v14_core import (
    build_optimizer, dynamic_repair_plan, file_sha256, model_identity,
    pool_signature, repair_diagnostics, repair_weight, repaired_soft_loss,
    should_audit, stage_indices,
)
from v14_core import recipe_config as v14_recipe_config


RECIPES = ("expanded_mlp",)
ROBUST_AUGMENTATION = "robust_quality_320_view_b"
MLP_CONFIG_KEYS = (
    "mlp_lora_targets", "mlp_lora_layers", "mlp_lora_rank",
    "mlp_lora_alpha", "mlp_lora_dropout",
)


def recipe_config(recipe):
    if recipe not in RECIPES:
        raise ValueError(f"unknown V15 recipe: {recipe}")
    return {
        **v14_recipe_config("expanded_robust"),
        "recipe": recipe,
        "mlp_lora_targets": ("fc1", "fc2"),
        "mlp_lora_layers": 12,
        "mlp_lora_rank": 16,
        "mlp_lora_alpha": 32.0,
        "mlp_lora_dropout": 0.0,
    }


def validate_recipe_augmentation(config):
    expected = recipe_config(config["recipe"])
    if config.get("augmentation") != expected["augmentation"]:
        raise ValueError("V15 recipe conflicts with augmentation")
    return expected


def validate_recipe_config(config):
    """Every scientific recipe setting is frozen for this one-candidate run."""
    expected = validate_recipe_augmentation(config)
    for key, value in expected.items():
        actual = config.get(key)
        if key in ("lora_targets", "mlp_lora_targets") and actual is not None:
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"V15 checkpoint recipe conflicts with {key}")
    return expected
