"""V14 keeps expanded's training recipe and changes only view-B image quality."""

from v13_core import (
    build_optimizer,
    dynamic_repair_plan,
    file_sha256,
    model_identity,
    pool_signature,
    repair_diagnostics,
    repair_weight,
    repaired_soft_loss,
    should_audit,
    stage_indices,
)
from v13_core import recipe_config as expanded_recipe_config


RECIPES = ("expanded_robust", "expanded")
ROBUST_AUGMENTATION = "robust_quality_320_view_b"
PLAIN_AUGMENTATION = "light_320_both_views"


def recipe_config(recipe):
    if recipe not in RECIPES:
        raise ValueError(f"unknown V14 recipe: {recipe}")
    return {
        **expanded_recipe_config("expanded"),
        "recipe": recipe,
        "augmentation": ROBUST_AUGMENTATION if recipe == "expanded_robust" else PLAIN_AUGMENTATION,
    }


def validate_recipe_augmentation(config):
    """Legacy expanded metadata may omit augmentation; recorded values must agree."""
    expected = recipe_config(config["recipe"])
    if "augmentation" in config and config["augmentation"] != expected["augmentation"]:
        raise ValueError("V14 recipe conflicts with augmentation")
    return expected
