"""V14 single expanded CLIP model and backward-compatible checkpoint loading."""

import numpy as np
import torch

from robust_clip import load_classifier_state, load_clip
from train_v6 import validate_clip_vit_b32
from v11_resolution import (
    IMAGE_SIZE, ZOOM_SIZE, FourViewTestDataset, HighResolutionPairedDataset,
    build_zoom_transform, resolution_processor,
)
from v13_model import build_classifier as expanded_classifier
from v13_model import build_classifier_from_checkpoint as legacy_classifier
from v13_model import checkpoint_resolution as legacy_resolution
from v14_core import model_identity, validate_recipe_augmentation


RESOLUTION_VERSION = "v14_320_zoom366_bicubic_v1"


def _validate_model_config(config):
    expected = validate_recipe_augmentation(config)
    for key in ("lora_rank", "lora_alpha", "lora_layers", "lora_dropout", "lora_targets"):
        actual = tuple(config[key]) if key == "lora_targets" else config[key]
        if actual != expected[key]:
            raise ValueError(f"V14 checkpoint recipe conflicts with {key}")


def build_classifier(model, prototypes, device, config):
    _validate_model_config(config)
    return expanded_classifier(model, prototypes, device, config)


def checkpoint_resolution(checkpoint):
    if checkpoint.get("format_version") != 14:
        return legacy_resolution(checkpoint)
    config = checkpoint["config"]
    _validate_model_config(config)
    if (config.get("image_size") != IMAGE_SIZE or config.get("zoom_shortest_edge") != ZOOM_SIZE
            or config.get("interpolate_pos_encoding") is not True):
        raise ValueError("V14 requires 320 input and positional interpolation")
    return IMAGE_SIZE


def build_classifier_from_checkpoint(checkpoint, model_dir, device):
    if checkpoint.get("format_version") != 14:
        return legacy_classifier(checkpoint, model_dir, device)
    checkpoint_resolution(checkpoint)
    if model_identity(model_dir) != checkpoint["config"]["base_model_identity"]:
        raise ValueError("official base weights differ from the training checkpoint")
    model, processor = load_clip(model_dir, device)
    validate_clip_vit_b32(model)
    classifier = build_classifier(
        model, np.zeros((len(checkpoint["class_names"]), int(model.config.projection_dim))),
        device, checkpoint["config"],
    )
    load_classifier_state(classifier, checkpoint["model"])
    classifier.eval()
    return classifier, resolution_processor(processor)
