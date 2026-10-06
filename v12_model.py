"""V12 LoRA architecture with the existing 320-pixel preprocessing."""

import numpy as np
import torch

from robust_clip import inject_lora_from_config, load_classifier_state, load_clip
from train_v6 import validate_clip_vit_b32
from v11_resolution import (
    IMAGE_SIZE, ZOOM_SIZE, FourViewTestDataset, HighResolutionPairedDataset,
    ResolutionCLIPClassifier, build_zoom_transform, resolution_processor,
    checkpoint_resolution as legacy_resolution,
    build_classifier_from_checkpoint as legacy_classifier,
)

RESOLUTION_VERSION = "v12_320_zoom366_bicubic_v1"
LORA_CONFIG = {"lora_rank": 16, "lora_alpha": 32.0, "lora_layers": 4,
               "lora_targets": ("q_proj", "k_proj", "v_proj", "out_proj"), "lora_dropout": 0.0}


def validate_lora_config(config):
    for key, expected in LORA_CONFIG.items():
        value = config.get(key)
        if key == "lora_targets" and value is not None:
            value = tuple(value)
        if value != expected:
            raise ValueError(f"V12 requires {key}={expected}")


def build_classifier(model, prototypes: np.ndarray, device):
    classifier = ResolutionCLIPClassifier(
        model, classifier_init=torch.as_tensor(prototypes, dtype=torch.float32, device=device),
        bottleneck=128, initial_logit_scale=30.0, max_logit_scale=50.0,
        learn_logit_scale=False, image_size=IMAGE_SIZE,
    ).to(device)
    modules = inject_lora_from_config(classifier.clip, LORA_CONFIG)
    if len(modules) != 16:
        raise ValueError(f"expected 16 V12 LoRA modules, got {len(modules)}")
    return classifier


def checkpoint_resolution(checkpoint):
    if checkpoint.get("format_version") in (9, 11):
        return legacy_resolution(checkpoint)
    config = checkpoint["config"]
    if checkpoint.get("format_version") != 12 or (
        config.get("image_size") != IMAGE_SIZE or config.get("zoom_shortest_edge") != ZOOM_SIZE
        or config.get("interpolate_pos_encoding") is not True
    ):
        raise ValueError("V12 checkpoint requires 320 inputs and positional interpolation")
    validate_lora_config(config)
    return IMAGE_SIZE


def build_classifier_from_checkpoint(checkpoint, model_dir, device):
    checkpoint_resolution(checkpoint)
    if checkpoint["format_version"] in (9, 11):
        return legacy_classifier(checkpoint, model_dir, device)
    model, processor = load_clip(model_dir, device)
    validate_clip_vit_b32(model)
    classifier = build_classifier(
        model, np.zeros((len(checkpoint["class_names"]), int(model.config.projection_dim)), dtype=np.float32), device,
    )
    load_classifier_state(classifier, checkpoint["model"])
    classifier.eval()
    return classifier, resolution_processor(processor)
