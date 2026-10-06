"""Single official CLIP backbone with recipe-specific V13 LoRA modules."""
import numpy as np
import torch

from robust_clip import inject_lora_from_config, load_classifier_state, load_clip
from train_v6 import validate_clip_vit_b32
from v11_resolution import (IMAGE_SIZE, ZOOM_SIZE, FourViewTestDataset, HighResolutionPairedDataset,
                            ResolutionCLIPClassifier, build_zoom_transform, resolution_processor)
from v12_model import build_classifier_from_checkpoint as legacy_classifier
from v12_model import checkpoint_resolution as legacy_resolution
from v13_core import recipe_config, model_identity

RESOLUTION_VERSION = "v13_320_zoom366_bicubic_v1"


def build_classifier(model, prototypes, device, config):
    classifier = ResolutionCLIPClassifier(
        model, classifier_init=torch.as_tensor(prototypes, dtype=torch.float32, device=device),
        bottleneck=128, initial_logit_scale=30.0, max_logit_scale=50.0,
        learn_logit_scale=False, image_size=IMAGE_SIZE,
    ).to(device)
    modules = inject_lora_from_config(classifier.clip, config)
    if len(modules) != 4 * config["lora_layers"]:
        raise ValueError("unexpected number of V13 LoRA modules")
    return classifier


def checkpoint_resolution(checkpoint):
    if checkpoint.get("format_version") != 13:
        return legacy_resolution(checkpoint)
    config = checkpoint["config"]
    expected = recipe_config(config["recipe"])
    for key in ("lora_rank", "lora_alpha", "lora_layers", "lora_dropout", "lora_targets"):
        actual = tuple(config[key]) if key == "lora_targets" else config[key]
        if actual != expected[key]:
            raise ValueError(f"checkpoint recipe conflicts with {key}")
    if (config.get("image_size") != IMAGE_SIZE or config.get("zoom_shortest_edge") != ZOOM_SIZE
            or config.get("interpolate_pos_encoding") is not True):
        raise ValueError("V13 requires 320 input and positional interpolation")
    return IMAGE_SIZE


def build_classifier_from_checkpoint(checkpoint, model_dir, device):
    if checkpoint.get("format_version") != 13:
        return legacy_classifier(checkpoint, model_dir, device)
    checkpoint_resolution(checkpoint)
    if model_identity(model_dir) != checkpoint["config"]["base_model_identity"]:
        raise ValueError("official base weights differ from the training checkpoint")
    model, processor = load_clip(model_dir, device)
    validate_clip_vit_b32(model)
    classifier = build_classifier(model, np.zeros((len(checkpoint["class_names"]),
                                                   int(model.config.projection_dim))), device, checkpoint["config"])
    load_classifier_state(classifier, checkpoint["model"])
    classifier.eval()
    return classifier, resolution_processor(processor)
