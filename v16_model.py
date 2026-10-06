"""V16's 72 LoRA modules and optional last-four-layer affine norm adaptation."""

import numpy as np
import torch
from torch import nn

from robust_clip import LoRALinear, load_clip
from robust_clip import load_classifier_state as legacy_load_classifier_state
from train_v6 import validate_clip_vit_b32
from v11_resolution import (
    IMAGE_SIZE, ZOOM_SIZE, FourViewTestDataset, HighResolutionPairedDataset,
    build_zoom_transform, resolution_processor,
)
from v13_model import build_classifier as attention_classifier
from v15_model import build_classifier_from_checkpoint as legacy_classifier
from v15_model import checkpoint_resolution as legacy_resolution
from v16_core import model_identity, validate_recipe_config, layernorm_parameter_names, recipe_config


RESOLUTION_VERSION = "v16_320_zoom366_bicubic_v1"


def trainable_parameter_names(classifier):
    return sorted(name for name, parameter in classifier.named_parameters() if parameter.requires_grad)


def validate_trainable_metadata(classifier, checkpoint):
    expected = trainable_parameter_names(classifier)
    recorded = checkpoint.get("trainable_parameter_names")
    if recorded != expected:
        raise ValueError("V16 checkpoint trainable_parameter_names are missing or inconsistent")
    return expected


def validate_trainable_state(classifier, state, *, exact=False):
    expected = set(trainable_parameter_names(classifier))
    model_state = classifier.state_dict()
    allowed = expected if exact else expected | {name for name in model_state if not name.startswith("clip.")}
    missing = sorted(allowed - set(state))
    unexpected = sorted(set(state) - allowed)
    if missing or unexpected:
        raise RuntimeError(f"V16 trainable state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    for name in allowed:
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != model_state[name].shape
                or not value.is_floating_point()):
            raise RuntimeError(f"V16 trainable state has invalid tensor shape: {name}")


def _official_trainable_shapes(class_count, config=None):
    config = recipe_config("mlp_light") if config is None else config
    validate_recipe_config(config)
    shapes = {
        "classifier": (class_count, 512), "adapter.scale": (),
        "adapter.net.0.weight": (128, 512), "adapter.net.0.bias": (128,),
        "adapter.net.3.weight": (512, 128), "adapter.net.3.bias": (512,),
    }
    for layer in range(12):
        prefix = f"clip.vision_model.encoder.layers.{layer}"
        for target in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[f"{prefix}.self_attn.{target}.lora_a"] = (16, 768)
            shapes[f"{prefix}.self_attn.{target}.lora_b"] = (768, 16)
        shapes[f"{prefix}.mlp.fc1.lora_a"] = (16, 768)
        shapes[f"{prefix}.mlp.fc1.lora_b"] = (3072, 16)
        shapes[f"{prefix}.mlp.fc2.lora_a"] = (16, 3072)
        shapes[f"{prefix}.mlp.fc2.lora_b"] = (768, 16)
    shapes.update({name: (768,) for name in layernorm_parameter_names(config)})
    return shapes


def validate_checkpoint_state_metadata(checkpoint):
    """Validate a complete official V16 state without constructing the base model."""
    if checkpoint.get("format_version") != 16:
        raise ValueError("V16 state metadata requires checkpoint format 16")
    checkpoint_resolution(checkpoint)
    classes = checkpoint.get("class_names")
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError("V16 checkpoint requires class names")
    expected = _official_trainable_shapes(len(classes), checkpoint["config"])
    if checkpoint.get("trainable_parameter_names") != sorted(expected):
        raise ValueError("V16 checkpoint trainable_parameter_names are missing or inconsistent")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or set(state) != set(expected) | {"logit_scale"}:
        raise ValueError("V16 checkpoint model keys differ from the complete trainable state")
    for name, shape in {**expected, "logit_scale": ()}.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
            raise ValueError(f"V16 official checkpoint tensor has invalid shape or dtype: {name}")
    return sorted(expected)


def load_classifier_state(classifier, state):
    """Check every trainable key before legacy loading can ignore CLIP keys."""
    validate_trainable_state(classifier, state)
    legacy_load_classifier_state(classifier, state)


def build_classifier(model, prototypes, device, config):
    validate_recipe_config(config)
    classifier = attention_classifier(model, prototypes, device, config)
    layers = classifier.clip.vision_model.encoder.layers
    start = len(layers) - int(config["mlp_lora_layers"])
    inserted = []
    for layer_id in range(start, len(layers)):
        for target in config["mlp_lora_targets"]:
            base = getattr(layers[layer_id].mlp, target)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected untouched MLP Linear at layer {layer_id}.{target}")
            setattr(layers[layer_id].mlp, target, LoRALinear(
                base, rank=config["mlp_lora_rank"], alpha=config["mlp_lora_alpha"],
                dropout=config["mlp_lora_dropout"],
            ))
            inserted.append((layer_id, target))
    modules = [module for module in classifier.clip.modules() if isinstance(module, LoRALinear)]
    if len(inserted) != 24 or len(modules) != 72:
        raise ValueError("V16 requires 48 attention and 24 MLP LoRA modules")
    allowed_norms = layernorm_parameter_names(config)
    parameters = dict(classifier.named_parameters())
    for name in allowed_norms:
        if name not in parameters:
            raise ValueError(f"V16 required LayerNorm parameter is missing: {name}")
        parameters[name].requires_grad_(True)
    unexpected = [name for name, parameter in classifier.named_parameters()
                  if parameter.requires_grad and name.startswith("clip.")
                  and not name.endswith(("lora_a", "lora_b")) and name not in allowed_norms]
    if unexpected:
        raise ValueError(f"V16 unexpected trainable backbone parameters: {unexpected}")
    return classifier


def checkpoint_resolution(checkpoint):
    if checkpoint.get("format_version") != 16:
        return legacy_resolution(checkpoint)
    config = checkpoint["config"]
    validate_recipe_config(config)
    if (config.get("image_size") != IMAGE_SIZE or config.get("zoom_shortest_edge") != ZOOM_SIZE
            or config.get("interpolate_pos_encoding") is not True):
        raise ValueError("V16 requires 320 input and positional interpolation")
    return IMAGE_SIZE


def build_classifier_from_checkpoint(checkpoint, model_dir, device):
    if checkpoint.get("format_version") != 16:
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
    validate_trainable_metadata(classifier, checkpoint)
    load_classifier_state(classifier, checkpoint["model"])
    classifier.eval()
    return classifier, resolution_processor(processor)
