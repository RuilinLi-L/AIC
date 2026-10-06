"""V19's 72 LoRA modules and complete trainable-state loading contract."""

import numpy as np
import torch
from torch import nn

from robust_clip import LoRALinear, load_clip, inject_lora_from_config
from robust_clip import load_classifier_state as legacy_load_classifier_state
from train_v6 import validate_clip_vit_b32
from v11_resolution import ResolutionCLIPClassifier
from v19_views import (
    IMAGE_SIZE, ZOOM_SIZE, FourViewTestDataset, HighResolutionPairedDataset,
    build_zoom_transform, resolution_processor,
)
from v18_model import build_classifier_from_checkpoint as legacy_classifier
from v18_model import checkpoint_resolution as legacy_resolution
from v19_core import model_identity, recipe_config, validate_recipe_config


RESOLUTION_VERSION = "v19_recipe_resolution_bicubic_v1"


def trainable_parameter_names(classifier):
    return sorted(name for name, parameter in classifier.named_parameters() if parameter.requires_grad)


def validate_trainable_metadata(classifier, checkpoint):
    expected = trainable_parameter_names(classifier)
    recorded = checkpoint.get("trainable_parameter_names")
    if recorded != expected:
        raise ValueError("V19 checkpoint trainable_parameter_names are missing or inconsistent")
    return expected


def validate_trainable_state(classifier, state, *, exact=False):
    expected = set(trainable_parameter_names(classifier))
    model_state = classifier.state_dict()
    allowed = expected if exact else expected | {name for name in model_state if not name.startswith("clip.")}
    missing = sorted(allowed - set(state))
    unexpected = sorted(set(state) - allowed)
    if missing or unexpected:
        raise RuntimeError(f"V19 trainable state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    for name in allowed:
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != model_state[name].shape
                or not value.is_floating_point()):
            raise RuntimeError(f"V19 trainable state has invalid tensor shape: {name}")


def _official_trainable_shapes(class_count, config=None):
    config = recipe_config("resolution384_rank32") if config is None else config
    validate_recipe_config(config)
    rank, mlp_rank = config["lora_rank"], config["mlp_lora_rank"]
    shapes = {
        "classifier": (class_count, 512), "adapter.scale": (),
        "adapter.net.0.weight": (128, 512), "adapter.net.0.bias": (128,),
        "adapter.net.3.weight": (512, 128), "adapter.net.3.bias": (512,),
    }
    for layer in range(12):
        prefix = f"clip.vision_model.encoder.layers.{layer}"
        for target in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[f"{prefix}.self_attn.{target}.lora_a"] = (rank, 768)
            shapes[f"{prefix}.self_attn.{target}.lora_b"] = (768, rank)
        shapes[f"{prefix}.mlp.fc1.lora_a"] = (mlp_rank, 768)
        shapes[f"{prefix}.mlp.fc1.lora_b"] = (3072, mlp_rank)
        shapes[f"{prefix}.mlp.fc2.lora_a"] = (mlp_rank, 3072)
        shapes[f"{prefix}.mlp.fc2.lora_b"] = (768, mlp_rank)
    return shapes


def validate_checkpoint_state_metadata(checkpoint):
    """Validate a complete official V19 state without constructing the base model."""
    if checkpoint.get("format_version") != 19:
        version = checkpoint.get("format_version")
        if version in (15, 16, 17, 18):
            from importlib import import_module
            return import_module(f"v{version}_model").validate_checkpoint_state_metadata(checkpoint)
        raise ValueError("V19 state metadata requires checkpoint format 19")
    checkpoint_resolution(checkpoint)
    classes = checkpoint.get("class_names")
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError("V19 checkpoint requires class names")
    expected = _official_trainable_shapes(len(classes), checkpoint["config"])
    if checkpoint.get("trainable_parameter_names") != sorted(expected):
        raise ValueError("V19 checkpoint trainable_parameter_names are missing or inconsistent")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or set(state) != set(expected) | {"logit_scale"}:
        raise ValueError("V19 checkpoint model keys differ from the complete trainable state")
    for name, shape in {**expected, "logit_scale": ()}.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
            raise ValueError(f"V19 official checkpoint tensor has invalid shape or dtype: {name}")
    return sorted(expected)


def load_classifier_state(classifier, state):
    """Check every trainable key before legacy loading can ignore CLIP keys."""
    validate_trainable_state(classifier, state)
    legacy_load_classifier_state(classifier, state)


def build_classifier(model, prototypes, device, config):
    validate_recipe_config(config)
    if len(model.vision_model.encoder.layers) != 12:
        raise ValueError("V19 requires twelve visual transformer layers")
    classifier = ResolutionCLIPClassifier(
        model, classifier_init=torch.as_tensor(prototypes, dtype=torch.float32, device=device),
        bottleneck=128, initial_logit_scale=30.0, max_logit_scale=50.0,
        learn_logit_scale=False, image_size=config["image_size"],
    ).to(device)
    attention_modules = inject_lora_from_config(classifier.clip, config)
    if len(attention_modules) != 48:
        raise ValueError("V19 requires 48 freshly initialized attention LoRA modules")
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
        raise ValueError("V19 requires 48 attention and 24 MLP LoRA modules")
    if any(not name.endswith(("lora_a", "lora_b"))
           for name, parameter in classifier.clip.named_parameters() if parameter.requires_grad):
        raise ValueError("V19 only trains LoRA parameters in the official backbone")
    return classifier


def checkpoint_resolution(checkpoint):
    if checkpoint.get("format_version") != 19:
        return legacy_resolution(checkpoint)
    config = checkpoint["config"]
    validate_recipe_config(config)
    return int(config["image_size"])


def build_classifier_from_checkpoint(checkpoint, model_dir, device):
    if checkpoint.get("format_version") != 19:
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
    return classifier, resolution_processor(processor, checkpoint["config"]["image_size"])
