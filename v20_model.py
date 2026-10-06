"""V20's 72 DoRA/LoRA+ modules and complete trainable-state loading contract."""

import math
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from robust_clip import LoRALinear, load_clip
from robust_clip import load_classifier_state as legacy_load_classifier_state
from train_v6 import validate_clip_vit_b32
from v11_resolution import ResolutionCLIPClassifier as BaseResolutionCLIPClassifier
from v20_views import (
    IMAGE_SIZE, ZOOM_SIZE, FourViewTestDataset, HighResolutionPairedDataset,
    build_zoom_transform, resolution_processor,
)
from v19_model import build_classifier_from_checkpoint as legacy_classifier
from v19_model import checkpoint_resolution as legacy_resolution
from v20_core import model_identity, recipe_config, validate_recipe_config


RESOLUTION_VERSION = "v20_320_zoom366_bicubic_v1"


class DoRALinear(nn.Module):
    """Frozen linear layer with low-rank direction and per-output magnitude."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("DoRA requires an untouched nn.Linear")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("DoRA rank must be a positive integer")
        if dropout != 0.0:
            raise ValueError("V20 DoRA fixes adapter dropout to zero")
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(
            base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lora_m = nn.Parameter(base.weight.detach().float().norm(p=2, dim=1))
        self.scaling = float(alpha) / rank
        self.dropout = 0.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # The dynamic norm is FP32 and detached as in the DoRA paper, section 4.3.
        with torch.no_grad(), torch.autocast(device_type=self.base.weight.device.type, enabled=False):
            direction = self.base.weight.float() + self.scaling * (self.lora_b.float() @ self.lora_a.float())
            weight_norm = direction.norm(p=2, dim=1).clamp_min(1e-6).detach()
        base_output = self.base(inputs)
        scale = (self.lora_m.float() / weight_norm).to(base_output.dtype)
        residual = F.linear(F.linear(inputs, self.lora_a), self.lora_b) * self.scaling
        without_bias = base_output if self.base.bias is None else base_output - self.base.bias.to(base_output.dtype)
        # This form preserves the exact initial base output when scale=1 and B=0.
        return base_output + (scale - 1) * without_bias + scale * residual


class ResolutionCLIPClassifier(BaseResolutionCLIPClassifier):
    def train(self, mode=True):
        super().train(mode)
        for module in self.clip.modules():
            if isinstance(module, DoRALinear):
                module.train(mode)
        return self


def trainable_parameter_names(classifier):
    return sorted(name for name, parameter in classifier.named_parameters() if parameter.requires_grad)


def validate_trainable_metadata(classifier, checkpoint):
    expected = trainable_parameter_names(classifier)
    recorded = checkpoint.get("trainable_parameter_names")
    if recorded != expected:
        raise ValueError("V20 checkpoint trainable_parameter_names are missing or inconsistent")
    return expected


def validate_trainable_state(classifier, state, *, exact=False):
    if not isinstance(state, Mapping):
        raise RuntimeError("V20 trainable state must be a mapping")
    expected = set(trainable_parameter_names(classifier))
    model_state = classifier.state_dict()
    allowed = expected if exact else expected | {name for name in model_state if not name.startswith("clip.")}
    missing = sorted(allowed - set(state))
    unexpected = sorted(set(state) - allowed)
    if missing or unexpected:
        raise RuntimeError(f"V20 trainable state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    for name in allowed:
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != model_state[name].shape
                or not value.is_floating_point() or not torch.isfinite(value).all().item()):
            raise RuntimeError(f"V20 trainable state has invalid tensor shape: {name}")


def _official_trainable_shapes(class_count, config=None):
    config = recipe_config("dora_rank32") if config is None else config
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
    if config["method"] == "dora":
        for name, shape in list(shapes.items()):
            if name.endswith(".lora_b"):
                shapes[name[:-1] + "m"] = (shape[0],)
    return shapes


def validate_checkpoint_state_metadata(checkpoint):
    """Validate a complete official V20 state without constructing the base model."""
    if checkpoint.get("format_version") != 20:
        version = checkpoint.get("format_version")
        if version in (15, 16, 17, 18, 19):
            from importlib import import_module
            return import_module(f"v{version}_model").validate_checkpoint_state_metadata(checkpoint)
        raise ValueError("V20 state metadata requires checkpoint format 20")
    checkpoint_resolution(checkpoint)
    classes = checkpoint.get("class_names")
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError("V20 checkpoint requires class names")
    expected = _official_trainable_shapes(len(classes), checkpoint["config"])
    if checkpoint.get("trainable_parameter_names") != sorted(expected):
        raise ValueError("V20 checkpoint trainable_parameter_names are missing or inconsistent")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or set(state) != set(expected) | {"logit_scale"}:
        raise ValueError("V20 checkpoint model keys differ from the complete trainable state")
    for name, shape in {**expected, "logit_scale": ()}.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"V20 official checkpoint tensor has invalid shape or dtype: {name}")
    for key in ("ema", "best_state"):
        if key not in checkpoint:
            continue
        shadow = checkpoint[key]
        if not isinstance(shadow, dict) or set(shadow) != set(expected):
            raise ValueError(f"V20 checkpoint {key} keys differ from the complete trainable state")
        for name, shape in expected.items():
            value = shadow[name]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or not value.is_floating_point() or not torch.isfinite(value).all().item()):
                raise ValueError(f"V20 checkpoint {key} tensor has invalid shape or dtype: {name}")
    return sorted(expected)


def load_classifier_state(classifier, state):
    """Check every trainable key before legacy loading can ignore CLIP keys."""
    validate_trainable_state(classifier, state)
    legacy_load_classifier_state(classifier, state)


def build_classifier(model, prototypes, device, config):
    validate_recipe_config(config)
    if len(model.vision_model.encoder.layers) != 12:
        raise ValueError("V20 requires twelve visual transformer layers")
    classifier = ResolutionCLIPClassifier(
        model, classifier_init=torch.as_tensor(prototypes, dtype=torch.float32, device=device),
        bottleneck=128, initial_logit_scale=30.0, max_logit_scale=50.0,
        learn_logit_scale=False, image_size=config["image_size"],
    ).to(device)
    adapter_class = DoRALinear if config["method"] == "dora" else LoRALinear
    layers = classifier.clip.vision_model.encoder.layers
    inserted = []
    # Keep V19's RNG order: all attention projections precede all MLP projections.
    for section, targets, prefix in (
        ("self_attn", config["lora_targets"], "lora"),
        ("mlp", config["mlp_lora_targets"], "mlp_lora"),
    ):
        for layer_id, layer in enumerate(layers):
            container = getattr(layer, section)
            for target in targets:
                base = getattr(container, target)
                if not isinstance(base, nn.Linear):
                    raise TypeError(f"Expected untouched Linear at layer {layer_id}.{section}.{target}")
                setattr(container, target, adapter_class(
                    base, rank=config[f"{prefix}_rank"], alpha=config[f"{prefix}_alpha"],
                    dropout=config[f"{prefix}_dropout"],
                ))
                inserted.append((layer_id, section, target))
    modules = [module for module in classifier.clip.modules() if isinstance(module, (DoRALinear, LoRALinear))]
    if len(inserted) != 72 or len(modules) != 72:
        raise ValueError("V20 requires 48 attention and 24 MLP adapter modules")
    suffixes = ("lora_a", "lora_b", "lora_m") if config["method"] == "dora" else ("lora_a", "lora_b")
    if any(not name.endswith(suffixes)
           for name, parameter in classifier.clip.named_parameters() if parameter.requires_grad):
        raise ValueError("V20 only trains adapter parameters in the official backbone")
    return classifier


def checkpoint_resolution(checkpoint):
    if checkpoint.get("format_version") != 20:
        return legacy_resolution(checkpoint)
    config = checkpoint["config"]
    validate_recipe_config(config)
    return int(config["image_size"])


def build_classifier_from_checkpoint(checkpoint, model_dir, device):
    if checkpoint.get("format_version") != 20:
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
