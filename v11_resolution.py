"""320-pixel CLIP views and positional interpolation, isolated from V9/V10."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from robust_clip import (
    PairedFolderImageDataset,
    RobustCLIPClassifier,
    _feature_tensor,
    inject_lora_from_config,
    load_classifier_state,
    load_clip,
)
from train_v6 import validate_clip_vit_b32
from v7_views import FourViewTestDataset as OriginalFourViewTestDataset


IMAGE_SIZE = 320
ZOOM_SIZE = 366
RESOLUTION_VERSION = "v11_320_zoom366_bicubic_v1"


def resolution_processor(processor, image_size: int = IMAGE_SIZE):
    """Copy preprocessing settings; keep the official 224 processor intact."""
    if image_size not in (224, IMAGE_SIZE):
        raise ValueError("supported image sizes are 224 (V9) and 320 (V11)")
    result = deepcopy(processor)
    result.image_processor.size = {"shortest_edge": image_size}
    result.image_processor.crop_size = {"height": image_size, "width": image_size}
    return result


def build_light_transform(processor):
    settings = processor.image_processor
    size = int(settings.crop_size["height"])
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size, scale=(0.90, 1.0), ratio=(0.95, 1.05),
            interpolation=transforms.InterpolationMode.BICUBIC, antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.image_mean, std=settings.image_std),
    ])


def build_zoom_transform(processor):
    settings = processor.image_processor
    size = int(settings.crop_size["height"])
    if size not in (224, IMAGE_SIZE):
        raise ValueError(f"unsupported crop size: {size}")
    return transforms.Compose([
        transforms.Resize(
            ZOOM_SIZE if size == IMAGE_SIZE else 256,
            interpolation=transforms.InterpolationMode.BICUBIC, antialias=True,
        ),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.image_mean, std=settings.image_std),
    ])


class HighResolutionPairedDataset(PairedFolderImageDataset):
    def __init__(self, rows, indices, processor):
        super().__init__(rows, indices, processor, augment="light")
        self.transform_a = build_light_transform(processor)
        self.transform_b = build_light_transform(processor)


class FourViewTestDataset(OriginalFourViewTestDataset):
    def __init__(self, paths, processor):
        super().__init__(paths, processor)
        # Retain legacy view keys (zoom256) for the existing calibration code.
        # The actual resize is 366 for V11, recorded in checkpoint metadata.
        self.zoom_transform = build_zoom_transform(processor)


class ResolutionCLIPClassifier(RobustCLIPClassifier):
    def __init__(self, *args, image_size: int = IMAGE_SIZE, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = int(image_size)

    def _image_base(self, pixels: torch.Tensor) -> torch.Tensor:
        if tuple(pixels.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(f"expected {self.image_size}x{self.image_size} pixels, got {pixels.shape}")
        kwargs = {"pixel_values": pixels, "interpolate_pos_encoding": self.image_size != 224}
        if self.has_trainable_backbone():
            features = _feature_tensor(self.clip.get_image_features(**kwargs))
        else:
            with torch.no_grad():
                features = _feature_tensor(self.clip.get_image_features(**kwargs))
        return F.normalize(features.float(), dim=-1)

    def encode_image(self, pixel_values):
        base = self._image_base(pixel_values)
        return base, self.adapter(base)

    def forward(self, pixel_values, text_features=None):
        return self.forward_features(self._image_base(pixel_values), text_features)


def build_classifier(model, prototypes: np.ndarray, device: torch.device):
    classifier = ResolutionCLIPClassifier(
        model, classifier_init=torch.as_tensor(prototypes, dtype=torch.float32, device=device),
        bottleneck=128, initial_logit_scale=30.0, max_logit_scale=50.0,
        learn_logit_scale=False, image_size=IMAGE_SIZE,
    ).to(device)
    modules = inject_lora_from_config(classifier.clip, {
        "lora_rank": 8, "lora_alpha": 16.0, "lora_layers": 4,
        "lora_targets": ("q_proj", "v_proj"), "lora_dropout": 0.0,
    })
    if len(modules) != 8:
        raise ValueError(f"expected 8 Q/V LoRA modules, got {len(modules)}")
    return classifier


def checkpoint_resolution(checkpoint):
    version = checkpoint.get("format_version")
    config = checkpoint["config"]
    if version == 9:
        if int(config.get("image_size", 224)) != 224:
            raise ValueError("V9 reference must use 224 inputs")
        return 224
    if version != 11 or (
        config.get("image_size") != IMAGE_SIZE
        or config.get("zoom_shortest_edge") != ZOOM_SIZE
        or config.get("interpolate_pos_encoding") is not True
    ):
        raise ValueError("V11 checkpoint must specify 320 inputs, zoom 366 and positional interpolation")
    return IMAGE_SIZE


def build_classifier_from_checkpoint(checkpoint, model_dir: str, device: torch.device):
    size = checkpoint_resolution(checkpoint)
    model, processor = load_clip(model_dir, device)
    validate_clip_vit_b32(model)
    config = checkpoint["config"]
    classifier = ResolutionCLIPClassifier(
        model,
        classifier_init=torch.zeros(len(checkpoint["class_names"]), int(model.config.projection_dim)),
        bottleneck=int(config.get("bottleneck", 128)),
        initial_logit_scale=float(config.get("initial_logit_scale", 30.0)),
        max_logit_scale=float(config.get("max_logit_scale", 50.0)),
        learn_logit_scale=bool(config.get("learn_logit_scale", False)),
        image_size=size,
    ).to(device)
    modules = inject_lora_from_config(classifier.clip, config)
    if len(modules) != 8:
        raise ValueError(f"expected 8 Q/V LoRA modules, got {len(modules)}")
    load_classifier_state(classifier, checkpoint["model"])
    classifier.eval()
    return classifier, resolution_processor(processor, size)
