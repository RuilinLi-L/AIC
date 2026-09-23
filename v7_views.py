"""Deterministic center/flip/zoom views shared by V7 calibration and prediction."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from robust_clip import _clip_image_transform, safe_open_image


VIEW_NAMES = ("center", "hflip", "zoom256", "zoom256_hflip")


def build_zoom_transform(processor):
    image_processor = processor.image_processor
    mean = image_processor.image_mean
    std = image_processor.image_std
    crop_height = int(image_processor.crop_size.get("height", 224))
    crop_width = int(image_processor.crop_size.get("width", 224))
    return transforms.Compose(
        [
            transforms.Resize(
                256,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop((crop_height, crop_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class FourViewFolderDataset(Dataset):
    """Decode once and return the two base tensors used for four fixed views."""

    def __init__(self, rows, indices: Sequence[int], processor):
        self.rows = rows
        self.indices = [int(value) for value in indices]
        self.center_transform = _clip_image_transform(processor, "none")
        self.zoom_transform = build_zoom_transform(processor)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        row_index = self.indices[item]
        path, label, class_name = self.rows[row_index]
        image = safe_open_image(path)
        return {
            "center": self.center_transform(image),
            "zoom256": self.zoom_transform(image),
            "label": torch.tensor(label, dtype=torch.long),
            "row_index": torch.tensor(row_index, dtype=torch.long),
            "path": path,
            "class_name": class_name,
        }


class FourViewTestDataset(Dataset):
    def __init__(self, paths: Sequence[str], processor):
        self.paths = list(paths)
        self.center_transform = _clip_image_transform(processor, "none")
        self.zoom_transform = build_zoom_transform(processor)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, item: int):
        path = self.paths[item]
        image = safe_open_image(path)
        return {
            "center": self.center_transform(image),
            "zoom256": self.zoom_transform(image),
            "path": path,
        }


def normalize_view_weights(weights: dict[str, float]) -> dict[str, float]:
    unknown = set(weights) - set(VIEW_NAMES)
    if unknown:
        raise ValueError(f"unknown V7 TTA views: {sorted(unknown)}")
    cleaned = {name: max(0.0, float(weights.get(name, 0.0))) for name in VIEW_NAMES}
    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("V7 TTA view weights must contain positive mass")
    normalized = {name: value / total for name, value in cleaned.items() if value > 0.0}
    # Absorb floating-point drift so inference always sees an exact simplex.
    last = next(reversed(normalized))
    normalized[last] += 1.0 - sum(normalized.values())
    return normalized
