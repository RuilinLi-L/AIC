"""V10 paired training views: V9 light view plus independent quality changes."""

from __future__ import annotations

from io import BytesIO
import random
from typing import Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset

from robust_clip import _clip_image_transform, safe_open_image


def degrade_view_b(image: Image.Image, rng=random) -> Image.Image:
    """Apply the two independent, seeded degradations before light augmentation."""
    if rng.random() < 0.5:
        target = rng.randint(320, 640)
        longest = max(image.size)
        if longest > target:
            width = max(1, round(image.width * target / longest))
            height = max(1, round(image.height * target / longest))
            image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
    if rng.random() < 0.5:
        quality = rng.randint(65, 95)
        encoded = BytesIO()
        image.save(encoded, format="JPEG", quality=quality, subsampling=2)
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            image = decoded.convert("RGB")
    return image


class RobustPairedFolderImageDataset(Dataset):
    """Decode once; preserve V9's view A and alter only view B's input image."""

    def __init__(self, rows: Sequence[tuple[str, int, str]], indices: Sequence[int], processor):
        self.rows = rows
        self.indices = [int(index) for index in indices]
        self.transform_a = _clip_image_transform(processor, "light")
        self.transform_b = _clip_image_transform(processor, "light")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        row_index = self.indices[item]
        path, label, class_name = self.rows[row_index]
        image = safe_open_image(path)
        view_a = self.transform_a(image)
        view_b = self.transform_b(degrade_view_b(image))
        return {
            "pixel_values_a": view_a,
            "pixel_values_b": view_b,
            "label": torch.tensor(label, dtype=torch.long),
            "row_index": torch.tensor(row_index, dtype=torch.long),
            "path": path,
            "class_name": class_name,
        }
