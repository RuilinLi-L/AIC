"""Recipe-sized V19 views with V15's unchanged quality degradation on view B."""

from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from robust_clip import safe_open_image
from v10_views import degrade_view_b
from v11_resolution import build_light_transform
from v7_views import FourViewTestDataset as OriginalFourViewTestDataset
from v7_views import FourViewFolderDataset as OriginalFourViewFolderDataset
from v19_core import validate_recipe_config


IMAGE_SIZE = 384
ZOOM_SIZE = 439
ZOOM_BY_IMAGE_SIZE = {224: 256, 320: 366, 384: 439}


def resolution_processor(processor, image_size=IMAGE_SIZE):
    """Copy official preprocessing metadata, without altering frozen 224 caches."""
    if isinstance(image_size, bool) or image_size not in ZOOM_BY_IMAGE_SIZE:
        raise ValueError("V19 supports image sizes 224, 320 and 384")
    result = deepcopy(processor)
    result.image_processor.size = {"shortest_edge": int(image_size)}
    result.image_processor.crop_size = {"height": int(image_size), "width": int(image_size)}
    return result


def _processor_size(processor):
    crop = processor.image_processor.crop_size
    size = crop.get("height")
    if size not in ZOOM_BY_IMAGE_SIZE or crop != {"height": size, "width": size}:
        raise ValueError("V19 views require square 224, 320 or 384 processor crops")
    return int(size)


def build_zoom_transform(processor):
    size = _processor_size(processor)
    settings = processor.image_processor
    return transforms.Compose([
        transforms.Resize(ZOOM_BY_IMAGE_SIZE[size],
                          interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.image_mean, std=settings.image_std),
    ])


class RobustPairedDataset(Dataset):
    def __init__(self, rows, indices, processor):
        if _processor_size(processor) not in (320, 384):
            raise ValueError("V19 training views require 320 or 384 inputs")
        self.rows = rows
        self.indices = list(indices)
        self.transform_a = build_light_transform(processor)
        self.transform_b = build_light_transform(processor)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row_index = self.indices[item]
        path, label, class_name = self.rows[row_index]
        image = safe_open_image(path)
        # Preserve V15's RNG order and original image as the input to degradation.
        # degrade_view_b keeps 320--640 downscale and JPEG 65--95 at every size.
        view_a = self.transform_a(image)
        view_b = self.transform_b(degrade_view_b(image))
        return {
            "pixel_values_a": view_a, "pixel_values_b": view_b,
            "label": torch.tensor(label, dtype=torch.long),
            "row_index": torch.tensor(row_index, dtype=torch.long),
            "path": path, "class_name": class_name,
        }


# Compatibility import for existing train entry points; V19 always uses robust views.
HighResolutionPairedDataset = RobustPairedDataset


def training_dataset(rows, indices, processor, config):
    validate_recipe_config(config)
    if _processor_size(processor) != config["image_size"]:
        raise ValueError("V19 training processor resolution differs from recipe")
    return RobustPairedDataset(rows, indices, processor)


class FourViewTestDataset(OriginalFourViewTestDataset):
    def __init__(self, paths, processor):
        super().__init__(paths, processor)
        # Legacy zoom256 keys identify the view, while config records its geometry.
        self.zoom_transform = build_zoom_transform(processor)


class FourViewFolderDataset(OriginalFourViewFolderDataset):
    def __init__(self, rows, indices, processor):
        super().__init__(rows, indices, processor)
        self.zoom_transform = build_zoom_transform(processor)
