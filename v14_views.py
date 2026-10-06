"""320-pixel expanded views, with V10 quality augmentation on view B only."""

import torch
from torch.utils.data import Dataset

from robust_clip import safe_open_image
from v10_views import degrade_view_b
from v11_resolution import IMAGE_SIZE, HighResolutionPairedDataset, build_light_transform
from v14_core import validate_recipe_augmentation


class Robust320PairedDataset(Dataset):
    def __init__(self, rows, indices, processor):
        if processor.image_processor.crop_size != {"height": IMAGE_SIZE, "width": IMAGE_SIZE}:
            raise ValueError("V14 training views require a 320-pixel processor")
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
        # Preserve the exact expanded view-A transform and draw it before any
        # quality augmentation. View B starts from the original decoded image.
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


def training_dataset(rows, indices, processor, config):
    validate_recipe_augmentation(config)
    if processor.image_processor.crop_size != {"height": IMAGE_SIZE, "width": IMAGE_SIZE}:
        raise ValueError("V14 training views require a 320-pixel processor")
    dataset_type = Robust320PairedDataset if config["recipe"] == "expanded_robust" else HighResolutionPairedDataset
    return dataset_type(rows, indices, processor)
