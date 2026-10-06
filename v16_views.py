"""V16 uses the unchanged expanded light transforms for both paired views."""

from v11_resolution import IMAGE_SIZE, HighResolutionPairedDataset
from v16_core import validate_recipe_augmentation


def training_dataset(rows, indices, processor, config):
    validate_recipe_augmentation(config)
    if processor.image_processor.crop_size != {"height": IMAGE_SIZE, "width": IMAGE_SIZE}:
        raise ValueError("V16 training views require a 320-pixel processor")
    return HighResolutionPairedDataset(rows, indices, processor)
