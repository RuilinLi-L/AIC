"""Reuse V14's exact robust paired views under the V15 recipe contract."""

from v14_views import Robust320PairedDataset
from v15_core import validate_recipe_augmentation


def training_dataset(rows, indices, processor, config):
    validate_recipe_augmentation(config)
    return Robust320PairedDataset(rows, indices, processor)
