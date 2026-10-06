"""V14 runtime controls, independent of the frozen scientific configuration."""

import os

import torch

from v12_runtime import (
    BatchedFeatureQueue,
    autocast_context,
    loader_options,
    precision_name,
    synchronized_time,
)


RUNTIME_CONFIG_KEYS = frozenset({
    "workers", "prefetch_factor", "eval_batch_size", "pin_memory",
    "batch_size", "gradient_accumulation",
})


def pin_memory_enabled(device):
    value = os.environ.get("AIC_PIN_MEMORY", "0")
    if value not in ("0", "1"):
        raise ValueError("AIC_PIN_MEMORY must be 0 or 1")
    return device.type == "cuda" and value == "1"


def scientific_config(config):
    """Allow loader tuning/resized microbatches while enforcing effective batch 256."""
    batch = config.get("batch_size", 256)
    accumulation = config.get("gradient_accumulation", 1)
    if (isinstance(batch, bool) or isinstance(accumulation, bool)
            or not isinstance(batch, int) or not isinstance(accumulation, int)
            or batch < 1 or accumulation < 1 or batch * accumulation != 256):
        raise ValueError("V14 requires batch-size * gradient-accumulation = 256")
    return {key: value for key, value in config.items() if key not in RUNTIME_CONFIG_KEYS}


class Progress:
    def __init__(self, stage, device, batches):
        self.stage, self.device, self.batches = stage, device, batches
        self.started = synchronized_time(device)
        self.rows = 0

    def update(self, batch_id, rows):
        self.rows += rows
        if batch_id % 50 == 0 or batch_id == self.batches:
            elapsed = synchronized_time(self.device) - self.started
            memory = ""
            if self.device.type == "cuda":
                memory = (f" allocated_gib={torch.cuda.memory_allocated(self.device) / 2**30:.2f}"
                          f" peak_gib={torch.cuda.max_memory_allocated(self.device) / 2**30:.2f}")
            print(f"v14_progress stage={self.stage} batches={batch_id}/{self.batches} "
                  f"rows={self.rows} images_per_s={self.rows / max(elapsed, 1e-9):.2f}"
                  f" elapsed_s={elapsed:.1f}{memory}", flush=True)
