"""V13 progress labels with shared, unchanged V12 numerical helpers."""
import torch
from v12_runtime import (autocast_context, BatchedFeatureQueue, loader_options,
                         synchronized_time, precision_name)


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
            print(f"v13_progress stage={self.stage} batches={batch_id}/{self.batches} "
                  f"rows={self.rows} images_per_s={self.rows / max(elapsed, 1e-9):.2f}"
                  f" elapsed_s={elapsed:.1f}{memory}", flush=True)


