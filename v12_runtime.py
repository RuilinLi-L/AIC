"""V12-only throughput helpers; older experiments retain their original behavior."""

from contextlib import nullcontext
import time

import torch
import torch.nn.functional as F

from v6_core import SelectiveFeatureQueue as OriginalQueue


def precision_name(device):
    return "bf16_autocast_fp32_stats" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"


def autocast_context(device):
    if precision_name(device).startswith("bf16"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def loader_options(workers, prefetch_factor=2):
    if workers < 0 or prefetch_factor < 1:
        raise ValueError("workers must be nonnegative and prefetch-factor positive")
    return {"prefetch_factor": prefetch_factor} if workers else {}


def synchronized_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


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
            print(f"v12_progress stage={self.stage} batches={batch_id}/{self.batches} "
                  f"rows={self.rows} images_per_s={self.rows / max(elapsed, 1e-9):.2f}"
                  f" elapsed_s={elapsed:.1f}{memory}", flush=True)


class BatchedFeatureQueue(OriginalQueue):
    @torch.no_grad()
    def enqueue(self, features, labels, trusted):
        # Stable class grouping retains source order. Only the final capacity
        # writes per class survive, making all destination slots unique.
        values = F.normalize(features.detach().float(), dim=-1).to(torch.float16)[trusted]
        selected = labels[trusted]
        if selected.numel() == 0:
            return
        order = torch.argsort(selected, stable=True)
        selected, values = selected[order], values[order]
        counts = torch.bincount(selected, minlength=self.num_classes)
        starts = counts.cumsum(0) - counts
        ordinal = torch.arange(len(selected), device=self.device) - starts[selected]
        keep = ordinal >= counts[selected] - self.capacity
        classes = selected[keep]
        slots = (self.pointer[classes] + ordinal[keep]) % self.capacity
        self.features[classes, slots] = values[keep]
        self.valid[classes, slots] = True
        self.pointer.add_(counts)
