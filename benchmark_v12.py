"""Isolated V12 throughput probe, dispatched by train_v12 --benchmark-only.

Images/prototypes come from the real training split. A controlled epoch-5
history and a populated queue exercise late-training branches without needing
to run two complete EMA audits. This measures throughput, not model quality.
"""

import gc
from itertools import islice
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from robust_clip import load_classifier_state, trainable_state_dict
from train_v5 import clone_trainable
from train_v6 import build_optimizer, configure_determinism, make_v6_scheduler
from v11_resolution import HighResolutionPairedDataset
from v12_runtime import BatchedFeatureQueue, loader_options, precision_name, synchronized_time


class BatchWindow:
    def __init__(self, iterator, batches):
        self.iterator, self.batches = iterator, batches

    def __len__(self):
        return self.batches

    def __iter__(self):
        return islice(self.iterator, self.batches)


def isolated_state(labels, ambiguous, class_count):
    n = len(labels)
    quality = np.where(np.arange(n) % 2 == 0, 0.8, 0.4).astype(np.float32)
    quality[ambiguous] = 0.25
    historical = np.full((n, class_count), 0.1 / class_count, dtype=np.float32)
    historical[np.arange(n), labels] += 0.9
    return dict(
        quality=quality, trusted=(quality >= 0.7) & ~ambiguous,
        repair_plan=(np.arange(n) % 16 == 1) & ~ambiguous,
        temporal=historical.astype(np.float16), temporal_seen=np.ones(n, dtype=np.bool_),
        previous_view_prediction=labels.copy(), stability_sum=np.zeros(n, np.float32),
        stability_count=np.zeros(n, np.int16), loss_sum=np.zeros(n, np.float32),
        loss_count=np.zeros(n, np.int16),
    )


def benchmark(args, classifier, processor, manifest, features, signals, device):
    from train_v12 import run_epoch

    # Never accept a path inside the formal run: even reports are separate.
    output = Path(args.benchmark_output).resolve() if args.benchmark_output else None
    if output is not None and (output == Path(args.output_dir).resolve()
                               or Path(args.output_dir).resolve() in output.parents):
        raise ValueError("benchmark-output must be outside the formal training directory")
    effective = 256
    n = effective * (args.benchmark_warmup_steps + args.benchmark_steps)
    rng = np.random.default_rng(2026)
    indices = rng.choice(manifest.train_indices, n, replace=n > len(manifest.train_indices))
    rows = [manifest.rows[i] for i in indices]
    labels = np.asarray([row[1] for row in rows], dtype=np.int64)
    ambiguous = manifest.ambiguous_mask[indices]
    candidates = [manifest.candidate_labels[i] for i in indices]
    centers = np.asarray(features[0, indices]).copy()
    prototypes = torch.as_tensor(signals.prototypes, device=device, dtype=torch.float32)
    class_count = len(manifest.class_names)
    # load_classifier_state also requires frozen non-CLIP state such as
    # logit_scale. clone_trainable alone omits it.
    initial = trainable_state_dict(classifier)
    results = []
    print("v12_benchmark isolated=1 epoch=5 real_images=1 controlled_history=1 "
          "queue=full repair_branch=enabled formal_checkpoints_written=0", flush=True)
    for batch_size in (64, 128, 256):
        accumulation = effective // batch_size
        configure_determinism(2026)
        load_classifier_state(classifier, initial)
        optimizer, _, _ = build_optimizer(classifier)
        scheduler = make_v6_scheduler(optimizer, math.ceil(len(manifest.train_indices) / effective) * 12)
        queue = BatchedFeatureQueue(class_count, 16, prototypes.shape[1], device)
        queue_labels = torch.arange(class_count, device=device).repeat_interleave(16)
        queue.enqueue(prototypes.repeat_interleave(16, dim=0), queue_labels,
                      torch.ones(len(queue_labels), dtype=torch.bool, device=device))
        state = isolated_state(labels, ambiguous, class_count)
        ema = clone_trainable(classifier)
        loader = DataLoader(
            HighResolutionPairedDataset(rows, range(n), processor), batch_size=batch_size,
            num_workers=args.workers, pin_memory=device.type == "cuda",
            **loader_options(args.workers, args.prefetch_factor),
            generator=torch.Generator().manual_seed(2026),
        )
        iterator = iter(loader)
        common = dict(
            classifier=classifier, optimizer=optimizer, scheduler=scheduler, ema=ema, queue=queue,
            device=device, epoch=5, labels=labels, class_prior=torch.full((class_count,), 1 / class_count, device=device),
            center_features=centers, prototypes=prototypes, ambiguous_mask=ambiguous,
            candidate_labels=candidates, gradient_accumulation=accumulation, soft_teacher=True, **state,
        )
        try:
            _, step = run_epoch(loader=BatchWindow(iterator, args.benchmark_warmup_steps * accumulation),
                                optimizer_step=0, **common)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = synchronized_time(device)
            stats, step = run_epoch(loader=BatchWindow(iterator, args.benchmark_steps * accumulation),
                                    optimizer_step=step, **common)
            elapsed = synchronized_time(device) - started
            if not np.isfinite(stats["loss"]):
                raise RuntimeError("benchmark loss is not finite")
            results.append(dict(batch_size=batch_size, gradient_accumulation=accumulation,
                                images_per_s=args.benchmark_steps * effective / elapsed,
                                seconds=elapsed, measured_optimizer_steps=args.benchmark_steps,
                                peak_allocated_gib=(torch.cuda.max_memory_allocated(device) / 2**30
                                                    if device.type == "cuda" else None), stats=stats))
        except torch.OutOfMemoryError:
            results.append(dict(batch_size=batch_size, gradient_accumulation=accumulation, status="out_of_memory"))
        finally:
            del common, iterator, loader, optimizer, scheduler, queue, ema, state
            for parameter in classifier.parameters():
                parameter.grad = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(json.dumps(results[-1]), flush=True)
    load_classifier_state(classifier, initial)
    valid = [r for r in results if "images_per_s" in r]
    report = dict(precision=precision_name(device), workers=args.workers, prefetch_factor=args.prefetch_factor,
                  scenario="real train images; controlled epoch-5 history and full queue; excludes full audits/validation",
                  configurations=results, fastest_batch_size=max(valid, key=lambda r: r["images_per_s"])["batch_size"] if valid else None)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not valid:
        raise RuntimeError("all benchmark configurations ran out of memory")
