"""Isolated real-training V16 worker benchmark; never saves model checkpoints.

The source checkpoint supplies cache identity only. Both this probe and formal
V16 training initialize independently from official weights; probe weights are discarded.
Controlled epoch-5 histories exercise dynamic repair, gradient projection, the
full contrastive queue and EMA inference without running two full audit epochs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from robust_clip import load_clip, resolve_device, trainable_state_dict
from train_v5 import clone_trainable
from train_v6 import configure_determinism, make_v6_scheduler
from v7_data import load_manifest_dataset
from v16_core import RECIPES, build_optimizer, recipe_config, stage_indices
from v16_model import build_classifier, load_classifier_state, resolution_processor
from v16_core import model_identity
from train_v6 import validate_clip_vit_b32
from benchmark_v14 import load_frozen_features
from v14_runtime import BatchedFeatureQueue, loader_options, precision_name, synchronized_time
from v16_views import training_dataset


SEED = 2026
EFFECTIVE_BATCH = 256


class BatchWindow:
    def __init__(self, iterator, batches):
        self.iterator, self.batches = iterator, batches

    def __len__(self):
        return self.batches

    def __iter__(self):
        return islice(self.iterator, self.batches)


class DeterministicImages(Dataset):
    """Use identical augmentations with different worker counts and microbatches."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Runs inside CPU workers; avoid torch.manual_seed, which also seeds CUDA.
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(SEED + index)
            np.random.seed(SEED + index)
            torch.set_rng_state(torch.Generator().manual_seed(SEED + index).get_state())
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


def isolated_state(labels, ambiguous, class_count):
    n = len(labels)
    quality = np.where(np.arange(n) % 2 == 0, .8, .4).astype(np.float32)
    quality[ambiguous] = .25
    repair = (np.arange(n) % 16 == 1) & ~ambiguous
    historical = np.full((n, class_count), .1 / class_count, dtype=np.float32)
    historical[np.arange(n), labels] += .9
    targets = historical.copy()
    alternate = (labels[repair] + 1) % class_count
    targets[repair] = .1 / class_count
    targets[np.flatnonzero(repair), alternate] += .9
    return dict(
        quality=quality, trusted=(quality >= .7) & ~ambiguous,
        repair_plan=repair, temporal=historical.astype(np.float16),
        temporal_seen=np.ones(n, dtype=np.bool_), audit_probabilities=targets.astype(np.float16),
        previous_view_prediction=labels.copy(), stability_sum=np.zeros(n, np.float32),
        stability_count=np.zeros(n, np.int16), loss_sum=np.zeros(n, np.float32),
        loss_count=np.zeros(n, np.int16),
    )


def benchmark(args, classifier, processor, manifest, features, prototypes, device):
    from train_v16 import run_epoch

    if args.steps < 1 or args.warmup_steps < 1:
        raise ValueError('steps and warmup-steps must be positive')
    train, _, validation = stage_indices(manifest, 'validate')
    if not train:
        raise ValueError('benchmark requires a nonempty training split')
    count = EFFECTIVE_BATCH * (args.warmup_steps + args.steps)
    indices = np.random.default_rng(SEED).choice(train, count, replace=count > len(train))
    if set(indices) & set(validation):
        raise ValueError('benchmark sampled validation data')
    rows = [manifest.rows[index] for index in indices]
    labels = np.asarray([row[1] for row in rows], dtype=np.int64)
    ambiguous = manifest.ambiguous_mask[indices]
    candidates = [manifest.candidate_labels[index] for index in indices]
    centers = np.asarray(features[0, indices]).copy()
    if not np.isfinite(centers).all():
        raise ValueError('benchmark frozen features contain nonfinite values')
    prototypes = F.normalize(torch.as_tensor(prototypes, dtype=torch.float32, device=device), dim=1)
    class_count = len(manifest.class_names)
    config = recipe_config(args.recipe)
    dataset = DeterministicImages(training_dataset(rows, range(count), processor, config))
    initial, was_training = trainable_state_dict(classifier), classifier.training
    results = []
    print('v16_benchmark isolated=1 real_images=1 epoch=5 dynamic_repair=1 '
          'full_queue=1 gradients=real ema=1 formal_checkpoints_written=0', flush=True)

    def probe(workers, batch_size):
        accumulation = EFFECTIVE_BATCH // batch_size
        result = dict(workers=workers, batch_size=batch_size, gradient_accumulation=accumulation,
                      prefetch_factor=1, pin_memory=False, success=False)
        optimizer = scheduler = queue = ema = state = loader = iterator = common = None
        try:
            configure_determinism(SEED)
            load_classifier_state(classifier, initial)
            optimizer, _, _ = build_optimizer(classifier, config)
            scheduler = make_v6_scheduler(optimizer, math.ceil(len(train) / EFFECTIVE_BATCH) * 24)
            queue = BatchedFeatureQueue(class_count, 16, prototypes.shape[1], device)
            queue_labels = torch.arange(class_count, device=device).repeat_interleave(16)
            queue.enqueue(prototypes.repeat_interleave(16, dim=0), queue_labels,
                          torch.ones(len(queue_labels), dtype=torch.bool, device=device))
            state = isolated_state(labels, ambiguous, class_count)
            ema = clone_trainable(classifier)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                pin_memory=False, **loader_options(workers, 1),
                                generator=torch.Generator().manual_seed(SEED))
            iterator = iter(loader)
            common = dict(classifier=classifier, optimizer=optimizer, scheduler=scheduler, ema=ema,
                          queue=queue, device=device, epoch=5, labels=labels,
                          class_prior=torch.full((class_count,), 1 / class_count, device=device),
                          center_features=centers, prototypes=prototypes, ambiguous_mask=ambiguous,
                          candidate_labels=candidates, gradient_accumulation=accumulation,
                          soft_teacher=True, dynamic_noise=True, schedule_epochs=24, **state)
            _, step = run_epoch(loader=BatchWindow(iterator, args.warmup_steps * accumulation),
                                optimizer_step=0, **common)
            if step != args.warmup_steps:
                raise RuntimeError('benchmark warmup did not complete requested optimizer steps')
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            started = synchronized_time(device)
            stats, final_step = run_epoch(loader=BatchWindow(iterator, args.steps * accumulation),
                                          optimizer_step=step, **common)
            elapsed = synchronized_time(device) - started
            if final_step - step != args.steps or not np.isfinite(stats['loss']):
                raise RuntimeError('benchmark timed training was incomplete or nonfinite')
            result.update(success=True, status='ok', images_per_s=args.steps * EFFECTIVE_BATCH / elapsed,
                          seconds=elapsed, measured_optimizer_steps=args.steps, stats=stats,
                          peak_allocated_gib=(torch.cuda.max_memory_allocated(device) / 2**30
                                              if device.type == 'cuda' else None))
        except torch.OutOfMemoryError as error:
            result.update(status='out_of_memory', error=str(error))
        except Exception as error:
            result.update(status='error', error=f'{type(error).__name__}: {error}')
        finally:
            # Also release worker processes after a CUDA OOM or data exception.
            if iterator is not None and hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()
            del common, iterator, loader, optimizer, scheduler, queue, ema, state
            for parameter in classifier.parameters():
                parameter.grad = None
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        print(json.dumps(result), flush=True)
        return result

    try:
        results.append(probe(8, 256))
        # Prefer the specified microbatch 256. Only actual CUDA OOM enables 128x2.
        if not any(row['success'] for row in results) and any(
                row['status'] == 'out_of_memory' for row in results):
            results.append(probe(8, 128))
    finally:
        load_classifier_state(classifier, initial)
        classifier.train(was_training)

    valid = [row for row in results if row['success']]
    winner = max(valid, key=lambda row: (row['images_per_s'], -row['workers'])) if valid else None
    report = dict(format_version=16, success=bool(winner), recipe=args.recipe,
                  initialization='official_base_only', trained_checkpoint_weights_loaded=False,
                  chosen_workers=winner['workers'] if winner else None,
                  batch_size=winner['batch_size'] if winner else None,
                  gradient_accumulation=winner['gradient_accumulation'] if winner else None,
                  prefetch_factor=1, pin_memory=False, precision=precision_name(device),
                  warmup_optimizer_steps=args.warmup_steps, measured_optimizer_steps=args.steps,
                  sampled_train_rows=count, sampled_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  dataset_signature=manifest.signature, configurations=results,
                  scenario='real train images; controlled epoch-5 targets and full queue; excludes full audits/validation')
    if winner:
        raw_hours = (len(train) + len(manifest.rows)) * 24 / winner['images_per_s'] / 3600
        report['budget_estimate'] = {
            'target_hours': 48, 'validation_epochs': 24, 'maximum_refit_epochs': 24,
            'raw_training_hours': raw_hours,
            'planning_hours_range': [1 + raw_hours * 1.5, 1 + raw_hours * 2.5],
            'method': 'throughput scaled to validate+maximum refit; heuristic 1.5-2.5x for audits/evaluation/prediction plus 1h setup',
            'excludes': 'external V14 dependency waits and changing server contention',
            'not_a_completion_guarantee': True,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(output)
    print(json.dumps(report, indent=2), flush=True)
    if not winner:
        raise RuntimeError(f'all V16 benchmark configurations failed; see {output}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--data-manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--feature-cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--recipe', choices=RECIPES, default='mlp_light')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--warmup-steps', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_steps < 1:
        parser.error('steps and warmup-steps must be positive')
    output = Path(args.output).resolve()
    protected = [Path(value).resolve() for value in (args.checkpoint, args.feature_cache, args.data_manifest)]
    cache = Path(args.feature_cache).resolve()
    protected.append(cache.with_suffix(cache.suffix + '.json'))
    if output in protected or output.suffix != '.json':
        parser.error('output must be a separate .json report')
    device = resolve_device(args.device)
    configure_determinism(SEED)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') not in (13, 14, 15, 16) or checkpoint['config']['recipe'] not in (
            'expanded', 'expanded_robust', 'expanded_mlp', *RECIPES):
        raise ValueError('benchmark metadata requires an expanded V13-V16 checkpoint')
    manifest = load_manifest_dataset(args.data_manifest, args.train_dir, checkpoint['config']['conflict_policy'])
    if (checkpoint['dataset_signature'] != manifest.signature or
            checkpoint['class_names'] != manifest.class_names):
        raise ValueError('benchmark checkpoint does not match the manifest/classes')
    features = load_frozen_features(args.feature_cache, checkpoint, manifest)
    if checkpoint['config']['base_model_identity'] != model_identity(args.model_dir):
        raise ValueError('metadata source and official base weights differ')
    # The old checkpoint supplies cache identity only, never learned weights.
    _, clean_train, _ = stage_indices(manifest, 'validate')
    labels = np.asarray([row[1] for row in manifest.rows])
    prototypes = np.zeros((len(manifest.class_names), features.shape[-1]), np.float32)
    for class_id in range(len(manifest.class_names)):
        ids = np.asarray(clean_train)[labels[clean_train] == class_id]
        if len(ids):
            prototypes[class_id] = np.asarray(features[0, ids]).mean(axis=0)
    model, processor = load_clip(args.model_dir, device)
    validate_clip_vit_b32(model)
    classifier = build_classifier(model, prototypes, device, recipe_config(args.recipe))
    processor = resolution_processor(processor)
    benchmark(args, classifier, processor, manifest, features, prototypes, device)


if __name__ == '__main__':
    main()
