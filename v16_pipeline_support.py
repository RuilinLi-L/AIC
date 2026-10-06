"""V16 orchestration contracts, GPU queueing, wall-clock budget and delivery provenance."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from v15_pipeline_support import atomic_json, sha256


def now():
    return datetime.datetime.now().astimezone().isoformat()


def pipeline_dir(root):
    return Path(os.environ.get('AIC_PIPELINE_DIR', str(root / 'v16_pipeline')))


def initialize(root):
    path = pipeline_dir(root) / 'budget.json'
    if not path.exists():
        started = float(os.environ.get('AIC_PIPELINE_STARTED_AT', time.time()))
        if not math.isfinite(started) or started < 0 or started > time.time():
            raise ValueError('AIC_PIPELINE_STARTED_AT must be a finite past Unix timestamp')
        atomic_json(path, {'format_version': 16,
                           'started_at': datetime.datetime.fromtimestamp(started).astimezone().isoformat(),
                           'started_at_unix': started,
                           'budget_hours': 48, 'includes': ['preparation', 'gpu_queue', 'validation',
                           'evaluation', 'baseline_wait', 'selection', 'refit', 'prediction'],
                           'automatic_short_training': False})
    return json.loads(path.read_text())


def state(root, stage, status, filename='status.json', detail=None):
    budget = initialize(root)
    elapsed = max(0, time.time() - budget['started_at_unix'])
    result = {'format_version': 16, 'stage': stage, 'status': status, 'detail': detail,
              'updated_at': now(), 'elapsed_wall_clock_seconds': elapsed,
              'budget_hours': budget['budget_hours'], 'budget_exceeded': elapsed > 48 * 3600,
              'automatic_short_training': False}
    atomic_json(pipeline_dir(root) / filename, result)
    return result


def wait_gpu(root, gpu, minimum, timeout, interval=30, state_file='gpu_status.json'):
    if not str(gpu).isdigit() or minimum < 1 or timeout < 0:
        raise ValueError('explicit numeric GPU, positive memory threshold and nonnegative timeout required')
    start = time.monotonic()
    while True:
        result = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
                                 '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
        value = result.stdout.strip()
        if not value.isdigit():
            raise ValueError(f'invalid nvidia-smi memory response: {value!r}')
        free = int(value)
        if free >= minimum:
            state(root, 'gpu_ready', 'ready', state_file,
                  {'gpu': str(gpu), 'free_mib': free, 'required_mib': minimum,
                   'queue_seconds': time.monotonic() - start})
            return
        state(root, 'gpu_queue', 'waiting', state_file,
              {'gpu': str(gpu), 'free_mib': free, 'required_mib': minimum,
               'queue_seconds': time.monotonic() - start})
        if time.monotonic() - start >= timeout:
            raise TimeoutError(f'GPU {gpu} free={free} MiB below required={minimum} MiB for {timeout}s')
        print(f'v16_gpu_queue gpu={gpu} free_mib={free} required_mib={minimum}', flush=True)
        time.sleep(min(interval, max(0, timeout - (time.monotonic() - start))))


def wait_baselines(root, directories, timeout, interval=30):
    start = time.monotonic()
    while True:
        missing = [str(path / name) for path in directories for name in ('strict_eval.json', 'model.pt')
                   if not (path / name).is_file()]
        if not missing:
            identity = {str(path.resolve()): {name: sha256(path / name)
                        for name in ('strict_eval.json', 'model.pt')} for path in directories}
            binding = pipeline_dir(root) / 'baseline_binding.json'
            if binding.exists() and json.loads(binding.read_text()) != identity:
                raise ValueError('baseline evaluation changed after V16 consumed it')
            if not binding.exists():
                atomic_json(binding, identity)
            return
        # A dependency may already have the required validation products even if a later refit failed.
        for version in ('v14', 'v15'):
            dependency = root / f'{version}_pipeline/status.json'
            if dependency.exists():
                status = json.loads(dependency.read_text())
                relevant_missing = any(version in item for item in missing)
                if relevant_missing and status.get('status') in ('failed', 'interrupted'):
                    raise RuntimeError(f'{version} dependency failed before required artifacts: {status}')
        state(root, 'baseline_wait', 'waiting', detail={'missing': missing,
              'wait_seconds': time.monotonic() - start})
        if time.monotonic() - start >= timeout:
            raise TimeoutError(f'baseline wait exceeded {timeout}s; missing: {missing}')
        print(f'v16_wait missing={missing}', flush=True)
        time.sleep(min(interval, max(0, timeout - (time.monotonic() - start))))


def decision(root, selection=None):
    from select_v16 import read_decision
    path = Path(selection or os.environ.get('AIC_SELECTION', str(root / 'v16_selection.json')))
    result = read_decision(path)
    identity = {'selection': str(path.resolve()), 'sha256': sha256(path)}
    binding = pipeline_dir(root) / 'selection_binding.json'
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError('V16 selection changed after being consumed')
    if not binding.exists():
        atomic_json(binding, identity)
    return result, path


def _matches(checkpoint, selected):
    import torch
    keys = ('selected_epoch', 'dataset_signature', 'class_names')
    if any(checkpoint.get(key) != selected.get(key) for key in keys):
        return False
    if checkpoint.get('config', {}).get('recipe') != selected['recipe']:
        return False
    if any(checkpoint.get('calibration', {}).get(key) != selected['calibration'].get(key)
           for key in ('view_weights', 'alpha', 'precision')):
        return False
    try:
        return torch.equal(torch.as_tensor(checkpoint['class_bias']).float(),
                           torch.as_tensor(selected['class_bias']).float())
    except (KeyError, TypeError):
        return False


def checkpoint_path(root, kind, selection=None):
    import torch
    selected, path = decision(root, selection)
    if kind == 'validation':
        result = Path(selected['source_checkpoint']).resolve()
        if sha256(result) != selected['source_checkpoint_sha256']:
            raise ValueError('selected validation checkpoint changed')
        return result
    if kind != 'refit' or not selected['refit_required']:
        raise ValueError('only an accepted V16 winner can use the V16 refit checkpoint')
    result = Path(os.environ.get('AIC_REFIT_DIR', str(root / 'v16_refit'))) / 'model.pt'
    checkpoint = torch.load(result, map_location='cpu', weights_only=False)
    if (checkpoint.get('training_stage') != 'refit' or not _matches(checkpoint, selected)
            or checkpoint.get('refit_selection_sha256') != sha256(path)):
        raise ValueError('V16 refit checkpoint differs from frozen selection')
    return result.resolve()


def verify_package(directory, checkpoint_path_value, test_dir, expected_rows):
    import torch
    from robust_clip import list_test_images
    directory = Path(directory)
    csv_path, zip_path = directory / 'pred_results.csv', directory / 'pred_results.zip'
    with csv_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.reader(handle, skipinitialspace=True))
    names = [Path(path).name for path in list_test_images(test_dir)]
    if (len(rows) != expected_rows or any(len(row) != 2 for row in rows)
            or len(names) != expected_rows or len(set(names)) != expected_rows
            or len({row[0] for row in rows}) != expected_rows
            or set(names) != {row[0] for row in rows}):
        raise ValueError('submission row count, fields, duplicates or filenames are invalid')
    checkpoint = torch.load(checkpoint_path_value, map_location='cpu', weights_only=False)
    classes = set(checkpoint['class_names'])
    if any(len(row[1]) != 4 or not row[1].isdigit() or row[1] not in classes for row in rows):
        raise ValueError('submission contains invalid class identifiers')
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ['pred_results.csv'] or archive.read('pred_results.csv') != csv_path.read_bytes():
            raise ValueError('ZIP must contain only the identical pred_results.csv')
    return checkpoint, len(rows)


def export_submission(root, kind, checkpoint_value, test_dir, expected_rows=37444, source=None):
    import torch
    selected, selection_path = decision(root)
    checkpoint_value = Path(checkpoint_value).resolve()
    if kind != 'fallback_refit' and checkpoint_value != checkpoint_path(root, kind):
        raise ValueError('submission checkpoint differs from frozen selection')
    if kind == 'fallback_refit':
        cp = torch.load(checkpoint_value, map_location='cpu', weights_only=False)
        if selected['refit_required'] or cp.get('training_stage') != 'refit' or not _matches(cp, selected):
            raise ValueError('fallback refit does not match selected baseline')
    out = Path(os.environ.get('AIC_DELIVERY_DIR', str(root / 'v16_delivery'))) / kind
    out.mkdir(parents=True, exist_ok=True)
    if source:
        source = Path(source)
        provenance_path = source / 'provenance.json'
        if not provenance_path.exists():
            raise ValueError('existing package has no checkpoint provenance')
        provenance = json.loads(provenance_path.read_text())
        if provenance.get('checkpoint_sha256') != sha256(checkpoint_value):
            raise ValueError('existing package checkpoint hash mismatch')
        verify_package(source, checkpoint_value, test_dir, expected_rows)
        for name in ('pred_results.csv', 'pred_results.zip'):
            temp = out / f'.{name}.{os.getpid()}.tmp'
            shutil.copyfile(source / name, temp)
            temp.replace(out / name)
    checkpoint, rows = verify_package(out, checkpoint_value, test_dir, expected_rows)
    result = {'format_version': 16, 'kind': kind, 'checkpoint': str(checkpoint_value),
              'checkpoint_sha256': sha256(checkpoint_value), 'selection_sha256': sha256(selection_path),
              'csv_sha256': sha256(out / 'pred_results.csv'), 'zip_sha256': sha256(out / 'pred_results.zip'),
              'recipe': checkpoint['config']['recipe'], 'stage': checkpoint['training_stage'],
              'epoch': checkpoint['selected_epoch'], 'view_weights': checkpoint['calibration']['view_weights'],
              'alpha': checkpoint['calibration']['alpha'], 'precision': checkpoint['calibration']['precision'],
              'linecount': rows, 'online_score': None, 'created_at': now(),
              'reused_from': str(source.resolve()) if source else None}
    existing = out / 'provenance.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        for key in ('checkpoint_sha256', 'selection_sha256', 'csv_sha256', 'zip_sha256'):
            if previous.get(key) != result[key]:
                raise ValueError(f'completed V16 delivery changed: {key}')
        return previous
    atomic_json(existing, result)
    return result


def reusable_package(root, kind):
    """Return only previously proven packages; absence is normal and never starts training."""
    import torch
    selected, _ = decision(root)
    directories = [root / 'v15_delivery' / name for name in
                   ('v14_validation', 'v14_refit', 'v15_validation', 'v15_refit')]
    directories += [root / 'v14_refit', root / 'v15_refit']
    override = os.environ.get('AIC_FALLBACK_VALIDATION_DIR' if kind == 'validation' else 'AIC_FALLBACK_REFIT_DIR')
    if override:
        directories.insert(0, Path(override))
    for directory in directories:
        provenance = directory / 'provenance.json'
        if not provenance.is_file():
            continue
        record = json.loads(provenance.read_text())
        checkpoint = Path(record.get('checkpoint', ''))
        if not checkpoint.is_file() or not all((directory / name).is_file() for name in ('pred_results.csv', 'pred_results.zip')):
            continue
        if record.get('checkpoint_sha256') != sha256(checkpoint):
            continue
        if kind == 'validation':
            if sha256(checkpoint) == selected['source_checkpoint_sha256']:
                return {'directory': str(directory.resolve()), 'checkpoint': str(checkpoint.resolve())}
        elif not selected['refit_required']:
            value = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if value.get('training_stage') == 'refit' and _matches(value, selected):
                return {'directory': str(directory.resolve()), 'checkpoint': str(checkpoint.resolve())}
    return None


def benchmark_needed(path, recipe):
    path = Path(path)
    if not path.exists():
        return True
    report = json.loads(path.read_text())
    if report.get('format_version') == 16 and report.get('recipe') == recipe and report.get('success') is False:
        return True
    benchmark_settings(path, recipe)
    return False


def benchmark_settings(path, recipe):
    report = json.loads(Path(path).read_text())
    if (report.get('format_version') != 16 or report.get('recipe') != recipe
            or report.get('success') is not True or report.get('initialization') != 'official_base_only'
            or report.get('trained_checkpoint_weights_loaded') is not False
            or report.get('pin_memory') is not False or report.get('prefetch_factor') != 1
            or report.get('chosen_workers') != 8):
        raise ValueError('benchmark is not a successful official-base V16 probe for this recipe')
    batch = report.get('batch_size')
    if batch not in (128, 256) or report.get('gradient_accumulation') * batch != 256:
        raise ValueError('benchmark must preserve effective batch 256')
    if batch == 128 and not any(row.get('batch_size') == 256 and row.get('status') == 'out_of_memory'
                                for row in report.get('configurations', [])):
        raise ValueError('128 microbatch requires a recorded actual 256 CUDA OOM')
    return report['chosen_workers'], batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'state', 'wait-gpu', 'wait-baselines', 'accepted',
                         'recipe', 'checkpoint', 'export', 'reusable', 'bind', 'benchmark-settings', 'benchmark-needed'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--state-file', default='status.json')
    parser.add_argument('--stage', default='starting'); parser.add_argument('--status', default='running')
    parser.add_argument('--benchmark', type=Path); parser.add_argument('--recipe')
    parser.add_argument('--detail'); parser.add_argument('--gpu')
    parser.add_argument('--minimum', type=int, default=32768)
    parser.add_argument('--timeout', type=float, default=48*3600)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--baseline-expanded', type=Path); parser.add_argument('--baseline-v15', type=Path)
    parser.add_argument('--kind', choices=('validation', 'refit', 'fallback_refit'))
    parser.add_argument('--checkpoint'); parser.add_argument('--source', type=Path)
    parser.add_argument('--test-dir', type=Path); parser.add_argument('--expected-rows', type=int, default=37444)
    args = parser.parse_args()
    if args.action == 'benchmark-needed': print(int(benchmark_needed(args.benchmark, args.recipe)))
    elif args.action == 'benchmark-settings': print(*benchmark_settings(args.benchmark, args.recipe))
    elif args.action == 'init': print(initialize(args.root)['started_at_unix'])
    elif args.action == 'state': state(args.root, args.stage, args.status, args.state_file, args.detail)
    elif args.action == 'wait-gpu': wait_gpu(args.root, args.gpu, args.minimum, args.timeout, args.interval, args.state_file)
    elif args.action == 'wait-baselines':
        wait_baselines(args.root, [args.baseline_expanded or args.root / 'v14_baseline_expanded',
                       args.baseline_v15 or args.root / 'v15_expanded_mlp'], args.timeout, args.interval)
    elif args.action in ('accepted', 'recipe', 'bind'):
        selected, _ = decision(args.root)
        if args.action == 'accepted': print(int(selected['refit_required']))
        elif args.action == 'recipe': print(selected['recipe'])
    elif args.action == 'checkpoint': print(checkpoint_path(args.root, args.kind))
    elif args.action == 'export':
        print(json.dumps(export_submission(args.root, args.kind, args.checkpoint, args.test_dir,
                                          args.expected_rows, args.source)))
    elif args.action == 'reusable': print(json.dumps(reusable_package(args.root, args.kind)))


if __name__ == '__main__': main()
