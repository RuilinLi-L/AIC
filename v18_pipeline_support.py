"""V18 orchestration contracts, GPU queueing, wall-clock budget and delivery provenance."""
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

RESOURCE_SPECS = {
    'resolution384': {'image_size': 384, 'lora_rank': 16, 'visual_trainable_parameters': 2654208},
    'rank32': {'image_size': 320, 'lora_rank': 32, 'visual_trainable_parameters': 5308416},
    'matched_control': {'image_size': 320, 'lora_rank': 16, 'visual_trainable_parameters': 2654208},
}
GATES = {'macro_gain': 0.002, 'tail_tolerance': 0.003, 'overall_noninferiority': True}


def now():
    return datetime.datetime.now().astimezone().isoformat()


def pipeline_dir(root):
    return Path(os.environ.get('AIC_PIPELINE_DIR', str(root / 'v18_pipeline')))


def initialize(root):
    path = pipeline_dir(root) / 'budget.json'
    if not path.exists():
        started = float(os.environ.get('AIC_PIPELINE_STARTED_AT', time.time()))
        if not math.isfinite(started) or started < 0 or started > time.time():
            raise ValueError('AIC_PIPELINE_STARTED_AT must be a finite past Unix timestamp')
        atomic_json(path, {'format_version': 18,
                           'started_at': datetime.datetime.fromtimestamp(started).astimezone().isoformat(),
                           'started_at_unix': started,
                           'budget_hours': 48, 'includes': ['preparation', 'gpu_queue', 'validation',
                           'evaluation', 'baseline_wait', 'selection', 'refit', 'prediction'],
                           'automatic_short_training': False})
    result = json.loads(path.read_text())
    if result.get('format_version') != 18:
        raise ValueError('V18 requires a fresh V18 budget record')
    return result


def state(root, stage, status, filename='status.json', detail=None):
    budget = initialize(root)
    elapsed = max(0, time.time() - budget['started_at_unix'])
    result = {'format_version': 18, 'stage': stage, 'status': status, 'detail': detail,
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
        print(f'v18_gpu_queue gpu={gpu} free_mib={free} required_mib={minimum}', flush=True)
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
                raise ValueError('baseline evaluation changed after V18 consumed it')
            if not binding.exists():
                atomic_json(binding, identity)
            return
        # A dependency may already have the required validation products even if a later refit failed.
        for version in ('v15',):
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
        print(f'v18_wait missing={missing}', flush=True)
        time.sleep(min(interval, max(0, timeout - (time.monotonic() - start))))


def decision(root, selection=None):
    from select_v18 import read_decision
    path = Path(selection or os.environ.get('AIC_SELECTION', str(root / 'v18_selection.json')))
    result = read_decision(path)
    resource_path = Path(os.environ.get('AIC_RESOURCE_JSON', result.get('resource_json',
                         str(pipeline_dir(root) / 'resource_plan.json'))))
    resource = read_resource_plan(resource_path)
    if (result.get('resource_sha256') != sha256(resource_path)
            or result.get('resource_branch') != resource['branch']):
        raise ValueError('V18 selection differs from frozen resource branch')
    identity = {'selection': str(path.resolve()), 'sha256': sha256(path)}
    binding = pipeline_dir(root) / 'selection_binding.json'
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError('V18 selection changed after being consumed')
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
    if checkpoint.get('config', {}).get('base_model_identity') != selected.get('base_model_identity'):
        return False
    if checkpoint.get('format_version') != selected.get('source_format_version'):
        return False
    if checkpoint.get('format_version') == 18:
        config = checkpoint.get('config', {})
        chosen = selected.get('training_config', {})
        if any(config.get(key) != chosen.get(key) for key in
               ('batch_size', 'gradient_accumulation', 'workers', 'image_size', 'zoom_shortest_edge',
                'lora_rank', 'lora_alpha', 'mlp_lora_rank', 'mlp_lora_alpha', 'schedule_epochs',
                'precision', 'resource_sha256', 'resource_branch')):
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
        raise ValueError('only an accepted V18 winner can use the V18 refit checkpoint')
    result = Path(os.environ.get('AIC_REFIT_DIR', str(root / 'v18_refit'))) / 'model.pt'
    checkpoint = torch.load(result, map_location='cpu', weights_only=False)
    if (checkpoint.get('training_stage') != 'refit' or not _matches(checkpoint, selected)
            or checkpoint.get('refit_selection_sha256') != sha256(path)):
        raise ValueError('V18 refit checkpoint differs from frozen selection')
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
    out = Path(os.environ.get('AIC_DELIVERY_DIR', str(root / 'v18_delivery'))) / kind
    out.mkdir(parents=True, exist_ok=True)
    if source:
        source = Path(source)
        provenance_path = source / 'provenance.json'
        if not provenance_path.exists():
            raise ValueError('existing package has no checkpoint provenance')
        provenance = json.loads(provenance_path.read_text())
        if provenance.get('checkpoint_sha256') != sha256(checkpoint_value):
            raise ValueError('existing package checkpoint hash mismatch')
        for name, key in (('pred_results.csv', 'csv_sha256'), ('pred_results.zip', 'zip_sha256')):
            if provenance.get(key) != sha256(source / name):
                raise ValueError(f'existing package {name} hash mismatch')
        verify_package(source, checkpoint_value, test_dir, expected_rows)
        for name in ('pred_results.csv', 'pred_results.zip'):
            temp = out / f'.{name}.{os.getpid()}.tmp'
            shutil.copyfile(source / name, temp)
            temp.replace(out / name)
    checkpoint, rows = verify_package(out, checkpoint_value, test_dir, expected_rows)
    result = {'format_version': 18, 'kind': kind, 'checkpoint': str(checkpoint_value),
              'checkpoint_sha256': sha256(checkpoint_value), 'selection_sha256': sha256(selection_path),
              'csv_sha256': sha256(out / 'pred_results.csv'), 'zip_sha256': sha256(out / 'pred_results.zip'),
              'recipe': checkpoint['config']['recipe'], 'stage': checkpoint['training_stage'],
              'source_format_version': checkpoint['format_version'],
              'dataset_signature': checkpoint['dataset_signature'],
              'base_model_identity': checkpoint['config'].get('base_model_identity'),
              'resource_sha256': selected['resource_sha256'], 'resource_branch': selected['resource_branch'],
              'epoch': checkpoint['selected_epoch'], 'view_weights': checkpoint['calibration']['view_weights'],
              'alpha': checkpoint['calibration']['alpha'], 'precision': checkpoint['calibration']['precision'],
              'linecount': rows, 'online_score': None, 'target_online_score': 72, 'created_at': now(),
              'reused_from': str(source.resolve()) if source else None}
    existing = out / 'provenance.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        for key in ('checkpoint_sha256', 'selection_sha256', 'csv_sha256', 'zip_sha256'):
            if previous.get(key) != result[key]:
                raise ValueError(f'completed V18 delivery changed: {key}')
        return previous
    atomic_json(existing, result)
    return result


def reusable_package(root, kind):
    """Return only previously proven packages; absence is normal and never starts training."""
    import torch
    selected, _ = decision(root)
    if kind not in ('validation', 'refit'):
        raise ValueError('reusable package kind must be validation or refit')
    directories = [root / 'v15_delivery' / name for name in ('v15_validation', 'v15_refit')]
    directories += [root / 'v15_refit']
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
        if any(record.get(key) != sha256(directory / name) for name, key in
               (('pred_results.csv', 'csv_sha256'), ('pred_results.zip', 'zip_sha256'))):
            continue
        if kind == 'validation':
            if sha256(checkpoint) == selected['source_checkpoint_sha256']:
                return {'directory': str(directory.resolve()), 'checkpoint': str(checkpoint.resolve())}
        elif not selected['refit_required']:
            value = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if value.get('training_stage') == 'refit' and _matches(value, selected):
                return {'directory': str(directory.resolve()), 'checkpoint': str(checkpoint.resolve())}
    return None


def _validate_official_report(report, recipe):
    required = ('frozen_backbone_unchanged', 'roundtrip_logits_exact',
                'all_attention_and_mlp_gradients_nonzero', 'missing_mlp_weight_rejected')
    spec = RESOURCE_SPECS.get(recipe)
    if (spec is None or report.get('format_version') != 18 or report.get('recipe') != recipe
            or report.get('success') is not True or report.get('lora_modules') != 72
            or report.get('visual_trainable_parameters') != spec['visual_trainable_parameters']
            or report.get('image_size') != spec['image_size']
            or report.get('lora_rank') != spec['lora_rank']
            or report.get('optimizer_steps') != 2
            or any(report.get(key) is not True for key in required)):
        raise ValueError('official check is not a successful V18 backbone check for this recipe')


def official_check(path, recipe, model_dir):
    from v18_core import model_identity
    report = json.loads(Path(path).read_text())
    _validate_official_report(report, recipe)
    if report.get('base_model_identity') != model_identity(model_dir):
        raise ValueError('official check base model identity changed')
    return report


def benchmark_needed(path, recipe):
    path = Path(path)
    if not path.exists():
        return True
    report = json.loads(path.read_text())
    if report.get('format_version') == 18 and report.get('recipe') == recipe and report.get('success') is False:
        return True
    benchmark_settings(path, recipe)
    return False


def benchmark_settings(path, recipe):
    report = json.loads(Path(path).read_text())
    spec = RESOURCE_SPECS.get(recipe)
    if (spec is None or report.get('format_version') != 18 or report.get('recipe') != recipe
            or report.get('success') is not True or report.get('initialization') != 'official_base_only'
            or report.get('trained_checkpoint_weights_loaded') is not False
            or report.get('pin_memory') is not False or report.get('prefetch_factor') != 1
            or report.get('chosen_workers') != 8
            or report.get('image_size') != spec['image_size']
            or report.get('lora_rank') != spec['lora_rank']
            or report.get('activation_checkpointing', False) is not False):
        raise ValueError('benchmark is not a successful official-base V18 probe for this recipe')
    batch = report.get('batch_size')
    accumulation = report.get('gradient_accumulation')
    if batch not in (128, 256) or accumulation not in (1, 2) or accumulation * batch != 256:
        raise ValueError('benchmark must preserve effective batch 256')
    if recipe == 'rank32' and batch != 256:
        raise ValueError('rank32 requires microbatch 256; a failed capacity experiment cannot become fallback')
    if recipe == 'matched_control' and batch != 128:
        raise ValueError('matched_control requires microbatch 128 and accumulation 2')
    if recipe == 'resolution384' and batch == 128 and not any(row.get('batch_size') == 256 and row.get('status') == 'out_of_memory'
                                for row in report.get('configurations', [])):
        raise ValueError('128 microbatch requires a recorded actual 256 CUDA OOM')
    return report['chosen_workers'], batch


def resource_for_recipe(resource, recipe):
    matches = [entry for entry in resource['candidates'].values() if entry['recipe'] == recipe]
    if len(matches) != 1:
        raise ValueError(f'recipe {recipe} is not active in the frozen V18 resource branch')
    return matches[0]


def read_resource_plan(path):
    """Validate the immutable experiment branch and every engineering source it consumed."""
    resource = json.loads(Path(path).read_text())
    branch = resource.get('branch')
    if (resource.get('format_version') != 18 or resource.get('kind') != 'v18_resource_plan'
            or branch not in ('standard', 'matched_control')
            or resource.get('activation_checkpointing') is not False or resource.get('gates') != GATES):
        raise ValueError('invalid V18 resource branch or frozen evaluation gates')
    if not resource.get('dataset_signature') or not resource.get('base_model_identity'):
        raise ValueError('V18 resource plan requires dataset and official model identities')
    entries = resource.get('candidates', {})
    if set(entries) != {'candidate_a', 'candidate_b'}:
        raise ValueError('V18 resource plan requires exactly two candidate entries')
    recipes = ('resolution384', 'rank32' if branch == 'standard' else 'matched_control')
    batch = 256 if branch == 'standard' else 128
    for key, recipe in zip(('candidate_a', 'candidate_b'), recipes):
        entry = entries[key]
        spec = RESOURCE_SPECS[recipe]
        if (entry.get('recipe') != recipe or entry.get('batch_size') != batch
                or entry.get('gradient_accumulation') != 256 // batch
                or entry.get('workers') != 8
                or any(entry.get(field) != spec[field] for field in ('image_size', 'lora_rank'))):
            raise ValueError(f'V18 {key} differs from frozen branch recipe or microbatch')
        for source_key in ('benchmark', 'official_check'):
            source = entry[source_key]
            if not Path(source['path']).is_file() or sha256(source['path']) != source['sha256']:
                raise ValueError(f'V18 frozen {source_key} source changed')
        workers, probed_batch = benchmark_settings(entry['benchmark']['path'], recipe)
        if (workers, probed_batch) != (entry['workers'], entry['batch_size']):
            raise ValueError('V18 resource branch differs from benchmark')
        for source_key in ('benchmark', 'official_check'):
            report = json.loads(Path(entry[source_key]['path']).read_text())
            if source_key == 'official_check':
                _validate_official_report(report, recipe)
            if report.get('base_model_identity') != resource.get('base_model_identity'):
                raise ValueError('V18 resource probes use different official base weights')
        report = json.loads(Path(entry['benchmark']['path']).read_text())
        if report.get('dataset_signature') != resource.get('dataset_signature'):
            raise ValueError('V18 resource probes use different datasets')
    if entries['candidate_a']['run_dir'] == entries['candidate_b']['run_dir']:
        raise ValueError('V18 candidates require independent output directories')
    baseline = resource['baseline']
    for name, key in (('model.pt', 'model_sha256'), ('strict_eval.json', 'evaluation_sha256')):
        source = Path(baseline['directory']) / name
        if not source.is_file() or sha256(source) != baseline[key]:
            raise ValueError('V18 resource baseline changed after freezing')
    return resource


def freeze_resources(root, benchmark_a, benchmark_b, official_a, official_b, candidate_a,
                     candidate_b, baseline, model_dir, output=None):
    """Freeze before either candidate starts; inference/accuracy never chooses the branch."""
    output = Path(output or os.environ.get('AIC_RESOURCE_JSON', str(pipeline_dir(root) / 'resource_plan.json')))
    _, batch = benchmark_settings(benchmark_a, 'resolution384')
    branch = 'standard' if batch == 256 else 'matched_control'
    recipes = ('resolution384', 'rank32' if branch == 'standard' else 'matched_control')
    entries = {}
    for key, recipe, benchmark, check, run in zip(('candidate_a', 'candidate_b'), recipes,
            (benchmark_a, benchmark_b), (official_a, official_b), (candidate_a, candidate_b)):
        workers, actual_batch = benchmark_settings(benchmark, recipe)
        if actual_batch != batch:
            raise ValueError('both V18 candidates must use the same frozen microbatch')
        official_check(check, recipe, model_dir)
        entries[key] = {'recipe': recipe, 'run_dir': str(Path(run).resolve()),
                        'batch_size': batch, 'gradient_accumulation': 256 // batch, 'workers': workers,
                        'image_size': RESOURCE_SPECS[recipe]['image_size'],
                        'lora_rank': RESOURCE_SPECS[recipe]['lora_rank'],
                        'benchmark': {'path': str(Path(benchmark).resolve()), 'sha256': sha256(benchmark)},
                        'official_check': {'path': str(Path(check).resolve()), 'sha256': sha256(check)}}
    report = json.loads(Path(benchmark_a).read_text())
    result = {'format_version': 18, 'kind': 'v18_resource_plan', 'branch': branch,
              'activation_checkpointing': False, 'candidates': entries, 'gates': GATES,
              'base_model_identity': report['base_model_identity'], 'dataset_signature': report['dataset_signature'],
              'baseline': {'directory': str(Path(baseline).resolve()),
                           'model_sha256': sha256(Path(baseline) / 'model.pt'),
                           'evaluation_sha256': sha256(Path(baseline) / 'strict_eval.json')}}
    if output.exists():
        if read_resource_plan(output) != result:
            raise ValueError('V18 resource branch already frozen with different inputs')
        return result
    # A missing freeze record cannot be reconstructed after training has begun.
    for run in (candidate_a, candidate_b):
        if any(Path(run).glob('*.pt')):
            raise ValueError('V18 resource branch must be frozen before either training run starts')
    atomic_json(output, result)
    try:
        read_resource_plan(output)
    except Exception:
        output.unlink()
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'state', 'wait-gpu', 'wait-baselines', 'accepted',
                         'recipe', 'checkpoint', 'export', 'reusable', 'bind', 'benchmark-settings',
                         'benchmark-needed', 'official-check', 'freeze-resources', 'resource-check',
                         'resource-branch', 'resource-field', 'resource-settings', 'second-recipe'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--state-file', default='status.json')
    parser.add_argument('--stage', default='starting'); parser.add_argument('--status', default='running')
    parser.add_argument('--benchmark', type=Path); parser.add_argument('--recipe')
    parser.add_argument('--report', type=Path); parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--resource-json', type=Path)
    parser.add_argument('--benchmark-a', type=Path); parser.add_argument('--benchmark-b', type=Path)
    parser.add_argument('--official-a', type=Path); parser.add_argument('--official-b', type=Path)
    parser.add_argument('--candidate-a', type=Path); parser.add_argument('--candidate-b', type=Path)
    parser.add_argument('--candidate', choices=('candidate_a', 'candidate_b'))
    parser.add_argument('--field', choices=('recipe', 'run_dir', 'benchmark', 'official_check', 'batch_size'))
    parser.add_argument('--detail'); parser.add_argument('--gpu')
    parser.add_argument('--minimum', type=int, default=32768)
    parser.add_argument('--timeout', type=float, default=48*3600)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--baseline-v15', type=Path)
    parser.add_argument('--kind', choices=('validation', 'refit', 'fallback_refit'))
    parser.add_argument('--checkpoint'); parser.add_argument('--source', type=Path)
    parser.add_argument('--test-dir', type=Path); parser.add_argument('--expected-rows', type=int, default=37444)
    args = parser.parse_args()
    resource_path = args.resource_json or Path(os.environ.get('AIC_RESOURCE_JSON', str(pipeline_dir(args.root) / 'resource_plan.json')))
    if args.action == 'freeze-resources':
        freeze_resources(args.root, args.benchmark_a, args.benchmark_b, args.official_a, args.official_b,
                         args.candidate_a, args.candidate_b, args.baseline_v15, args.model_dir, resource_path)
    elif args.action.startswith('resource-'):
        resource = read_resource_plan(resource_path)
        if args.action == 'resource-branch': print(resource['branch'])
        elif args.action == 'resource-settings':
            entry = resource_for_recipe(resource, args.recipe)
            print(entry['workers'], entry['batch_size'])
        elif args.action == 'resource-field':
            entry = resource['candidates'][args.candidate] if args.candidate else resource_for_recipe(resource, args.recipe)
            value = entry[args.field]
            print(value['path'] if isinstance(value, dict) else value)
    elif args.action == 'second-recipe':
        _, batch = benchmark_settings(args.benchmark, 'resolution384')
        print('rank32' if batch == 256 else 'matched_control')
    elif args.action == 'official-check': official_check(args.report, args.recipe, args.model_dir)
    elif args.action == 'benchmark-needed': print(int(benchmark_needed(args.benchmark, args.recipe)))
    elif args.action == 'benchmark-settings': print(*benchmark_settings(args.benchmark, args.recipe))
    elif args.action == 'init': print(initialize(args.root)['started_at_unix'])
    elif args.action == 'state': state(args.root, args.stage, args.status, args.state_file, args.detail)
    elif args.action == 'wait-gpu': wait_gpu(args.root, args.gpu, args.minimum, args.timeout, args.interval, args.state_file)
    elif args.action == 'wait-baselines':
        wait_baselines(args.root, [args.baseline_v15 or args.root / 'v15_expanded_mlp'], args.timeout, args.interval)
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
