"""V19 orchestration contracts, GPU queueing, wall-clock budget and delivery provenance."""
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
    'resolution384_rank32': {'image_size': 384, 'zoom_shortest_edge': 439, 'lora_alpha': 64.0, 'lora_rank': 32, 'lora_dropout': 0.0,
                           'mlp_lora_dropout': 0.0, 'visual_trainable_parameters': 5308416},
    'rank32_dropout': {'image_size': 320, 'zoom_shortest_edge': 366, 'lora_alpha': 64.0, 'lora_rank': 32, 'lora_dropout': 0.05,
                      'mlp_lora_dropout': 0.05, 'visual_trainable_parameters': 5308416},
}
CONTROL_SPEC = {'image_size': 320, 'zoom_shortest_edge': 366, 'lora_alpha': 64.0,
                'lora_rank': 32, 'lora_dropout': 0.0, 'mlp_lora_dropout': 0.0,
                'visual_trainable_parameters': 5308416}
ALL_SPECS = {**RESOURCE_SPECS, 'rank32_control': CONTROL_SPEC}

GATES = {'macro_gain': 0.002, 'tail_tolerance': 0.003, 'overall_noninferiority': True}


def now():
    return datetime.datetime.now().astimezone().isoformat()


def pipeline_dir(root):
    return Path(os.environ.get('AIC_PIPELINE_DIR', str(root / 'v19_pipeline')))


def initialize(root):
    path = pipeline_dir(root) / 'budget.json'
    if not path.exists():
        started = float(os.environ.get('AIC_PIPELINE_STARTED_AT', time.time()))
        if not math.isfinite(started) or started < 0 or started > time.time():
            raise ValueError('AIC_PIPELINE_STARTED_AT must be a finite past Unix timestamp')
        atomic_json(path, {'format_version': 19,
                           'started_at': datetime.datetime.fromtimestamp(started).astimezone().isoformat(),
                           'started_at_unix': started,
                           'budget_hours': 48, 'includes': ['preparation', 'gpu_queue', 'validation',
                           'evaluation', 'baseline_wait', 'selection', 'refit', 'prediction'],
                           'automatic_short_training': False})
    result = json.loads(path.read_text())
    if result.get('format_version') != 19:
        raise ValueError('V19 requires a fresh V19 budget record')
    return result


def state(root, stage, status, filename='status.json', detail=None):
    budget = initialize(root)
    elapsed = max(0, time.time() - budget['started_at_unix'])
    result = {'format_version': 19, 'stage': stage, 'status': status, 'detail': detail,
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
        print(f'v19_gpu_queue gpu={gpu} free_mib={free} required_mib={minimum}', flush=True)
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
                raise ValueError('baseline evaluation changed after V19 consumed it')
            if not binding.exists():
                atomic_json(binding, identity)
            return
        # A dependency may already have the required validation products even if a later refit failed.
        for version in ('v18',):
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
        print(f'v19_wait missing={missing}', flush=True)
        time.sleep(min(interval, max(0, timeout - (time.monotonic() - start))))


def decision(root, selection=None):
    from select_v19 import read_decision
    path = Path(selection or os.environ.get('AIC_SELECTION', str(root / 'v19_selection.json')))
    result = read_decision(path)
    resource_path = Path(os.environ.get('AIC_RESOURCE_JSON', result.get('resource_json',
                         str(pipeline_dir(root) / 'resource_plan.json'))))
    resource = read_resource_plan(resource_path)
    if (result.get('resource_sha256') != sha256(resource_path)
            or result.get('resource_branch') != resource['branch']):
        raise ValueError('V19 selection differs from frozen resource branch')
    identity = {'selection': str(path.resolve()), 'sha256': sha256(path)}
    binding = pipeline_dir(root) / 'selection_binding.json'
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError('V19 selection changed after being consumed')
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
    if checkpoint.get('format_version') in (18, 19):
        config = checkpoint.get('config', {})
        chosen = selected.get('training_config', {})
        if any(config.get(key) != chosen.get(key) for key in
               ('batch_size', 'gradient_accumulation', 'workers', 'image_size', 'zoom_shortest_edge',
                'lora_rank', 'lora_alpha', 'mlp_lora_rank', 'mlp_lora_alpha', 'lora_dropout', 'mlp_lora_dropout', 'schedule_epochs',
                'precision', 'resource_sha256', 'resource_branch', 'audit_json', 'audit_sha256')):
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
        raise ValueError('only an accepted V19 winner can use the V19 refit checkpoint')
    result = Path(os.environ.get('AIC_REFIT_DIR', str(root / 'v19_refit'))) / 'model.pt'
    checkpoint = torch.load(result, map_location='cpu', weights_only=False)
    if (checkpoint.get('training_stage') != 'refit' or not _matches(checkpoint, selected)
            or checkpoint.get('refit_selection_sha256') != sha256(path)):
        raise ValueError('V19 refit checkpoint differs from frozen selection')
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
    out = Path(os.environ.get('AIC_DELIVERY_DIR', str(root / 'v19_delivery'))) / kind
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
    result = {'format_version': 19, 'kind': kind, 'checkpoint': str(checkpoint_value),
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
                raise ValueError(f'completed V19 delivery changed: {key}')
        return previous
    atomic_json(existing, result)
    return result


def export_historical_fallback(root, source, test_dir, expected_rows=37444):
    """Preserve the known online V18 full refit if no clean V19 candidate wins."""
    selected, selection_path = decision(root)
    resource = read_resource_plan(selected['resource_json'])
    if (selected['refit_required'] or resource['branch'] != 'deduplicated_control'):
        raise ValueError('historical fallback requires a rejected deduplicated V19 experiment')
    frozen = resource['historical_refit']
    source = Path(source).resolve()
    if source != Path(frozen['directory']).resolve():
        raise ValueError('historical full refit source differs from frozen resource plan')
    provenance = json.loads((source / 'provenance.json').read_text())
    checkpoint = Path(provenance['checkpoint'])
    for path, key in ((source / 'provenance.json', 'provenance_sha256'),
                      (checkpoint, 'checkpoint_sha256'),
                      (source / 'pred_results.csv', 'csv_sha256'),
                      (source / 'pred_results.zip', 'zip_sha256')):
        if sha256(path) != frozen[key]:
            raise ValueError(f'historical V18 package changed: {key}')
    _, rows = verify_package(source, checkpoint, test_dir, expected_rows)
    out = Path(os.environ.get('AIC_DELIVERY_DIR', str(root / 'v19_delivery'))) / 'fallback_refit'
    out.mkdir(parents=True, exist_ok=True)
    result = {'format_version': 19, 'kind': 'historical_v18_fallback_refit',
              'source_format_version': 18, 'source': str(source),
              'selection_sha256': sha256(selection_path), 'resource_sha256': sha256(selected['resource_json']),
              'checkpoint': str(checkpoint), 'checkpoint_sha256': frozen['checkpoint_sha256'],
              'csv_sha256': frozen['csv_sha256'], 'zip_sha256': frozen['zip_sha256'],
              'linecount': rows, 'online_score': 69.31951714560411,
              'created_at': now()}
    existing = out / 'provenance.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        if any(previous.get(key) != result[key] for key in
               ('selection_sha256','resource_sha256','checkpoint_sha256','csv_sha256','zip_sha256')):
            raise ValueError('completed historical fallback changed')
        return previous
    for name in ('pred_results.csv','pred_results.zip'):
        temp = out / f'.{name}.{os.getpid()}.tmp'
        shutil.copyfile(source / name, temp)
        temp.replace(out / name)
    verify_package(out, checkpoint, test_dir, expected_rows)
    atomic_json(existing, result)
    return result


def reusable_package(root, kind):
    """Return only previously proven packages; absence is normal and never starts training."""
    import torch
    selected, _ = decision(root)
    if kind not in ('validation', 'refit'):
        raise ValueError('reusable package kind must be validation or refit')
    directories = [root / 'v18_delivery' / name for name in ('validation', 'refit')]
    directories += [root / 'v18_refit']
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
    spec = ALL_SPECS.get(recipe)
    if (spec is None or report.get('format_version') != 19 or report.get('recipe') != recipe
            or report.get('success') is not True or report.get('lora_modules') != 72
            or report.get('visual_trainable_parameters') != spec['visual_trainable_parameters']
            or report.get('image_size') != spec['image_size']
            or any(report.get(k) != spec[k] for k in ('zoom_shortest_edge','lora_rank','lora_alpha','lora_dropout','mlp_lora_dropout'))
            or report.get('optimizer_steps') != 2
            or any(report.get(key) is not True for key in required)):
        raise ValueError('official check is not a successful V19 backbone check for this recipe')


def official_check(path, recipe, model_dir):
    from v19_core import model_identity
    report = json.loads(Path(path).read_text())
    _validate_official_report(report, recipe)
    if report.get('base_model_identity') != model_identity(model_dir):
        raise ValueError('official check base model identity changed')
    return report


def benchmark_status(path, recipe):
    report = json.loads(Path(path).read_text())
    spec = ALL_SPECS.get(recipe)
    if (spec is None or report.get('format_version') != 19 or report.get('recipe') != recipe
            or report.get('initialization') != 'official_base_only'
            or report.get('trained_checkpoint_weights_loaded') is not False
            or report.get('batch_size') != 256 or report.get('gradient_accumulation') != 1
            or any(report.get(k) != spec[k] for k in ('image_size','zoom_shortest_edge','lora_rank','lora_alpha','lora_dropout','mlp_lora_dropout'))):
        raise ValueError('benchmark identity or frozen V19 recipe mismatch')
    if report.get('success') is False:
        if (report.get('status') != 'cuda_oom' or not any(
                row.get('batch_size') == 256 and row.get('status') in ('out_of_memory','cuda_oom')
                for row in report.get('configurations', []))):
            raise ValueError('failed benchmark is not a recorded actual CUDA OOM')
        return 'resource_infeasible'
    if (report.get('success') is not True or report.get('chosen_workers') != 8
            or report.get('pin_memory') is not False or report.get('prefetch_factor') != 1
            or report.get('activation_checkpointing', False) is not False):
        raise ValueError('benchmark is not a successful official-base V19 probe')
    return 'runnable'


def benchmark_needed(path, recipe):
    if not Path(path).exists():
        return True
    benchmark_status(path, recipe)  # A failure is never silently retried.
    return False


def benchmark_settings(path, recipe):
    if benchmark_status(path, recipe) != 'runnable':
        raise ValueError('resource-infeasible candidate cannot train')
    return 8, 256


def resource_for_recipe(resource, recipe):
    matches = [e for e in resource['candidates'].values() if e['recipe'] == recipe]
    if resource.get('control') is not None and resource['control']['recipe'] == recipe:
        matches.append(resource['control'])
    if len(matches) != 1:
        raise ValueError(f'recipe {recipe} is not in the frozen V19 plan')
    return matches[0]


def validate_audit(path, dataset_signature=None):
    audit = json.loads(Path(path).read_text())
    if (audit.get('format_version') != 19 or audit.get('kind') != 'v19_generalization_audit'
            or audit.get('status') != 'passed' or type(audit.get('decoded_cross_split_groups')) is not int
            or type(audit.get('decoded_cross_split_pairs')) is not int or audit.get('decoded_cross_split_groups') != 0
            or audit.get('decoded_cross_split_pairs') != 0):
        raise ValueError('V19 data audit failed: decoded train/validation overlap or incomplete audit')
    if dataset_signature is not None and audit.get('dataset_signature') != dataset_signature:
        raise ValueError('audit dataset differs from frozen resource dataset')
    for name in ('manifest','feature_cache','feature_metadata'):
        record = audit['sources'][name]
        source = Path(record['path'])
        if not source.is_file():
            raise ValueError(f'audit source missing: {name}')
        stat = source.stat()
        if stat.st_size != record['size'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError(f'audit source changed: {name}')
        # The multi-view cache is large and immutable; its full SHA was computed during audit.
        if name != 'feature_cache' and sha256(source) != record['sha256']:
            raise ValueError(f'audit source hash changed: {name}')
    for name in ('decoded_overlap','decoded_images','validation_neighbors'):
        record = audit['artifacts'][name]
        if not Path(record['path']).is_file() or sha256(record['path']) != record['sha256']:
            raise ValueError(f'audit artifact changed: {name}')
    return audit


def read_resource_plan(path):
    resource = json.loads(Path(path).read_text())
    if (resource.get('format_version') != 19 or resource.get('kind') != 'v19_resource_plan'
            or resource.get('branch') not in ('standard', 'deduplicated_control')
            or resource.get('activation_checkpointing') is not False
            or resource.get('gates') != GATES):
        raise ValueError('invalid V19 resource plan or frozen gates')
    if not resource.get('dataset_signature') or not resource.get('base_model_identity'):
        raise ValueError('resource plan requires dataset and official model identities')
    audit_binding = resource['audit']
    if sha256(audit_binding['path']) != audit_binding['sha256']:
        raise ValueError('frozen audit report changed')
    audit = validate_audit(audit_binding['path'], resource['dataset_signature'])
    if audit['base_model_identity'] != resource['base_model_identity']:
        raise ValueError('audit and probes official models differ')
    entries = resource.get('candidates', {})
    if set(entries) != {'candidate_a','candidate_b'}:
        raise ValueError('two explicit candidate statuses are required')
    check_entries = [(key, recipe, entries[key]) for key, recipe in
                     zip(('candidate_a','candidate_b'), RESOURCE_SPECS)]
    if resource['branch'] == 'deduplicated_control':
        source = resource.get('source_manifest', {})
        if not Path(source.get('path', '')).is_file() or sha256(source['path']) != source.get('sha256'):
            raise ValueError('deduplicated V19 source manifest missing or changed')
        snapshot = json.loads(Path(source['path']).read_text())
        snapshot_root = Path(snapshot.get('remote_snapshot', ''))
        if (snapshot_root != Path(source['path']).parent
                or any(not (snapshot_root / name).is_file() or sha256(snapshot_root / name) != digest
                       for name, digest in snapshot.get('sha256', {}).items())
                or not snapshot.get('sha256')):
            raise ValueError('deduplicated V19 source snapshot changed')
        control = resource.get('control')
        if not isinstance(control, dict):
            raise ValueError('deduplicated V19 plan requires a matched control')
        dedup = resource.get('dedup', {})
        if (not Path(dedup.get('path', '')).is_file()
                or sha256(dedup['path']) != dedup.get('sha256')):
            raise ValueError('deduplicated V19 input derivation missing or changed')
        derivation = json.loads(Path(dedup['path']).read_text())
        if (derivation.get('kind') != 'v19_deduplicated_split'
                or derivation.get('excluded_train_count') != 12
                or derivation.get('dataset_signature') != resource['dataset_signature']
                or derivation.get('derived_manifest_sha256') != sha256(derivation['derived_manifest'])
                or derivation.get('derived_feature_metadata_sha256') != sha256(str(derivation['derived_feature_cache'])+'.json')
                or derivation.get('feature_cache_sha256') != audit['sources']['feature_cache']['sha256']):
            raise ValueError('deduplicated V19 input provenance differs from audit')
        historical = resource.get('historical_refit', {})
        source_dir = Path(historical.get('directory', ''))
        if (not source_dir.is_dir() or not (source_dir / 'provenance.json').is_file()
                or sha256(source_dir / 'provenance.json') != historical.get('provenance_sha256')):
            raise ValueError('historical V18 full-refit provenance changed')
        historical_provenance = json.loads((source_dir / 'provenance.json').read_text())
        for key, source in (('checkpoint_sha256', Path(historical_provenance['checkpoint'])),
                            ('csv_sha256', source_dir / 'pred_results.csv'),
                            ('zip_sha256', source_dir / 'pred_results.zip')):
            if sha256(source) != historical.get(key) or historical_provenance.get(key) != historical[key]:
                raise ValueError(f'historical V18 full-refit {key} changed')
        check_entries.append(('control', 'rank32_control', control))
    elif resource.get('control') is not None:
        raise ValueError('standard V19 plan must not contain a control run')
    for key, recipe, entry in check_entries:
        spec = ALL_SPECS[recipe]
        if (entry.get('recipe') != recipe or entry.get('batch_size') != 256
                or entry.get('gradient_accumulation') != 1 or entry.get('workers') != 8
                or any(entry.get(k) != spec[k] for k in ('image_size','zoom_shortest_edge','lora_rank','lora_alpha','lora_dropout','mlp_lora_dropout'))):
            raise ValueError(f'{key} differs from its frozen recipe')
        for kind in ('benchmark','official_check'):
            source = entry[kind]
            if not Path(source['path']).is_file() or sha256(source['path']) != source['sha256']:
                raise ValueError(f'frozen {kind} source changed')
            report = json.loads(Path(source['path']).read_text())
            if kind == 'official_check':
                _validate_official_report(report, recipe)
            if report.get('base_model_identity') != resource['base_model_identity']:
                raise ValueError('probe official base differs from frozen resource')
        status = benchmark_status(entry['benchmark']['path'], recipe)
        if entry.get('status') != status:
            raise ValueError('candidate resource status differs from actual benchmark')
        report = json.loads(Path(entry['benchmark']['path']).read_text())
        if report.get('dataset_signature') != resource['dataset_signature']:
            raise ValueError('probe dataset differs from resource')
    if len({entry['run_dir'] for _, _, entry in check_entries}) != len(check_entries):
        raise ValueError('V19 run directories must be independent')
    baseline = resource['baseline']
    for name, key in (('model.pt','model_sha256'),('strict_eval.json','evaluation_sha256')):
        source = Path(baseline['directory']) / name
        if not source.is_file() or sha256(source) != baseline[key]:
            raise ValueError('frozen V18 baseline changed')
    return resource


def freeze_resources(root, benchmark_a, benchmark_b, official_a, official_b, candidate_a,
                     candidate_b, baseline, model_dir, output=None, audit_path=None,
                     *, benchmark_control=None, official_control=None, control_run=None,
                     branch='standard', dedup_record=None, source_manifest=None):
    output = Path(output or os.environ.get('AIC_RESOURCE_JSON', str(pipeline_dir(root)/'resource_plan.json')))
    audit_path = Path(audit_path or os.environ.get('AIC_AUDIT_JSON',str(root/'v19_startup/audit/audit.json')))
    audit = validate_audit(audit_path)
    entries = {}
    for key, recipe, benchmark, check, run in zip(('candidate_a','candidate_b'), RESOURCE_SPECS,
            (benchmark_a,benchmark_b),(official_a,official_b),(candidate_a,candidate_b)):
        status = benchmark_status(benchmark, recipe)
        official_check(check, recipe, model_dir)
        entries[key] = {'recipe':recipe, 'run_dir':str(Path(run).resolve()), 'status':status,
            'batch_size':256,'gradient_accumulation':1,'workers':8,
            **{k:RESOURCE_SPECS[recipe][k] for k in ('image_size','zoom_shortest_edge','lora_rank','lora_alpha','lora_dropout','mlp_lora_dropout')},
            'benchmark':{'path':str(Path(benchmark).resolve()),'sha256':sha256(benchmark)},
            'official_check':{'path':str(Path(check).resolve()),'sha256':sha256(check)}}
    report = json.loads(Path(benchmark_a).read_text())
    if branch not in ('standard', 'deduplicated_control'):
        raise ValueError('unknown V19 resource branch')
    if branch == 'deduplicated_control':
        if any(value is None for value in (benchmark_control, official_control, control_run)):
            raise ValueError('deduplicated branch requires the matched control probe and run')
        recipe = 'rank32_control'
        status = benchmark_status(benchmark_control, recipe)
        official_check(official_control, recipe, model_dir)
        control_report = json.loads(Path(benchmark_control).read_text())
        if (control_report.get('dataset_signature') != report['dataset_signature']
                or control_report.get('base_model_identity') != report['base_model_identity']):
            raise ValueError('control benchmark differs from candidate inputs')
        spec = CONTROL_SPEC
        control = {'recipe':recipe,'run_dir':str(Path(control_run).resolve()),'status':status,
            'batch_size':256,'gradient_accumulation':1,'workers':8,
            **{k:spec[k] for k in ('image_size','zoom_shortest_edge','lora_rank','lora_alpha','lora_dropout','mlp_lora_dropout')},
            'benchmark':{'path':str(Path(benchmark_control).resolve()),'sha256':sha256(benchmark_control)},
            'official_check':{'path':str(Path(official_control).resolve()),'sha256':sha256(official_control)}}
        if status != 'runnable':
            raise ValueError('matched control cannot be resource-infeasible')
    else:
        control = None
    result = {'format_version':19,'kind':'v19_resource_plan','branch':branch,
        'activation_checkpointing':False,'candidates':entries,'gates':GATES,
        'base_model_identity':report['base_model_identity'],'dataset_signature':report['dataset_signature'],
        'audit':{'path':str(audit_path.resolve()),'sha256':sha256(audit_path)},
        'baseline':{'directory':str(Path(baseline).resolve()),'model_sha256':sha256(Path(baseline)/'model.pt'),
                    'evaluation_sha256':sha256(Path(baseline)/'strict_eval.json')}}
    if control is not None:
        result['control'] = control
        if dedup_record is None:
            raise ValueError('deduplicated branch requires a source derivation record')
        result['dedup'] = {'path':str(Path(dedup_record).resolve()),'sha256':sha256(dedup_record)}
        if source_manifest is None:
            raise ValueError('deduplicated branch requires a frozen source manifest')
        result['source_manifest'] = {'path':str(Path(source_manifest).resolve()),
                                     'sha256':sha256(source_manifest)}
        historical_dir = Path(root) / 'v18_delivery' / 'refit'
        historical_provenance = json.loads((historical_dir / 'provenance.json').read_text())
        result['historical_refit'] = {
            'directory': str(historical_dir.resolve()),
            'provenance_sha256': sha256(historical_dir / 'provenance.json'),
            'checkpoint_sha256': sha256(historical_provenance['checkpoint']),
            'csv_sha256': sha256(historical_dir / 'pred_results.csv'),
            'zip_sha256': sha256(historical_dir / 'pred_results.zip')}
    if output.exists():
        if read_resource_plan(output) != result:
            raise ValueError('V19 resource plan already frozen with different inputs')
        return result
    for run in (candidate_a,candidate_b,control_run):
        if run is None:
            continue
        if any(Path(run).glob('*.pt')):
            raise ValueError('resource plan must be frozen before training')
    atomic_json(output,result)
    try:
        read_resource_plan(output)
    except Exception:
        output.unlink()
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'state', 'wait-gpu', 'wait-baselines', 'accepted',
                         'recipe', 'checkpoint', 'export', 'export-historical', 'reusable', 'bind', 'benchmark-settings',
                         'benchmark-needed', 'official-check', 'freeze-resources', 'resource-check',
                         'resource-branch', 'resource-field', 'resource-settings', 'benchmark-status', 'audit-check'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--state-file', default='status.json')
    parser.add_argument('--stage', default='starting'); parser.add_argument('--status', default='running')
    parser.add_argument('--benchmark', type=Path); parser.add_argument('--recipe')
    parser.add_argument('--report', type=Path); parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--resource-json', type=Path)
    parser.add_argument('--audit-json', type=Path)
    parser.add_argument('--benchmark-a', type=Path); parser.add_argument('--benchmark-b', type=Path)
    parser.add_argument('--benchmark-control', type=Path)
    parser.add_argument('--official-a', type=Path); parser.add_argument('--official-b', type=Path)
    parser.add_argument('--official-control', type=Path)
    parser.add_argument('--candidate-a', type=Path); parser.add_argument('--candidate-b', type=Path)
    parser.add_argument('--control-run', type=Path)
    parser.add_argument('--dedup-record', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--branch', choices=('standard', 'deduplicated_control'), default='standard')
    parser.add_argument('--candidate', choices=('candidate_a', 'candidate_b', 'control'))
    parser.add_argument('--field', choices=('recipe', 'run_dir', 'benchmark', 'official_check', 'batch_size', 'status'))
    parser.add_argument('--detail'); parser.add_argument('--gpu')
    parser.add_argument('--minimum', type=int, default=32768)
    parser.add_argument('--timeout', type=float, default=48*3600)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--baseline-v18', type=Path)
    parser.add_argument('--kind', choices=('validation', 'refit', 'fallback_refit'))
    parser.add_argument('--checkpoint'); parser.add_argument('--source', type=Path)
    parser.add_argument('--test-dir', type=Path); parser.add_argument('--expected-rows', type=int, default=37444)
    args = parser.parse_args()
    resource_path = args.resource_json or Path(os.environ.get('AIC_RESOURCE_JSON', str(pipeline_dir(args.root) / 'resource_plan.json')))
    if args.action == 'freeze-resources':
        freeze_resources(args.root, args.benchmark_a, args.benchmark_b, args.official_a, args.official_b,
                         args.candidate_a, args.candidate_b, args.baseline_v18, args.model_dir, resource_path, args.audit_json,
                         benchmark_control=args.benchmark_control, official_control=args.official_control,
                         control_run=args.control_run, branch=args.branch,
                         dedup_record=args.dedup_record, source_manifest=args.source_manifest)
    elif args.action.startswith('resource-'):
        resource = read_resource_plan(resource_path)
        if args.action == 'resource-branch': print(resource['branch'])
        elif args.action == 'resource-settings':
            entry = resource_for_recipe(resource, args.recipe)
            if entry['status'] != 'runnable': raise ValueError('candidate cannot train')
            print(entry['workers'], entry['batch_size'])
        elif args.action == 'resource-field':
            entry = (resource['control'] if args.candidate == 'control' else resource['candidates'][args.candidate]) if args.candidate else resource_for_recipe(resource, args.recipe)
            value = entry[args.field]
            print(value['path'] if isinstance(value, dict) else value)
    elif args.action == 'audit-check': validate_audit(args.audit_json)
    elif args.action == 'export-historical':
        export_historical_fallback(args.root, args.source, args.test_dir, args.expected_rows)
    elif args.action == 'benchmark-status': print(benchmark_status(args.benchmark, args.recipe))
    elif args.action == 'official-check': official_check(args.report, args.recipe, args.model_dir)
    elif args.action == 'benchmark-needed': print(int(benchmark_needed(args.benchmark, args.recipe)))
    elif args.action == 'benchmark-settings': print(*benchmark_settings(args.benchmark, args.recipe))
    elif args.action == 'init': print(initialize(args.root)['started_at_unix'])
    elif args.action == 'state': state(args.root, args.stage, args.status, args.state_file, args.detail)
    elif args.action == 'wait-gpu': wait_gpu(args.root, args.gpu, args.minimum, args.timeout, args.interval, args.state_file)
    elif args.action == 'wait-baselines':
        wait_baselines(args.root, [args.baseline_v18 or args.root / 'v18_rank32'], args.timeout, args.interval)
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
