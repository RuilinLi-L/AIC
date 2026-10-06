"""Immutable V20 preflight contracts and independent orchestration state."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import time

from v15_pipeline_support import atomic_json, sha256
from v19_pipeline_support import validate_audit, verify_package

GATES = {'macro_gain': .002, 'tail_tolerance': .003, 'overall_noninferiority': True}
RESOURCE_SPECS = {
    'dora_rank32': {'visual_trainable_parameters': 5391360},
    'loraplus_rank32': {'visual_trainable_parameters': 5308416},
}
for _spec in RESOURCE_SPECS.values():
    _spec.update(image_size=320, zoom_shortest_edge=366, lora_rank=32,
                 lora_alpha=64., lora_dropout=0., mlp_lora_dropout=0.)


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def pipeline_dir(root):
    return Path(os.environ.get('AIC_PIPELINE_DIR', str(Path(root) / 'v20_pipeline')))


def initialize(root):
    path = pipeline_dir(root) / 'budget.json'
    if not path.exists():
        started = float(os.environ.get('AIC_PIPELINE_STARTED_AT', time.time()))
        if not 0 <= started <= time.time():
            raise ValueError('invalid V20 budget start')
        atomic_json(path, {'format_version': 20, 'started_at_unix': started,
                          'started_at': datetime.datetime.fromtimestamp(started, datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                          'budget_hours': 48, 'automatic_short_training': False,
                          'includes': ['preparation', 'queue', 'validation', 'joint_evaluation', 'baseline_wait', 'refit', 'prediction']})
    result = json.loads(path.read_text())
    if result.get('format_version') != 20:
        raise ValueError('V20 requires its own budget record')
    return result


def state(root, stage, status, filename='status.json', detail=None):
    budget = initialize(root)
    elapsed = max(0., time.time() - budget['started_at_unix'])
    result = dict(format_version=20, stage=stage, status=status, detail=detail,
                  updated_at=now(), elapsed_wall_clock_seconds=elapsed,
                  budget_hours=48, budget_exceeded=elapsed > 48 * 3600,
                  automatic_short_training=False)
    atomic_json(pipeline_dir(root) / filename, result)
    return result


def _binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def _check_binding(binding):
    path = Path(binding['path'])
    if not path.is_file() or sha256(path) != binding['sha256']:
        raise ValueError(f'frozen V20 input changed: {path}')
    return path


def official_check(path, recipe, model_dir=None):
    report = json.loads(Path(path).read_text())
    spec = RESOURCE_SPECS[recipe]
    required = ('frozen_backbone_unchanged', 'roundtrip_logits_exact',
                'all_attention_and_mlp_gradients_nonzero', 'missing_mlp_weight_rejected')
    if (report.get('format_version') != 20 or report.get('recipe') != recipe
            or report.get('success') is not True or report.get('lora_modules') != 72
            or report.get('optimizer_steps') != 2
            or any(report.get(k) != v for k, v in spec.items())
            or any(report.get(k) is not True for k in required)):
        raise ValueError('not a successful complete V20 official-model check')
    if model_dir is not None:
        from v20_core import model_identity
        if report.get('base_model_identity') != model_identity(model_dir):
            raise ValueError('official weights differ from V20 check')
    if recipe == 'dora_rank32' and report.get('missing_magnitude_weight_rejected') is not True:
        raise ValueError('V20 DoRA check must reject a missing magnitude weight')
    return report


def benchmark_status(path, recipe):
    report = json.loads(Path(path).read_text())
    spec = RESOURCE_SPECS[recipe]
    if (report.get('format_version') != 20 or report.get('recipe') != recipe
            or report.get('initialization') != 'official_base_only'
            or report.get('trained_checkpoint_weights_loaded') is not False
            or report.get('batch_size') != 256 or report.get('gradient_accumulation') != 1
            or any(report.get(k) != spec[k] for k in spec if k != 'visual_trainable_parameters')):
        raise ValueError('V20 benchmark provenance differs')
    if report.get('success') is False:
        if report.get('status') != 'cuda_oom' or not any(
                row.get('batch_size') == 256 and row.get('status') in ('cuda_oom', 'out_of_memory')
                for row in report.get('configurations', [])):
            raise ValueError('failed V20 probe is not a recorded typed CUDA OOM')
        return 'resource_infeasible'
    if (report.get('success') is not True or report.get('chosen_workers') != 8
            or report.get('pin_memory') is not False or report.get('prefetch_factor') != 1
            or report.get('activation_checkpointing', False) is not False):
        raise ValueError('V20 probe did not validate the frozen runtime')
    return 'runnable'


def resource_for_recipe(resource, recipe):
    for entry in resource['candidates'].values():
        if entry['recipe'] == recipe:
            return entry
    raise ValueError(f'unknown V20 resource recipe: {recipe}')


def read_resource_plan(path):
    resource = json.loads(Path(path).read_text())
    if (resource.get('format_version') != 20 or resource.get('kind') != 'v20_resource_plan'
            or resource.get('branch') != 'deduplicated_control' or resource.get('gates') != GATES
            or resource.get('activation_checkpointing') is not False
            or resource.get('allowed_gpus') != [0, 1, 2, 3]
            or resource.get('minimum_free_mib') != 71680):
        raise ValueError('invalid frozen V20 resource plan')
    for field in ('audit', 'dedup', 'source_manifest', 'v19_resource'):
        _check_binding(resource[field])
    audit = validate_audit(resource['audit']['path'], resource['dataset_signature'])
    if audit['base_model_identity'] != resource['base_model_identity']:
        raise ValueError('V20 audit base differs')
    derivation = json.loads(Path(resource['dedup']['path']).read_text())
    if (derivation.get('dataset_signature') != resource['dataset_signature']
            or derivation.get('kind') != 'v19_deduplicated_split'
            or sha256(derivation['derived_manifest']) != derivation['derived_manifest_sha256']
            or sha256(str(derivation['derived_feature_cache']) + '.json') != derivation['derived_feature_metadata_sha256']):
        raise ValueError('V20 dedup provenance changed')
    snapshot = json.loads(Path(resource['source_manifest']['path']).read_text())
    source_root = Path(snapshot['remote_snapshot'])
    if source_root != Path(resource['source_manifest']['path']).parent or not snapshot.get('sha256'):
        raise ValueError('V20 source snapshot missing')
    for name, digest in snapshot['sha256'].items():
        if sha256(source_root / name) != digest:
            raise ValueError(f'V20 frozen source changed: {name}')
    entries = resource.get('candidates', {})
    if set(entries) != {'candidate_a', 'candidate_b'}:
        raise ValueError('V20 requires two explicit candidate statuses')
    for key, recipe in zip(('candidate_a', 'candidate_b'), RESOURCE_SPECS):
        entry = entries[key]
        if (entry.get('recipe') != recipe or entry.get('batch_size') != 256
                or entry.get('gradient_accumulation') != 1 or entry.get('workers') != 8
                or any(entry.get(k) != v for k, v in RESOURCE_SPECS[recipe].items())):
            raise ValueError('V20 resource entry differs from fixed recipe')
        for kind in ('benchmark', 'official_check'):
            _check_binding(entry[kind])
            report = json.loads(Path(entry[kind]['path']).read_text())
            if report.get('base_model_identity') != resource['base_model_identity']:
                raise ValueError('V20 probe official identity differs')
            if kind == 'benchmark' and report.get('dataset_signature') != resource['dataset_signature']:
                raise ValueError('V20 probe dataset identity differs')
        official_check(entry['official_check']['path'], recipe)
        if benchmark_status(entry['benchmark']['path'], recipe) != entry['status']:
            raise ValueError('V20 candidate feasibility changed')
    old = json.loads(Path(resource['v19_resource']['path']).read_text())
    if old.get('dataset_signature') != resource['dataset_signature'] or old.get('branch') != 'deduplicated_control':
        raise ValueError('V19 dependencies do not match V20 data')
    if resource['control']['run_dir'] != old['control']['run_dir']:
        raise ValueError('V20 must wait for the fixed V19 control')
    if (resource['control'] != {k: old['control'][k] for k in ('recipe', 'run_dir', 'status')}
            or set(resource['dependencies']) != {'resolution384_rank32', 'rank32_dropout'}
            or resource.get('historical_refit') != old.get('historical_refit')):
        raise ValueError('V20 historical dependencies changed')
    for recipe, entry in resource['dependencies'].items():
        expected = next(item for item in old['candidates'].values() if item['recipe'] == recipe)
        if entry != {k: expected[k] for k in ('recipe', 'run_dir', 'status')}:
            raise ValueError('V19 dependency silently changed')
    directories = [entry['run_dir'] for entry in entries.values()] + [resource['control']['run_dir']]
    if len(set(directories)) != len(directories):
        raise ValueError('V20 training directories collide')
    return resource


def freeze_resources(root, args):
    from v19_pipeline_support import read_resource_plan as read_v19
    old = read_v19(args.v19_resource)
    if old['branch'] != 'deduplicated_control':
        raise ValueError('V20 requires the matched deduplicated V19 experiment')
    entries = {}
    for key, recipe, run in zip(('candidate_a', 'candidate_b'), RESOURCE_SPECS, (args.candidate_a, args.candidate_b)):
        run = Path(run).resolve()
        check = official_check(run / 'official_check.json', recipe, args.model_dir)
        benchmark = json.loads((run / 'benchmark.json').read_text())
        if (check['base_model_identity'] != old['base_model_identity']
                or benchmark['dataset_signature'] != old['dataset_signature']):
            raise ValueError('V20 probes differ from V19 inputs')
        entries[key] = dict(recipe=recipe, run_dir=str(run), status=benchmark_status(run / 'benchmark.json', recipe),
                            batch_size=256, gradient_accumulation=1, workers=8, **RESOURCE_SPECS[recipe],
                            benchmark=_binding(run / 'benchmark.json'), official_check=_binding(run / 'official_check.json'))
    result = dict(format_version=20, kind='v20_resource_plan', branch='deduplicated_control',
                  activation_checkpointing=False, candidates=entries, gates=GATES,
                  allowed_gpus=[0, 1, 2, 3], minimum_free_mib=71680,
                  dataset_signature=old['dataset_signature'], base_model_identity=old['base_model_identity'],
                  audit=old['audit'], dedup=old['dedup'], source_manifest=_binding(args.source_manifest),
                  v19_resource=_binding(args.v19_resource),
                  control={k: old['control'][k] for k in ('recipe', 'run_dir', 'status')},
                  dependencies={entry['recipe']: {k: entry[k] for k in ('recipe', 'run_dir', 'status')}
                                for entry in old['candidates'].values()}, historical_refit=old['historical_refit'])
    output = Path(args.resource_json)
    if output.exists():
        if read_resource_plan(output) != result:
            raise ValueError('V20 resources already frozen differently')
    else:
        if any(any(Path(entry['run_dir']).glob('*.pt')) for entry in entries.values()):
            raise ValueError('V20 resource freeze must precede training')
        atomic_json(output, result)
        read_resource_plan(output)
    return result


def export_submission(root, kind, checkpoint_path, test_dir, expected_rows=37444):
    from select_v20 import read_decision
    selected_path = Path(os.environ['AIC_SELECTION'])
    selected = read_decision(selected_path)
    out = Path(os.environ['AIC_DELIVERY_DIR']) / kind
    checkpoint, rows = verify_package(out, checkpoint_path, test_dir, expected_rows)
    if kind in ('validation', 'refit'):
        if (selected['refit_required'] is not True
                or checkpoint.get('format_version') != selected['source_format_version']
                or checkpoint.get('training_stage') != ('validation' if kind == 'validation' else 'refit')
                or checkpoint['config']['recipe'] != selected['recipe']
                or checkpoint['selected_epoch'] != selected['selected_epoch']
                or checkpoint['dataset_signature'] != selected['dataset_signature']
                or checkpoint.get('class_names') != selected['class_names']
                or checkpoint['config'].get('base_model_identity') != selected['base_model_identity']):
            raise ValueError('V20 delivery differs from selection')
        expected_calibration = dict(selected['calibration'])
        if kind == 'refit':
            expected_calibration['frozen_before_refit'] = True
        if checkpoint.get('calibration') != expected_calibration:
            raise ValueError('V20 delivery calibration settings changed')
        from importlib import import_module
        config = checkpoint['config']
        import_module(f"v{checkpoint['format_version']}_model").validate_checkpoint_state_metadata(checkpoint)
        import_module(f"v{checkpoint['format_version']}_core").validate_recipe_config(config)
        scientific = import_module(f"v{checkpoint['format_version']}_runtime").scientific_config
        allowed_refit_changes = {'stage', 'epoch_selection', 'validation_split', 'selection_sha256',
                                 'output_dir', 'neighbor_cache', 'training_pool_signature', 'stop_epoch'}
        def frozen_science(value):
            return {k: v for k, v in scientific(value).items() if k not in allowed_refit_changes}
        if frozen_science(config) != frozen_science(selected['training_config']):
            raise ValueError('V20 delivery training settings changed')
        if kind == 'refit':
            from v7_data import read_manifest
            from v20_core import pool_signature
            rows_manifest = read_manifest(config['data_manifest'])
            source_rows = sorted((row for row in rows_manifest['rows'] if row['role'] in ('clean', 'ambiguous')),
                                 key=lambda row: row['relative_path'])
            supervised = [i for i, row in enumerate(source_rows) if row['role'] == 'clean']
            refit_dir = Path(checkpoint_path).resolve().parent
            if (config.get('stage') != 'refit' or config.get('epoch_selection') != 'frozen_selection'
                    or config.get('validation_split') != 'none_refit_all_manifest_rows'
                    or config.get('selection_sha256') != sha256(selected_path)
                    or config.get('stop_epoch') != selected['selected_epoch']
                    or Path(config['output_dir']).resolve() != refit_dir
                    or Path(config['neighbor_cache']).resolve() != refit_dir / 'cache' / 'neighbors.npz'
                    or config.get('training_pool_signature') != pool_signature(config['feature_signature'], supervised, [])
                    or checkpoint.get('validation', {}).get('indices') != []
                    or checkpoint.get('quality_summary', {}).get('n') != len(supervised)):
                raise ValueError('V20 delivery is not the frozen full-data refit')
        import torch
        if not torch.equal(torch.as_tensor(checkpoint['class_bias']).float(), torch.as_tensor(selected['class_bias']).float()):
            raise ValueError('V20 delivery calibration changed')
        if kind == 'validation' and sha256(checkpoint_path) != selected['source_checkpoint_sha256']:
            raise ValueError('V20 selected validation checkpoint changed')
        if kind == 'refit' and checkpoint.get('refit_selection_sha256') != sha256(selected_path):
            raise ValueError('V20 refit did not freeze selection')
    elif kind == 'fallback_refit':
        frozen = selected['fallback']
        if (selected['refit_required'] is not False
                or str(Path(checkpoint_path).resolve()) != selected['source_checkpoint']
                or sha256(checkpoint_path) != frozen['checkpoint_sha256']
                or sha256(out / 'pred_results.csv') != frozen['csv_sha256']
                or sha256(out / 'pred_results.zip') != frozen['zip_sha256']):
            raise ValueError('V20 fallback differs from frozen historical full-refit package')
    else:
        raise ValueError('unknown V20 delivery kind')
    provenance = dict(format_version=20, kind=kind, checkpoint=str(Path(checkpoint_path).resolve()),
                      checkpoint_sha256=sha256(checkpoint_path), csv_sha256=sha256(out / 'pred_results.csv'),
                      zip_sha256=sha256(out / 'pred_results.zip'), selection_sha256=sha256(selected_path),
                      recipe=checkpoint['config']['recipe'], epoch=checkpoint['selected_epoch'],
                      dataset_signature=checkpoint['dataset_signature'], linecount=rows,
                      view_weights=checkpoint['calibration']['view_weights'], alpha=checkpoint['calibration']['alpha'],
                      precision=checkpoint['calibration']['precision'], online_score=None, created_at=now())
    atomic_json(out / 'provenance.json', provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'state', 'resource-check', 'freeze-resources'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', default='preparation')
    parser.add_argument('--status', default='running')
    parser.add_argument('--state-file', default='status.json')
    parser.add_argument('--resource-json')
    parser.add_argument('--v19-resource')
    parser.add_argument('--candidate-a')
    parser.add_argument('--candidate-b')
    parser.add_argument('--source-manifest')
    parser.add_argument('--model-dir')
    args = parser.parse_args()
    if args.command == 'init':
        print(initialize(args.root)['started_at_unix'])
    elif args.command == 'state':
        state(args.root, args.stage, args.status, args.state_file)
    elif args.command == 'resource-check':
        read_resource_plan(args.resource_json)
    else:
        freeze_resources(args.root, args)


if __name__ == '__main__':
    main()
