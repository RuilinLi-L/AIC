"""Dependency contracts and isolated submission provenance for the V15 pipeline."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def baseline_selection(root):
    from select_v14 import load_selection

    path = root / 'v14_selection.json'
    if not (root / 'v14_pipeline/selection.done').is_file():
        raise ValueError('V14 selection is not marked complete')
    selection = load_selection(path)
    binding = root / 'v15_pipeline/v14_selection_binding.json'
    identity = {'selection': str(path.resolve()), 'sha256': sha256(path)}
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError('V14 selection changed after V15 consumed it')
    if not binding.exists():
        atomic_json(binding, identity)
    return selection


def wait_v14(root, kind, timeout, interval=30):
    deadline = time.monotonic() + timeout
    marker = root / 'v14_pipeline' / ('selection.done' if kind == 'selection' else 'prediction.done')
    while not marker.is_file():
        status_path = root / 'v14_pipeline/status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get('status') == 'failed':
                raise RuntimeError(f'V14 dependency failed: {status}')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'waiting for V14 {kind} exceeded {timeout}s')
        print(f'v15_wait dependency=v14_{kind} remaining_hours={(deadline-time.monotonic())/3600:.2f}', flush=True)
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
    baseline_selection(root)


def checkpoint_path(root, kind):
    import torch

    if kind.startswith('v14_'):
        selection = baseline_selection(root)
        selection_path = root / 'v14_selection.json'
    else:
        from select_v15 import load_selection
        selection_path = root / 'v15_selection.json'
        selection = load_selection(selection_path)
        if selection.get('candidate_accepted') is not True:
            raise ValueError('V15 candidate was not accepted; no V15 submission or refit is allowed')
    if kind.endswith('_validation'):
        return Path(selection['source_checkpoint']).resolve()
    path = root / ('v14_refit' if kind == 'v14_refit' else 'v15_refit') / 'model.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint.get('training_stage') != 'refit'
            or checkpoint.get('refit_selection_sha256') != sha256(selection_path)
            or checkpoint.get('selected_epoch') != selection['selected_epoch']
            or checkpoint.get('dataset_signature') != selection['dataset_signature']
            or checkpoint.get('class_names') != selection['class_names']
            or checkpoint.get('config', {}).get('recipe') != selection['recipe']
            or checkpoint.get('calibration', {}).get('view_weights') != selection['calibration']['view_weights']
            or checkpoint.get('calibration', {}).get('alpha') != selection['calibration']['alpha']
            or checkpoint.get('calibration', {}).get('precision') != selection['calibration']['precision']):
        raise ValueError(f'{kind} model does not match its frozen selection')
    if not torch.equal(torch.as_tensor(checkpoint.get('class_bias')).float(),
                       torch.as_tensor(selection['class_bias']).float()):
        raise ValueError(f'{kind} class bias differs from its frozen selection')
    return path.resolve()


def export_submission(root, kind, checkpoint_path_value, expected_rows=37444, collect=False, test_dir=None):
    import torch

    checkpoint_path_value = Path(checkpoint_path_value).resolve()
    if checkpoint_path_value != checkpoint_path(root, kind):
        raise ValueError('submission checkpoint differs from frozen selection')
    out = root / 'v15_delivery' / kind
    out.mkdir(parents=True, exist_ok=True)
    if collect:
        if kind != 'v14_refit' or not (root / 'v14_pipeline/prediction.done').is_file():
            raise ValueError('only completed V14 refit submissions may be collected')
        for name in ('pred_results.csv', 'pred_results.zip'):
            destination = out / name
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            shutil.copyfile(root / 'v14_refit' / name, temporary)
            temporary.replace(destination)
    csv_path, zip_path = out / 'pred_results.csv', out / 'pred_results.zip'
    with csv_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.reader(handle, skipinitialspace=True))
    if len(rows) != expected_rows or any(len(row) != 2 for row in rows):
        raise ValueError('submission row count or field count is invalid')
    if len({row[0] for row in rows}) != expected_rows:
        raise ValueError('submission has duplicate filenames')
    if test_dir is None:
        raise ValueError('test directory is required to verify submission filenames')
    from robust_clip import list_test_images
    names = [Path(path).name for path in list_test_images(test_dir)]
    if len(names) != expected_rows or len(set(names)) != expected_rows or set(names) != {row[0] for row in rows}:
        raise ValueError('submission filenames differ from the official test directory')
    checkpoint = torch.load(checkpoint_path_value, map_location='cpu', weights_only=False)
    classes = set(checkpoint['class_names'])
    if any(len(row[1]) != 4 or not row[1].isdigit() or row[1] not in classes for row in rows):
        raise ValueError('submission contains invalid class identifiers')
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ['pred_results.csv'] or archive.read('pred_results.csv') != csv_path.read_bytes():
            raise ValueError('ZIP must contain only the identical pred_results.csv')
    calibration = checkpoint['calibration']
    provenance = {
        'kind': kind, 'checkpoint': str(checkpoint_path_value),
        'checkpoint_sha256': sha256(checkpoint_path_value),
        'csv_sha256': sha256(csv_path), 'zip_sha256': sha256(zip_path),
        'epoch': checkpoint['selected_epoch'], 'recipe': checkpoint['config']['recipe'],
        'stage': checkpoint['training_stage'], 'view_weights': calibration['view_weights'],
        'alpha': calibration['alpha'], 'precision': calibration['precision'], 'linecount': len(rows),
        'online_score': None, 'target_online_score': 72,
        'created_at': datetime.datetime.now().astimezone().isoformat(),
    }
    existing = out / 'provenance.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        for key in ('checkpoint_sha256', 'csv_sha256', 'zip_sha256'):
            if previous[key] != provenance[key]:
                raise ValueError(f'completed delivery changed: {kind}/{key}')
        return previous
    atomic_json(existing, provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('wait-selection', 'wait-prediction', 'checkpoint', 'export', 'accepted', 'baseline-evaluation', 'state'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--kind', choices=('v14_validation', 'v14_refit', 'v15_validation', 'v15_refit'))
    parser.add_argument('--checkpoint')
    parser.add_argument('--test-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=48 * 3600)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--expected-rows', type=int, default=37444)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--stage')
    parser.add_argument('--status')
    parser.add_argument('--state-file', default='status.json')
    args = parser.parse_args()
    if args.action.startswith('wait-'):
        wait_v14(args.root, args.action.removeprefix('wait-'), args.timeout, args.interval)
    elif args.action == 'checkpoint':
        print(checkpoint_path(args.root, args.kind))
    elif args.action == 'export':
        print(json.dumps(export_submission(args.root, args.kind, args.checkpoint, args.expected_rows, args.collect, args.test_dir)))
    elif args.action in ('accepted', 'baseline-evaluation'):
        from select_v15 import read_decision
        baseline_selection(args.root)
        selection = read_decision(args.root / 'v15_selection.json')
        print(selection['comparison']['baseline'] if args.action == 'baseline-evaluation'
              else ('1' if selection['candidate_accepted'] else '0'))
    else:
        atomic_json(args.root / 'v15_pipeline' / args.state_file, {
            'stage': args.stage, 'status': args.status, 'target_online_score': 72,
            'updated_at': datetime.datetime.now().astimezone().isoformat(),
        })


if __name__ == '__main__':
    main()
