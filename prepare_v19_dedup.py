"""Derive a leakage-free V19 validation split without changing official images."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from v7_data import _manifest_digest, load_manifest_dataset, read_manifest, write_manifest
from v15_pipeline_support import sha256


def prepare(original_manifest: Path, overlap_csv: Path, feature_cache: Path,
            output_dir: Path, train_dir: Path, base_identity: dict) -> dict:
    original_manifest = original_manifest.resolve()
    overlap_csv = overlap_csv.resolve()
    feature_cache = feature_cache.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'dataset_manifest_v19_dedup.json'
    cache_path = output_dir / 'frozen.npy'
    metadata_path = Path(str(cache_path) + '.json')
    record_path = output_dir / 'dedup_derivation.json'
    if any(path.exists() for path in (manifest_path, cache_path, metadata_path, record_path)):
        raise FileExistsError('derived V19 inputs already exist; verify rather than overwrite')

    original = read_manifest(original_manifest)
    active = sorted((row for row in original['rows'] if row['role'] in ('clean', 'ambiguous')),
                    key=lambda row: row['relative_path'])
    active_by_path = {row['relative_path']: row for row in active}
    if len(active_by_path) != len(active):
        raise ValueError('active image paths are not unique')
    with overlap_csv.open(newline='') as handle:
        overlaps = list(csv.DictReader(handle))
    if len(overlaps) != 12 or len({row['train_path'] for row in overlaps}) != 12 or len({row['validation_path'] for row in overlaps}) != 12:
        raise ValueError('expected exactly twelve unique decoded overlaps')
    for pair in overlaps:
        train = active_by_path[pair['train_path']]
        valid = active_by_path[pair['validation_path']]
        if (train['role'] != 'clean' or train['split'] != 'train'
                or valid['role'] != 'clean' or valid['split'] != 'validation'
                or int(train['label']) != int(pair['train_label'])
                or int(valid['label']) != int(pair['validation_label'])
                or pair['train_label'] != pair['validation_label']):
            raise ValueError('decoded overlap differs from official manifest split or label')
    old = load_manifest_dataset(original_manifest, train_dir, 'partial')
    original_metadata = json.loads(Path(str(feature_cache) + '.json').read_text())
    old_feature_signature = hashlib.sha256(json.dumps({
        'dataset': old.signature, 'base': base_identity,
        'views': ['center', 'hflip', 'light_seed_2', 'light_seed_3'], 'size': 224, 'version': 13,
    }, sort_keys=True).encode()).hexdigest()
    features = np.load(feature_cache, mmap_mode='r', allow_pickle=False)
    if (original_metadata.get('dataset_signature') != old_feature_signature
            or list(features.shape) != [4, len(active), 512]):
        raise ValueError('old frozen feature source does not match official manifest')

    # The active row order is deliberately unchanged, so the frozen values remain valid.
    changed = {row['train_path'] for row in overlaps}
    derived = json.loads(json.dumps(original))
    for row in derived['rows']:
        if row['relative_path'] in changed:
            row['split'] = 'excluded'
    derived['summary']['clean_train_files'] -= len(changed)
    derived['summary']['v19_decoded_train_duplicates_excluded'] = len(changed)
    derived['dataset_signature'] = _manifest_digest(derived['rows'])
    write_manifest(derived, manifest_path)
    refreshed = load_manifest_dataset(manifest_path, train_dir, 'partial')
    if ([row[0] for row in old.rows] != [row[0] for row in refreshed.rows]
            or old.validation_indices != refreshed.validation_indices
            or len(old.train_indices) - len(refreshed.train_indices) != 12):
        raise ValueError('derived data changed active row order or validation split')
    os.link(feature_cache, cache_path)
    new_signature = hashlib.sha256(json.dumps({
        'dataset': refreshed.signature, 'base': base_identity,
        'views': ['center', 'hflip', 'light_seed_2', 'light_seed_3'], 'size': 224, 'version': 13,
    }, sort_keys=True).encode()).hexdigest()
    new_metadata = {**original_metadata, 'dataset_signature': new_signature}
    metadata_path.write_text(json.dumps(new_metadata, indent=2) + '\n')
    if os.stat(cache_path).st_ino != os.stat(feature_cache).st_ino:
        raise AssertionError('frozen cache was not hard-linked')
    result = {
        'format_version': 19, 'kind': 'v19_deduplicated_split',
        'policy': 'preserve all validation rows; exclude twelve decoded duplicate train rows; preserve all official rows for full refit',
        'original_manifest': str(original_manifest), 'original_manifest_sha256': sha256(original_manifest),
        'overlap_csv': str(overlap_csv), 'overlap_csv_sha256': sha256(overlap_csv),
        'original_feature_cache': str(feature_cache), 'feature_cache_sha256': sha256(feature_cache),
        'original_feature_metadata_sha256': sha256(Path(str(feature_cache)+'.json')),
        'derived_manifest': str(manifest_path), 'derived_manifest_sha256': sha256(manifest_path),
        'derived_feature_cache': str(cache_path), 'derived_feature_metadata_sha256': sha256(metadata_path),
        'excluded_train_paths': sorted(changed), 'excluded_train_count': 12,
        'train_rows': len(refreshed.train_indices), 'validation_rows': len(refreshed.validation_indices),
        'final_refit_rows': len(refreshed.final_indices), 'dataset_signature': refreshed.signature,
    }
    record_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-manifest', required=True, type=Path)
    parser.add_argument('--overlap-csv', required=True, type=Path)
    parser.add_argument('--feature-cache', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--train-dir', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    args = parser.parse_args()
    from v13_core import model_identity
    print(json.dumps(prepare(args.original_manifest, args.overlap_csv, args.feature_cache,
                             args.output_dir, args.train_dir, model_identity(args.model_dir))), flush=True)


if __name__ == '__main__':
    main()
