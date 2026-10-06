"""Audit active training/holdout images and frozen nearest neighbours; never reads test data."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import warnings

import numpy as np
from PIL import Image, ImageFile, ImageOps

from v7_data import load_manifest_dataset, read_manifest
from v13_core import model_identity
from v15_pipeline_support import atomic_json, sha256

VIEWS = ['center', 'hflip', 'light_seed_2', 'light_seed_3']


def source_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'sha256': sha256(path), 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns}


def decoded_hash(item):
    index, path, expected_sha = item
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError(f'image content changed since manifest: {path}')
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            with Image.open(io.BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.load()
        except Exception:
            with Image.open(io.BytesIO(data)) as source:
                image = source.convert('RGB')
                image.load()
    digest = hashlib.sha256(struct.pack('>II', *image.size) + image.tobytes()).hexdigest()
    return index, digest, image.width, image.height


def decoded_overlaps(hashes, train_indices, validation_indices):
    training = {}
    for index in train_indices:
        training.setdefault(hashes[index], []).append(index)
    groups = {}
    for index in validation_indices:
        digest = hashes[index]
        if digest in training:
            groups.setdefault(digest, {'train': training[digest], 'validation': []})['validation'].append(index)
    return groups


def nearest_training(features, train_indices, validation_indices, batch_size=128):
    """Cosine nearest neighbour with deterministic smallest-row-index tie breaking."""
    if not train_indices or not validation_indices or set(train_indices) & set(validation_indices):
        raise ValueError('nearest-neighbour pools must be nonempty and disjoint')
    train = np.asarray(sorted(train_indices), dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    values = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError('frozen features contain nonfinite or zero vectors')
    values = values / norms
    training = np.ascontiguousarray(values[train].T)
    for start in range(0, len(validation), batch_size):
        ids = validation[start:start + batch_size]
        scores = values[ids] @ training
        best = scores.argmax(axis=1)
        yield from zip(ids.tolist(), train[best].tolist(), scores[np.arange(len(ids)), best].tolist())


def audit(manifest_path, train_dir, feature_cache, model_dir, output_dir, *, workers=8):
    started = time.monotonic()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'audit.json'
    if report_path.exists():
        raise FileExistsError('audit output already exists; verify and reuse the frozen report')
    manifest = load_manifest_dataset(manifest_path, train_dir, 'partial')
    payload = read_manifest(manifest_path)
    active = sorted((r for r in payload['rows'] if r['role'] in ('clean', 'ambiguous')),
                    key=lambda r: r['relative_path'])
    cache = Path(feature_cache).resolve()
    metadata_path = Path(str(cache) + '.json')
    metadata = json.loads(metadata_path.read_text())
    expected = hashlib.sha256(json.dumps({'dataset': manifest.signature,
        'base': model_identity(model_dir), 'views': VIEWS, 'size': 224, 'version': 13},
        sort_keys=True).encode()).hexdigest()
    features = np.load(cache, mmap_mode='r')
    if (metadata.get('dataset_signature') != expected or metadata.get('views') != VIEWS
            or list(features.shape) != [4, len(manifest.rows), 512]
            or metadata.get('shape') != list(features.shape)):
        raise ValueError('frozen feature provenance or shape mismatch')
    sources = {'manifest': source_record(manifest_path), 'feature_cache': source_record(cache),
               'feature_metadata': source_record(metadata_path)}
    hashes = [None] * len(manifest.rows)
    size_rows = []
    jobs = ((i, row[0], active[i]['sha256']) for i, row in enumerate(manifest.rows))
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    results = executor.map(decoded_hash, jobs, chunksize=64) if executor else map(decoded_hash, jobs)
    try:
        for completed, (index, digest, width, height) in enumerate(results, 1):
            hashes[index] = digest
            size_rows.append((index, digest, width, height))
            if completed % 5000 == 0:
                print(f'v19_audit decoded={completed}/{len(hashes)} elapsed_s={time.monotonic()-started:.1f}', flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    groups = decoded_overlaps(hashes, manifest.train_indices, manifest.validation_indices)
    overlap_path = output / 'decoded_overlap.csv'
    with overlap_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['decoded_sha256', 'train_row', 'validation_row', 'train_path', 'validation_path', 'train_label', 'validation_label'])
        for digest, group in sorted(groups.items()):
            for ti in group['train']:
                for vi in group['validation']:
                    writer.writerow([digest, ti, vi, active[ti]['relative_path'], active[vi]['relative_path'],
                                     manifest.rows[ti][1], manifest.rows[vi][1]])
    decoded_path = output / 'decoded_images.csv'
    training_set = set(manifest.train_indices)
    validation_set = set(manifest.validation_indices)
    with decoded_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row_index','relative_path','split','label','decoded_sha256','width','height'])
        for i, digest, width, height in sorted(size_rows):
            split = 'train' if i in training_set else 'validation' if i in validation_set else 'excluded'
            writer.writerow([i, active[i]['relative_path'], split,
                             manifest.rows[i][1], digest, width, height])
    # Average existing frozen views; no fitting or labels are used for neighbour selection.
    mean_features = np.zeros((len(manifest.rows), 512), dtype=np.float32)
    for view in features:
        mean_features += view.astype(np.float32) / 4.0
    neighbors_path = output / 'validation_neighbors.csv'
    with neighbors_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['validation_row','validation_path','validation_label','train_row','train_path','train_label','cosine_similarity','same_label'])
        for vi, ti, score in nearest_training(mean_features, manifest.train_indices, manifest.validation_indices):
            writer.writerow([vi, active[vi]['relative_path'], manifest.rows[vi][1], ti, active[ti]['relative_path'],
                             manifest.rows[ti][1], score, int(manifest.rows[vi][1] == manifest.rows[ti][1])])
    result = {'format_version': 19, 'kind': 'v19_generalization_audit',
              'status': 'failed' if groups else 'passed', 'dataset_signature': manifest.signature,
              'train_rows': len(manifest.train_indices), 'validation_rows': len(manifest.validation_indices),
              'decoded_rows': len(hashes), 'decoded_cross_split_groups': len(groups),
              'decoded_cross_split_pairs': sum(len(g['train'])*len(g['validation']) for g in groups.values()),
              'sources': sources, 'base_model_identity': model_identity(model_dir), 'feature_signature': expected,
              'artifacts': {key: source_record(path) for key, path in
                  [('decoded_overlap', overlap_path), ('decoded_images', decoded_path), ('validation_neighbors', neighbors_path)]},
              'elapsed_seconds': time.monotonic()-started,
              'scope': 'official training manifest only; nearest neighbours are diagnostic, not automatic relabeling',
              'heldout_labels_are_verified_clean': False}
    atomic_json(report_path, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','artifacts','base_model_identity')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-manifest','train-dir','feature-cache','model-dir','output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    result = audit(args.data_manifest,args.train_dir,args.feature_cache,args.model_dir,args.output_dir,workers=args.workers)
    if result['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
