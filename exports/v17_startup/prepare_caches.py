"""Verify and copy immutable V15 train-fold evidence into fresh V17 outputs."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from v7_data import load_manifest_dataset
from v17_core import file_sha256, model_identity, pool_signature, stage_indices
from train_v17 import validate_neighbor_evidence
from v8_neighbors import NeighborEvidence

root = Path('/data/mcxu/AIC/outputs')
manifest = load_manifest_dataset(root / 'v7/dataset_manifest_v7.json',
                                 '/home/mcxu/lrl/AIC/data/train', 'partial')
train, supervised, validation = stage_indices(manifest, 'validate')
identity = model_identity('/home/mcxu/lrl/AIC/clip-ViT-B-32')
signature = hashlib.sha256(json.dumps({
    'dataset': manifest.signature, 'base': identity,
    'views': ['center', 'hflip', 'light_seed_2', 'light_seed_3'],
    'size': 224, 'version': 13,
}, sort_keys=True).encode()).hexdigest()
features = root / 'v13_expanded/cache/frozen.npy'
feature_metadata = json.loads(features.with_suffix('.npy.json').read_text())
assert feature_metadata['dataset_signature'] == signature
assert np.load(features, mmap_mode='r').shape == (4, len(manifest.rows), 512)
pool = pool_signature(signature, supervised, validation)
source = root / 'v15_expanded_mlp/cache/neighbors.npz'
with np.load(source, allow_pickle=False) as cache:
    metadata = json.loads(str(cache['metadata']))
    assert metadata == {'signature': pool, 'k': 32, 'temperature': 0.07, 'folds': 5}
    neighbor = NeighborEvidence(*(cache[key].copy() for key in (
        'label_support', 'top_label', 'top_support', 'percentile', 'fold_ids')))
validate_neighbor_evidence(neighbor, len(manifest.rows))
assert np.all(neighbor.fold_ids[supervised] >= 0)
assert np.all(neighbor.fold_ids[validation] == -1)
digest = file_sha256(source)
destinations = []
for recipe in ('agreement_recovery', 'dynamic_prototype'):
    run = root / f'v17_{recipe}'
    assert not list(run.glob('*.pt')), f'Already trained: {run}'
    destination = run / 'cache/neighbors.npz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    assert file_sha256(destination) == digest
    destinations.append(str(destination))
labels = np.array([row[1] for row in manifest.rows], dtype=np.int64)
counts = np.bincount(labels[supervised], minlength=len(manifest.class_names))
record = {'format_version': 17, 'success': True,
          'dataset_signature': manifest.signature, 'feature_signature': signature,
          'feature_cache': str(features), 'feature_cache_sha256': file_sha256(features),
          'feature_metadata': feature_metadata, 'training_pool_signature': pool,
          'base_model_identity': identity, 'neighbor_source': str(source),
          'neighbor_metadata': metadata, 'neighbor_sha256': digest,
          'neighbor_destinations': destinations, 'train_rows': len(train),
          'supervised_rows': len(supervised), 'validation_rows': len(validation),
          'minimum_supervised_class_rows': int(counts.min()),
          'validation_excluded_from_neighbor_references': True}
(root / 'v17_startup/cache_preparation.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
