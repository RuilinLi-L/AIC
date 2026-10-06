"""Create a data/cache identity stub for V19 engineering probes; no learned weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from v7_data import load_manifest_dataset
from v13_core import model_identity
from v15_pipeline_support import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--train-dir', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('benchmark metadata is immutable once written')
    manifest = load_manifest_dataset(args.manifest, args.train_dir, 'partial')
    identity = model_identity(args.model_dir)
    payload = {
        'format_version': 19,
        'training_stage': 'engineering_metadata_only',
        'dataset_signature': manifest.signature,
        'class_names': manifest.class_names,
        'config': {'recipe': 'rank32_control', 'conflict_policy': 'partial',
                   'base_model_identity': identity},
        'model': {'classifier': torch.zeros((len(manifest.class_names), 512))},
        'note': 'Shape and identity only; benchmark initializes the official base separately',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({'path': str(args.output.resolve()), 'sha256': sha256(args.output),
                      'dataset_signature': manifest.signature,
                      'trained_weights_loaded': False}), flush=True)


if __name__ == '__main__':
    main()
