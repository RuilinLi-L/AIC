"""Correct diagnostic CSV split names for excluded active rows in an already complete audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from v7_data import load_manifest_dataset
from audit_generalization_v19 import source_record
from v15_pipeline_support import atomic_json, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--train-dir', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.audit.read_text())
    if (report.get('format_version') != 19 or report.get('status') != 'passed'
            or report.get('decoded_cross_split_groups') != 0
            or report.get('decoded_cross_split_pairs') != 0
            or 'decoded_csv_split_correction' in report):
        raise ValueError('only a fresh passed audit can receive one diagnostic split correction')
    source = report['sources']['manifest']
    if sha256(source['path']) != source['sha256']:
        raise ValueError('audited data manifest changed')
    manifest = load_manifest_dataset(source['path'], args.train_dir, 'partial')
    if manifest.signature != report['dataset_signature']:
        raise ValueError('audited data signature changed')
    csv_record = report['artifacts']['decoded_images']
    path = Path(csv_record['path'])
    if sha256(path) != csv_record['sha256']:
        raise ValueError('decoded CSV differs from audit output')
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    train = set(manifest.train_indices)
    validation = set(manifest.validation_indices)
    changed = []
    for index, row in enumerate(rows):
        if int(row['row_index']) != index:
            raise ValueError('decoded CSV rows are not in manifest order')
        expected = 'train' if index in train else 'validation' if index in validation else 'excluded'
        if row['split'] != expected:
            if row['split'] != 'validation' or expected != 'excluded':
                raise ValueError('unexpected decoded CSV split error')
            changed.append(index)
            row['split'] = expected
    if len(changed) != 12 or len(rows) != report['decoded_rows']:
        raise ValueError('expected twelve excluded active rows')
    temp = path.with_name('.decoded_images.corrected.tmp.csv')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    report['artifacts']['decoded_images'] = source_record(path)
    report['decoded_csv_split_correction'] = {
        'rows': changed, 'changed_field': 'split', 'from': 'validation', 'to': 'excluded',
        'cross_split_hash_comparison_changed': False,
        'reason': 'Original diagnostic CSV called all nontraining active rows validation; derived manifest excludes twelve.',
    }
    atomic_json(args.audit, report)
    print(json.dumps({'corrected_rows': len(changed), 'audit_sha256': sha256(args.audit)}), flush=True)


if __name__ == '__main__':
    main()
