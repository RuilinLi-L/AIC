"""Check the derived split and matched-control decision boundary."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_v19 import FROZEN_V18_NATIVE, choose, expected_for_branch
from prepare_v19_dedup import prepare
from v7_data import _manifest_digest, load_manifest_dataset, read_manifest, write_manifest


class V19DedupTests(unittest.TestCase):
    def test_exact_train_copies_leave_validation_and_feature_rows_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / 'train' / '0000'; train.mkdir(parents=True)
            rows = []
            for index in range(24):
                path = train / f'{index:04d}.jpg'; path.write_bytes(bytes([index]))
                rows.append({'relative_path': f'0000/{index:04d}.jpg', 'label': 0,
                    'class_name': '0000', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'decode_status': 'ok', 'duplicate_group': None, 'candidate_labels': [0],
                    'role': 'clean', 'split': 'train' if index < 12 else 'validation'})
            payload = {'format_version':1,'seed':2026,'train_root':str(root/'train'),
                'class_names':['0000'],'summary':{'clean_train_files':12},'rows':rows}
            payload['dataset_signature'] = _manifest_digest(rows)
            original = root / 'manifest.json'; write_manifest(payload, original)
            overlap = root / 'overlap.csv'
            with overlap.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['train_path','validation_path','train_label','validation_label'])
                writer.writeheader()
                for index in range(12):
                    writer.writerow({'train_path':rows[index]['relative_path'],
                        'validation_path':rows[index+12]['relative_path'],
                        'train_label':0,'validation_label':0})
            identity = {'config':'test'}
            old = load_manifest_dataset(original, root/'train', 'partial')
            features = root / 'frozen.npy'
            np.save(features, np.ones((4,24,512),np.float16))
            signature = hashlib.sha256(json.dumps({'dataset':old.signature,'base':identity,
                'views':['center','hflip','light_seed_2','light_seed_3'],'size':224,'version':13},
                sort_keys=True).encode()).hexdigest()
            Path(str(features)+'.json').write_text(json.dumps({'dataset_signature':signature}))
            result = prepare(original, overlap, features, root/'out', root/'train', identity)
            refreshed = load_manifest_dataset(result['derived_manifest'], root/'train', 'partial')
            self.assertEqual(refreshed.validation_indices, old.validation_indices)
            self.assertEqual(len(refreshed.train_indices), 0)
            self.assertEqual(len(refreshed.final_indices), 24)
            self.assertEqual(os.stat(features).st_ino, os.stat(result['derived_feature_cache']).st_ino)
            self.assertEqual(read_manifest(original)['dataset_signature'], payload['dataset_signature'])
            self.assertTrue(all(row['split']=='excluded' for row in
                read_manifest(result['derived_manifest'])['rows'][:12]))

    def test_clean_control_must_beat_itself_and_original_v18_floor(self):
        self.assertEqual(expected_for_branch('deduplicated_control')['baseline_v18'], (19,'rank32_control'))
        def summary(macro, tail, micro):
            return {'conditions':{'native':{'outer_metrics':{
                'macro_accuracy':macro,'tail_accuracy':tail,'micro_accuracy':micro,'macro_nll':1.2}}}}
        control = summary(.78,.72,.78)
        old = FROZEN_V18_NATIVE
        a = summary(.782,.72,.78)
        b = summary(.781,.73,.79)
        _, eligible, winner = choose({'baseline_v18':control,'candidate_a':a,'candidate_b':b},'deduplicated_control')
        self.assertTrue(eligible['candidate_a'])
        self.assertFalse(eligible['candidate_b'])
        self.assertEqual(winner,'candidate_a')
        lower_control = summary(.75,.70,.75)
        below_original = summary(.752,.71,.76)
        self.assertGreaterEqual(below_original['conditions']['native']['outer_metrics']['macro_accuracy']
                                - lower_control['conditions']['native']['outer_metrics']['macro_accuracy'], .002-1e-12)
        self.assertLess(below_original['conditions']['native']['outer_metrics']['macro_accuracy'], old['macro_accuracy']+.002)
        self.assertFalse(choose({'baseline_v18':lower_control,'candidate_a':below_original},
                                'deduplicated_control')[1]['candidate_a'])


if __name__ == '__main__':
    unittest.main()
