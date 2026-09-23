import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from v7_data import (
    assert_no_validation_hash_overlap,
    build_manifest,
    effective_repeat_factors,
    read_manifest,
    write_manifest,
)


class V7DataTests(unittest.TestCase):
    def _image(self, path: Path, color):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (9, 7), color=color).save(path)

    def test_duplicate_roles_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train"
            self._image(root / "0000" / "a.png", (255, 0, 0))
            (root / "0001").mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / "0000" / "a.png", root / "0001" / "z.png")
            self._image(root / "0000" / "b.png", (0, 0, 255))
            shutil.copy2(root / "0000" / "b.png", root / "0000" / "c.png")
            self._image(root / "0001" / "unique.png", (0, 255, 0))
            bad = root / "0001" / "bad.jpg"
            bad.write_bytes(b"not an image")

            first = build_manifest(root)
            second = build_manifest(root)
            self.assertEqual(first["dataset_signature"], second["dataset_signature"])
            by_path = {row["relative_path"]: row for row in first["rows"]}
            self.assertEqual(by_path["0000/a.png"]["role"], "ambiguous")
            self.assertEqual(by_path["0000/a.png"]["candidate_labels"], [0, 1])
            self.assertEqual(by_path["0001/z.png"]["role"], "drop_cross_label_duplicate")
            self.assertEqual(by_path["0000/b.png"]["role"], "clean")
            self.assertEqual(by_path["0000/c.png"]["role"], "drop_same_label_duplicate")
            self.assertEqual(by_path["0001/bad.jpg"]["role"], "unreadable")
            assert_no_validation_hash_overlap(first)
            manifest_path = Path(directory) / "manifest.json"
            write_manifest(first, manifest_path)
            self.assertEqual(read_manifest(manifest_path)["dataset_signature"], first["dataset_signature"])

    def test_repeat_factors_are_tail_only_and_ambiguous_stays_one(self):
        labels = np.asarray([0] + [1] * 4 + [2] * 9 + [0], dtype=np.int64)
        clean = np.ones(len(labels), dtype=np.bool_)
        clean[-1] = False
        factors, effective = effective_repeat_factors(labels, clean, range(len(labels)))
        self.assertAlmostEqual(factors[0], 2.0)
        self.assertTrue(np.allclose(factors[1:5], 1.0))
        self.assertTrue(np.allclose(factors[5:14], 1.0))
        self.assertEqual(factors[-1], 1.0)
        np.testing.assert_allclose(effective, [2.0, 4.0, 9.0])


if __name__ == "__main__":
    unittest.main()
