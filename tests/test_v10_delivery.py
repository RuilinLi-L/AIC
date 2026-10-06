"""V10 comparison and competition-package checks."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from compare_v10 import CONDITIONS, compare
from predict_v10 import validate_submission


class V10DeliveryTests(unittest.TestCase):
    def _evaluation(self, root: Path, name: str, predictions: list[int]) -> Path:
        directory = root / name
        directory.mkdir()
        labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
        logits = np.full((20, 2), -1.0, dtype=np.float32)
        logits[np.arange(20), predictions] = 1.0
        metrics = {
            "macro_accuracy": float(np.mean([
                np.mean(np.asarray(predictions)[labels == cls] == cls) for cls in (0, 1)
            ])),
            "micro_accuracy": float(np.mean(np.asarray(predictions) == labels)),
            "tail_accuracy": float(np.mean(np.asarray(predictions) == labels)),
        }
        (directory / "strict_eval.json").write_text(json.dumps({
            "dataset_signature": "same-dataset",
            "outer_fold_ids": [index % 5 for index in range(20)],
            "conditions": {condition: {"outer_metrics": metrics} for condition in CONDITIONS},
        }), encoding="utf-8")
        (directory / "val_row_keys.json").write_text(
            json.dumps([f"{labels[index]}/{index:02d}.jpg" for index in range(20)]), encoding="utf-8"
        )
        np.save(directory / "val_labels.npy", labels)
        for condition in CONDITIONS:
            np.save(directory / f"val_oof_{condition}.npy", logits)
        return directory

    def test_recommendation_uses_paired_out_of_fold_predictions(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            baseline = self._evaluation(root, "baseline", [0] * 5 + [1] * 5 + [1] * 5 + [0] * 5)
            candidate = self._evaluation(root, "candidate", [0] * 10 + [1] * 10)
            result = compare(baseline, candidate, bootstrap_repeats=100)
            self.assertTrue(result["recommended_for_submission"])
            self.assertEqual(result["status"], "recommended_for_online_test")
            self.assertEqual(result["conditions"]["combined"]["macro_delta"], 0.5)
            (candidate / "val_row_keys.json").write_text(
                json.dumps(["0/01.jpg", "0/00.jpg"] +
                           [f"{0 if index < 10 else 1}/{index:02d}.jpg" for index in range(2, 20)]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different validation rows"):
                compare(baseline, candidate, bootstrap_repeats=10)

    def test_zip_must_match_headerless_four_digit_predictions(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            csv_path = root / "pred_results.csv"
            zip_path = root / "pred_results.zip"
            lines = "a.jpg, 0001\nb.jpg, 0749\n"
            csv_path.write_text(lines, encoding="utf-8")
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(csv_path, arcname="pred_results.csv")
            validate_submission(csv_path, zip_path, 2, ["a.jpg", "b.jpg"])
            csv_path.write_text("filename,class\n" + lines, encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_submission(csv_path, zip_path, 2)


if __name__ == "__main__":
    unittest.main()
