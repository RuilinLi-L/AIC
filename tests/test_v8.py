import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from predict_v8 import validate_submission
from train_v5 import clone_trainable
from train_v8 import RESUME_VERSION, load_resume, save_resume
from v6_core import SelectiveFeatureQueue
from v7_views import VIEW_NAMES
from v8_calibration import nested_calibration
from v8_neighbors import (
    NeighborEvidence,
    _query_pool,
    blend_quality,
    build_neighbor_evidence,
    load_or_create_neighbor_evidence,
    trusted_mask,
)


class V8NeighborTests(unittest.TestCase):
    def test_self_and_same_hash_are_excluded(self):
        embeddings = F.normalize(torch.tensor([
            [1.0, 0.0], [1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]
        ]), dim=1)
        labels = np.asarray([0, 0, 1, 0])
        hashes = ["duplicate", "duplicate", "other", "distant"]
        support = np.zeros(4, dtype=np.float32)
        top = np.full(4, -1, dtype=np.int32)
        top_support = np.zeros(4, dtype=np.float32)
        _query_pool(
            embeddings, labels, hashes, np.asarray([0]), np.asarray([0, 1, 2, 3]),
            2, np.asarray([3.0, 1.0]), 1, 0.07, 1, support, top, top_support,
        )
        self.assertEqual(int(top[0]), 1)
        self.assertEqual(float(support[0]), 0.0)

    def test_tail_vote_gets_inverse_sqrt_frequency_weight(self):
        embeddings = F.normalize(torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [0.9, -0.1]
        ]), dim=1)
        labels = np.asarray([1, 0, 1])
        support = np.zeros(3, dtype=np.float32)
        top = np.full(3, -1, dtype=np.int32)
        top_support = np.zeros(3, dtype=np.float32)
        _query_pool(
            embeddings, labels, ["q", "a", "b"], np.asarray([0]), np.asarray([1, 2]),
            2, np.asarray([100.0, 4.0]), 2, 0.07, 1, support, top, top_support,
        )
        self.assertGreater(float(support[0]), 0.5)
        self.assertEqual(int(top[0]), 1)

    def test_fold_evidence_is_deterministic_and_validation_is_not_reference(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(4, 20, 8)).astype(np.float32)
        labels = np.asarray([0, 1] * 10)
        hashes = [str(i) for i in range(20)]
        train = list(range(15))
        validation = list(range(15, 20))
        a = build_neighbor_evidence(
            features, labels, hashes, train, validation, 2, torch.device("cpu"),
            k=3, batch_size=4,
        )
        changed = labels.copy()
        changed[validation] = 1 - changed[validation]
        b = build_neighbor_evidence(
            features, changed, hashes, train, validation, 2, torch.device("cpu"),
            k=3, batch_size=4,
        )
        np.testing.assert_array_equal(a.label_support[train], b.label_support[train])
        np.testing.assert_array_equal(a.fold_ids[train], b.fold_ids[train])
        self.assertTrue(np.all(a.fold_ids[validation] == -1))
        self.assertTrue(np.all((a.percentile[train] >= 0) & (a.percentile[train] <= 1)))

    def test_blend_and_trusted_mask_keep_partial_rows_untrusted(self):
        evidence = NeighborEvidence(
            np.asarray([0.8, 0.6, 0.9]), np.asarray([0, 1, 0]),
            np.asarray([0.8, 0.6, 0.9]), np.asarray([0.8, 0.8, 0.8]),
            np.asarray([0, 1, -1]),
        )
        ambiguous = np.asarray([False, False, True])
        quality = blend_quality(np.asarray([0.8, 0.8, 0.8]), evidence, ambiguous)
        trusted = trusted_mask(
            quality, np.asarray([1, 0, 0]), np.asarray([0, 1, 0]), evidence, ambiguous
        )
        np.testing.assert_array_equal(trusted, [True, True, False])
        self.assertEqual(float(quality[2]), 0.25)

    def test_neighbor_cache_round_trip_and_signature_check(self):
        rng = np.random.default_rng(4)
        features = rng.normal(size=(4, 12, 4)).astype(np.float32)
        labels = np.asarray([0, 1] * 6)
        hashes = [str(index) for index in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neighbor_evidence.npz"
            first = load_or_create_neighbor_evidence(
                path, "signature-a", features, labels, hashes,
                list(range(10)), [10, 11], 2, torch.device("cpu"),
            )
            cached = load_or_create_neighbor_evidence(
                path, "signature-a", features, labels, hashes,
                list(range(10)), [10, 11], 2, torch.device("cpu"),
            )
            np.testing.assert_array_equal(first.percentile, cached.percentile)
            with self.assertRaises(ValueError):
                load_or_create_neighbor_evidence(
                    path, "signature-b", features, labels, hashes,
                    list(range(10)), [10, 11], 2, torch.device("cpu"),
                )


class V8CalibrationTests(unittest.TestCase):
    def test_outer_holdout_is_never_used_for_parameter_or_bias_fit(self):
        labels = np.asarray([0, 1] * 5)
        base = np.zeros((10, 2), dtype=np.float32)
        base[np.arange(10), labels] = 2.0
        views = {name: base.copy() for name in VIEW_NAMES}
        fit_rows = []
        bias_rows = []

        def fake_fit(_views, _labels, indices, _family, _device, folds):
            fit_rows.append((tuple(indices), folds))
            return {"view_weights": {"center": 1.0}, "alpha": 0.0,
                    "inner_cv_macro_accuracy": 1.0, "inner_cv_macro_nll": 0.0}

        def fake_bias(logits, _alpha):
            bias_rows.append(len(logits))
            return np.zeros(2, dtype=np.float32)

        with patch("v8_calibration._fit_family", side_effect=fake_fit), patch(
            "v8_calibration._bias", side_effect=fake_bias
        ):
            result = nested_calibration(views, labels, torch.device("cpu"))
        self.assertEqual(len(fit_rows), 11)
        self.assertTrue(all(folds == 4 for _, folds in fit_rows[:10]))
        self.assertTrue(all(folds == 5 for _, folds in fit_rows[10:]))
        self.assertTrue(all(len(indices) == 8 for indices, _ in fit_rows[:10]))
        self.assertEqual(bias_rows, [8] * 10 + [10])
        self.assertEqual(result.summary["selected_family"], "two_view")
        self.assertAlmostEqual(result.summary["selected_outer_metrics"]["macro_accuracy"], 1.0)

    def test_submission_and_resume_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "pred_results.csv"
            archive = root / "pred_results.zip"
            csv.write_text("a.jpg, 0001\nb.jpg, 0000\n", encoding="utf-8")
            with zipfile.ZipFile(archive, "w") as handle:
                handle.write(csv, arcname="pred_results.csv")
            validate_submission(csv, archive, 2)
            with self.assertRaises(ValueError):
                validate_submission(csv, archive, 3)
            resume = root / "resume.pt"
            torch.save({
                "kind": "train_v8_resume", "resume_version": RESUME_VERSION,
                "stage": "validation", "config": {"seed": 2026}, "dataset_signature": "abc",
            }, resume)
            self.assertEqual(
                load_resume(str(resume), {"seed": 2026}, "abc", torch.device("cpu"))["stage"],
                "validation",
            )
            with self.assertRaises(ValueError):
                load_resume(str(resume), {"seed": 2027}, "abc", torch.device("cpu"))

    def test_resume_round_trip_keeps_model_optimizer_and_quality(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        loss = model.classifier.square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        queue = SelectiveFeatureQueue(2, capacity=2, dim=2, device=torch.device("cpu"))
        arrays = {"quality": np.asarray([0.7, 0.25], dtype=np.float32)}
        config = {"version": "v8", "seed": 2026}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            save_resume(
                path, stage="validation", epoch=2, classifier=model, optimizer=optimizer,
                scheduler=scheduler, ema=clone_trainable(model), queue=queue,
                arrays=arrays, optimizer_step=1, history=[{"epoch": 1}],
                best_state=clone_trainable(model), best_record={"epoch": 1},
                selected_epoch=None, config=config, dataset_digest="signature",
            )
            state = load_resume(str(path), config, "signature", torch.device("cpu"))
            self.assertEqual(state["epoch"], 2)
            self.assertEqual(state["optimizer_step"], 1)
            np.testing.assert_array_equal(state["quality"], arrays["quality"])
            self.assertTrue(torch.equal(state["model"]["classifier"], model.classifier.detach()))


if __name__ == "__main__":
    unittest.main()
