"""Joint selection mathematics and immutable mixed-version evaluation sources."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate_tta_v20 as evaluation
import v20_evaluation as joint
from test_v19_delivery import checkpoint as checkpoint19
from v10_evaluation import select_epoch
from v20_core import recipe_config
from v20_model import _official_trainable_shapes


def checkpoint(recipe="rank32_control", epoch=1):
    if recipe in ("rank32_control", "resolution384_rank32", "rank32_dropout"):
        value = checkpoint19(recipe, epoch)
    else:
        value = checkpoint19("rank32_control", epoch)
        value["format_version"] = 20
        value["config"].update(recipe_config(recipe), epoch_selection="joint_epoch_tta_bias_nested_cv")
        shapes = {**_official_trainable_shapes(2, value["config"]), "logit_scale": ()}
        zero = torch.zeros(())
        value["model"] = {name: zero.expand(shape) for name, shape in shapes.items()}
        value["trainable_parameter_names"] = sorted(set(shapes) - {"logit_scale"})
    value["config"]["resource_branch"] = "deduplicated_control"
    return value


def manifest():
    return SimpleNamespace(signature="dataset", validation_indices=[0, 1],
        rows=[("0000/a.jpg", 0, "0000"), ("0001/b.jpg", 1, "0001")],
        class_names=["0000", "0001"])


class V20JointSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_joint_selection_finds_epoch_missed_by_uncalibrated_two_views(self):
        labels = np.repeat(np.arange(2), 10)
        correct = np.eye(2, dtype=np.float32)[labels] * 8
        first = correct.copy()
        first[[0, 1, 10, 11]] = first[[0, 1, 10, 11], ::-1]
        wrong = correct[:, ::-1].copy()
        native = {1: {"two_view": first, "four_view": wrong},
                  2: {"two_view": wrong, "four_view": correct}}
        legacy_epoch, _ = select_epoch({1: first, 2: wrong}, labels, np.arange(20))
        selected = joint.fit_joint_epoch_tta(native, labels, np.arange(20), folds=5, device=torch.device("cpu"))
        self.assertEqual(legacy_epoch, 1)
        self.assertEqual((selected["epoch"], selected["family"]), (2, "four_view"))
        self.assertEqual(selected["inner_cv_macro_accuracy"], 1.)

    def test_outer_rows_labels_and_logits_do_not_enter_joint_selection(self):
        labels = np.repeat(np.arange(2), 6)
        logits = np.eye(2, dtype=np.float32)[labels] * 6
        native = {epoch: {family: logits.copy() for family in joint.FAMILY_WEIGHTS} for epoch in (1, 2)}
        fit = np.array([0, 1, 2, 3, 6, 7, 8, 9])
        before = joint.fit_joint_epoch_tta(native, labels, fit, folds=4, device=torch.device("cpu"))
        held = np.array([4, 5, 10, 11])
        labels[held] = 1 - labels[held]
        for families in native.values():
            for value in families.values():
                value[held] = value[held, ::-1] * 1000
        after = joint.fit_joint_epoch_tta(native, labels, fit, folds=4, device=torch.device("cpu"))
        self.assertEqual(before, after)

    def test_inner_macro_uses_pooled_per_class_counts_with_sparse_tail(self):
        labels = np.array([0] * 5 + [1] * 2 + [2])
        logits = np.tile(np.array([4., 0., 0.], np.float32), (8, 1))
        native = {1: {family: logits for family in joint.FAMILY_WEIGHTS}}
        with patch.object(joint, "ALPHAS", np.array([0.], dtype=np.float32)):
            selected = joint.fit_joint_epoch_tta(native, labels, np.arange(8), folds=4, device=torch.device("cpu"))
        self.assertAlmostEqual(selected["inner_cv_macro_accuracy"], 1 / 3, places=12)
        self.assertEqual((selected["epoch"], selected["family"], selected["alpha"]), (1, "two_view", 0.))

    def test_complete_grid_materializes_native_four_views_but_only_selected_stress_epochs(self):
        labels = np.repeat(np.arange(2), 5)
        logits = np.eye(2, dtype=np.float32)[labels] * 5
        calls = []
        def views(epoch, condition, four):
            calls.append((epoch, condition, four))
            value = logits if condition == "native" else -logits
            return {name: value for name in evaluation.VIEW_NAMES}
        result = joint.strict_nested_evaluation(views, list(range(1, 25)), labels,
            np.array([5, 5]), device=torch.device("cpu"))
        self.assertEqual(result.summary["method"], joint.METHOD)
        self.assertEqual(result.summary["evaluation_protocol"], joint.PROTOCOL)
        self.assertEqual(result.summary["class_training_support"], [5, 5])
        self.assertEqual(result.summary["selected"]["epoch"], 1)
        self.assertEqual({epoch for epoch, condition, four in calls if condition == "native" and four}, set(range(1, 25)))
        self.assertEqual({epoch for epoch, condition, _ in calls if condition != "native"}, {1})
        self.assertEqual(len(result.summary["selected"]["inner_candidates"]), 48)
        self.assertEqual(result.summary["conditions"]["native"]["outer_metrics"]["macro_accuracy"], 1.)
        with self.assertRaisesRegex(ValueError, "24 consecutive"):
            joint.strict_nested_evaluation(views, list(range(1, 24)), labels,
                np.array([5, 5]), device=torch.device("cpu"))

    def test_tie_order_is_fixed(self):
        base = dict(epoch=2, family="four_view", alpha=.5, inner_cv_macro_accuracy=.8, inner_cv_macro_nll=1.)
        self.assertGreater(joint.selection_key({**base, "inner_cv_macro_accuracy": .9}), joint.selection_key(base))
        self.assertGreater(joint.selection_key({**base, "inner_cv_macro_nll": .9}), joint.selection_key(base))
        self.assertGreater(joint.selection_key({**base, "alpha": .4}), joint.selection_key(base))
        self.assertGreater(joint.selection_key({**base, "family": "two_view"}), joint.selection_key(base))
        self.assertGreater(joint.selection_key({**base, "epoch": 1}), joint.selection_key(base))


class V20EvaluationProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_five_source_recipes_require_complete_consistent_24_epochs(self):
        for recipe in ("rank32_control", "resolution384_rank32", "rank32_dropout", "dora_rank32", "loraplus_rank32"):
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); epochs = root / "validation_epochs"; epochs.mkdir()
                for epoch in range(1, 25):
                    torch.save(checkpoint(recipe, epoch), epochs / f"epoch_{epoch:02d}.pt")
                paths, values = evaluation.validate_epoch_checkpoints(root, manifest(), np.array([0, 1]))
                self.assertEqual(len(paths), 24)
                self.assertEqual(values[24]["_evaluation_source_sha256"], evaluation.sha256_file(paths[24]))
                changed = checkpoint(recipe, 24); changed["dataset_signature"] = "changed"
                torch.save(changed, paths[24])
                with self.assertRaisesRegex(ValueError, "dataset signature"):
                    evaluation.validate_epoch_checkpoints(root, manifest(), np.array([0, 1]))
                paths[24].unlink()
                with self.assertRaises(FileNotFoundError):
                    evaluation.validate_epoch_checkpoints(root, manifest(), np.array([0, 1]))

    def test_v19_resource_is_validated_by_legacy_reader_and_requires_dedup(self):
        value = checkpoint()
        with patch("evaluate_tta_v19.validate_resource_checkpoint") as validate:
            evaluation.validate_resource_checkpoint(value, Path("source"))
            validate.assert_called_once_with(value, Path("source"))
            value["config"]["resource_branch"] = "standard"
            with self.assertRaisesRegex(ValueError, "deduplicated"):
                evaluation.validate_resource_checkpoint(value, Path("source"))

    def test_cache_rejects_content_source_row_order_and_config_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "epoch_01.pt"; source.write_bytes(b"checkpoint")
            value = checkpoint(); labels = np.array([0, 1])
            views = {name: np.eye(2, dtype=np.float32) for name in evaluation.VIEW_NAMES}
            def make_cache(current=None, keys=None):
                return evaluation.LogitCache(output_dir=root / "evaluation", paths={1: source},
                    checkpoints={1: current or value}, manifest=manifest(), indices=[0, 1], labels=labels,
                    row_keys=keys or ["0000/a.jpg", "0001/b.jpg"], model_dir="official", device=torch.device("cpu"),
                    batch_size=2, workers=0, force=False)
            def collect(*args, four, requested_views=None, **kwargs):
                names = requested_views or (evaluation.VIEW_NAMES if four else evaluation.VIEW_NAMES[:2])
                return {name: views[name] for name in names}, labels
            with patch.object(evaluation, "build_classifier_from_checkpoint", return_value=(object(), object())), \
                    patch.object(evaluation, "collect_view_logits", side_effect=collect):
                cache = make_cache(); cache.get_views(1, "native", True)
            target = cache._path(1, "native", True)
            with patch.object(evaluation, "build_classifier_from_checkpoint", side_effect=RuntimeError("recompute")):
                reused = make_cache(); actual = reused.get_views(1, "native", True)
                np.testing.assert_array_equal(actual["center"], views["center"])
                self.assertEqual(reused.artifact_hashes[target.name], evaluation.sha256_file(target))
                with self.assertRaisesRegex(RuntimeError, "recompute"):
                    make_cache(keys=["0001/b.jpg", "0000/a.jpg"]).get_views(1, "native", True)
                changed = deepcopy(value); changed["config"]["base_model_identity"] = {"weights": "changed"}
                with self.assertRaisesRegex(RuntimeError, "recompute"):
                    make_cache(current=changed).get_views(1, "native", True)
                target.write_bytes(target.read_bytes() + b"changed")
                with self.assertRaisesRegex(RuntimeError, "recompute"):
                    make_cache().get_views(1, "native", True)
                fresh = make_cache(); source.write_bytes(b"changed checkpoint")
                with self.assertRaisesRegex(ValueError, "source checkpoint changed"):
                    fresh.get_views(1, "native", True)

    def test_native_four_views_reuse_hashed_training_center_and_only_infer_zoom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "epoch_01.pt"; source.write_bytes(b"checkpoint")
            value = checkpoint(); labels = np.array([0, 1]); logits = np.eye(2, dtype=np.float32)
            saved = {"center": logits, "hflip": logits, "labels": labels, "row_indices": labels,
                "precision": np.asarray("fp32"), "format_version": np.asarray(19),
                "checkpoint_sha256": np.asarray(evaluation.sha256_file(source)),
                **{name: np.asarray(value["config"][name]) for name in
                   ("image_size", "zoom_shortest_edge", "lora_rank", "lora_dropout", "mlp_lora_dropout")}}
            training = root / "epoch_01_logits.npz"; np.savez(training, **saved)
            cache = evaluation.LogitCache(output_dir=root / "evaluation", paths={1: source},
                checkpoints={1: value}, manifest=manifest(), indices=[0, 1], labels=labels,
                model_dir="official", device=torch.device("cpu"), batch_size=2, workers=0, force=False)
            zoom = {name: 2 * logits for name in evaluation.VIEW_NAMES[2:]}
            with patch.object(evaluation, "build_classifier_from_checkpoint", return_value=(object(), object())), \
                    patch.object(evaluation, "collect_view_logits", return_value=(zoom, labels)) as collect:
                views = cache.get_views(1, "native", True)
            collect.assert_called_once()
            self.assertEqual(collect.call_args.kwargs["requested_views"], evaluation.VIEW_NAMES[2:])
            np.testing.assert_array_equal(views["center"], logits)
            np.testing.assert_array_equal(views["zoom256"], 2 * logits)
            self.assertEqual(cache.source_training_logit_hashes[str(training.resolve())], evaluation.sha256_file(training))

    def test_source_loaded_tensor_hash_must_match_cache_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "epoch_01.pt"; torch.save(checkpoint(), source)
            loaded = evaluation._read_epoch(source)
            source.write_bytes(b"changed after load")
            with self.assertRaisesRegex(ValueError, "source checkpoint changed after validation"):
                evaluation.LogitCache(output_dir=root / "evaluation", paths={1: source}, checkpoints={1: loaded},
                    manifest=manifest(), indices=[0, 1], labels=np.array([0, 1]), model_dir="official",
                    device=torch.device("cpu"), batch_size=2, workers=0, force=False)

    def test_output_cannot_touch_source_or_existing_legacy_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root / "source"; run.mkdir()
            for output, target in ((run, run / "model.pt"), (run / "eval", run / "eval/model.pt"),
                                   (root, root / "model.pt"), (root / "new", run / "model.pt")):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    evaluation.validate_output_paths(run, output, target)
            legacy = root / "legacy"; legacy.mkdir()
            (legacy / "strict_eval.json").write_text(json.dumps({"method": "old"}))
            with self.assertRaisesRegex(ValueError, "not owned"):
                evaluation.validate_output_paths(run, legacy, legacy / "model.pt")
            evaluation.validate_output_paths(run, root / "new", root / "new/model.pt")

    def test_calibration_keeps_source_version_and_weights(self):
        for recipe in ("rank32_control", "dora_rank32"):
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root / "source/validation_epochs/epoch_01.pt"
                source.parent.mkdir(parents=True)
                original = checkpoint(recipe); torch.save(original, source)
                digest = evaluation.sha256_file(source)
                logits = np.eye(2, dtype=np.float32); bias = np.array([.125, -.125], np.float32)
                selected = dict(epoch=1, family="four_view", alpha=.5,
                    view_weights={name: .25 for name in evaluation.VIEW_NAMES})
                summary = {"method": joint.METHOD, "selected": selected,
                    "conditions": {name: {"outer_metrics": {"macro_accuracy": 1.}} for name in joint.CONDITIONS}}
                result = joint.StrictEvaluation(summary, {name: logits for name in joint.CONDITIONS}, logits + bias, bias)
                cache = SimpleNamespace(device=torch.device("cpu"), manifest=manifest(), hashes={1: digest}, artifact_hashes={})
                output = root / "new"; target = output / "model.pt"
                result_summary = evaluation.write_result(result, output_dir=output, output_checkpoint=target,
                    paths={1: source}, checkpoints={1: evaluation._read_epoch(source)}, cache=cache,
                    row_keys=["0000/a.jpg", "0001/b.jpg"], labels=np.array([0, 1]))
                loaded = torch.load(target, map_location="cpu", weights_only=False)
                self.assertEqual(loaded["format_version"], original["format_version"])
                self.assertEqual(loaded["config"], original["config"])
                self.assertEqual(loaded["calibration"]["method"], joint.METHOD)
                self.assertEqual(loaded["calibration"]["evaluation_protocol"], joint.PROTOCOL)
                self.assertNotIn("_evaluation_source_sha256", loaded)
                self.assertEqual(result_summary["evaluation_format_version"], 20)
                self.assertEqual(evaluation.sha256_file(source), digest)
                for name in loaded["model"]:
                    self.assertTrue(torch.equal(loaded["model"][name], original["model"][name]))


if __name__ == "__main__":
    unittest.main()
