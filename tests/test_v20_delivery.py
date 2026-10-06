"""V20 export must preserve the selected model, refit science, and fallback bytes."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
import torch

import v20_pipeline_support as support
from v20_core import pool_signature
from v20_evaluation import METHOD, PROTOCOL


class V20DeliveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.test_dir = self.root / "test"
        self.test_dir.mkdir()
        for name in ("one.jpg", "two.jpg"):
            Image.new("RGB", (2, 2), "red").save(self.test_dir / name)
        self.delivery = self.root / "delivery"
        self.selection_path = self.root / "selection.json"
        self.selection_path.write_text('{"frozen_selection": true}')
        self.manifest_rows = [
            {"relative_path": "a.jpg", "role": "clean"},
            {"relative_path": "b.jpg", "role": "ambiguous"},
            {"relative_path": "c.jpg", "role": "clean"},
        ]
        env = patch.dict(os.environ, {"AIC_SELECTION": str(self.selection_path),
                                    "AIC_DELIVERY_DIR": str(self.delivery)})
        env.start(); self.addCleanup(env.stop)
        reader = patch("select_v20.read_decision", side_effect=lambda path: deepcopy(self.selected))
        reader.start(); self.addCleanup(reader.stop)
        manifest = patch("v7_data.read_manifest", return_value={"rows": self.manifest_rows})
        manifest.start(); self.addCleanup(manifest.stop)
        # Model/recipe tensor-shape contracts have their own tests. Keep the
        # export path real: torch files, CSV/ZIP contents, hashes, and science.
        for version in (19, 20):
            for module, function in (("model", "validate_checkpoint_state_metadata"),
                                     ("core", "validate_recipe_config")):
                validator = patch(f"v{version}_{module}.{function}")
                validator.start(); self.addCleanup(validator.stop)

    def _package(self, kind, *, flipped=False, comment=b""):
        out = self.delivery / kind
        out.mkdir(parents=True, exist_ok=True)
        csv = out / "pred_results.csv"
        csv.write_text("one.jpg, 0001\ntwo.jpg, 0000\n" if flipped else "one.jpg, 0000\ntwo.jpg, 0001\n")
        with zipfile.ZipFile(out / "pred_results.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(csv, arcname="pred_results.csv")
            archive.comment = comment
        return out

    def _prepare(self, kind="refit", *, version=20):
        recipe = {18: "rank32", 19: "rank32_dropout", 20: "loraplus_rank32"}[version]
        identity = {"model.safetensors": "official-base"}
        config = {
            "recipe": recipe, "stage": "validate", "batch_size": 256,
            "gradient_accumulation": 1, "workers": 8, "base_model_identity": identity,
            "schedule_epochs": 24, "epochs": 24, "seed": 2026,
            "image_size": 320, "zoom_shortest_edge": 366, "lora_rank": 32,
            "lora_lr_a": 5e-5, "lora_lr_b": 2e-4, "precision": "fp32",
            "feature_signature": "frozen-features", "data_manifest": str(self.root / "manifest.json"),
            "epoch_selection": "joint_epoch_tta_bias_nested_cv",
            "validation_split": "manifest_clean_tail_safe_capped_8_percent",
            "resource_sha256": "frozen-resource", "resource_branch": "deduplicated_control",
        }
        calibration = {
            "method": METHOD, "evaluation_protocol": PROTOCOL,
            "selected_epoch": 13, "view_weights": {"center": .5, "hflip": .5},
            "alpha": .2, "precision": "fp32", "image_size": 320,
            "zoom_shortest_edge": 366, "lora_rank": 32,
            "lora_dropout": 0., "mlp_lora_dropout": 0.,
        }
        self.selected = {
            "refit_required": kind != "fallback_refit", "source_format_version": version,
            "recipe": recipe, "selected_epoch": 13, "dataset_signature": "frozen-dataset",
            "class_names": ["0000", "0001"], "base_model_identity": identity,
            "training_config": deepcopy(config), "calibration": deepcopy(calibration), "class_bias": [.1, -.1],
        }
        checkpoint_dir = self.root / "models" / kind
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "model.pt"
        checkpoint = {
            "format_version": version, "training_stage": "validation" if kind == "validation" else "refit",
            "selected_epoch": 13, "dataset_signature": "frozen-dataset", "class_names": ["0000", "0001"],
            "config": deepcopy(config), "calibration": deepcopy(calibration),
            "class_bias": torch.tensor([.1, -.1]), "model": {"small": torch.tensor([1.])},
        }
        if kind == "refit":
            checkpoint["config"].update(
                stage="refit", epoch_selection="frozen_selection", validation_split="none_refit_all_manifest_rows",
                selection_sha256=support.sha256(self.selection_path), output_dir=str(checkpoint_dir),
                neighbor_cache=str(checkpoint_dir / "cache" / "neighbors.npz"),
                training_pool_signature=pool_signature("frozen-features", [0, 2], []), stop_epoch=13,
            )
            checkpoint["calibration"]["frozen_before_refit"] = True
            checkpoint["validation"] = {"indices": []}
            checkpoint["quality_summary"] = {"n": 2}
            checkpoint["refit_selection_sha256"] = support.sha256(self.selection_path)
        torch.save(checkpoint, checkpoint_path)
        out = self._package(kind)
        self.selected.update(source_checkpoint=str(checkpoint_path),
                             source_checkpoint_sha256=support.sha256(checkpoint_path))
        self.selected["fallback"] = {
            "checkpoint_sha256": support.sha256(checkpoint_path),
            "csv_sha256": support.sha256(out / "pred_results.csv"),
            "zip_sha256": support.sha256(out / "pred_results.zip"),
        }
        return checkpoint, checkpoint_path, out

    def _export(self, kind, checkpoint_path):
        return support.export_submission(self.root, kind, checkpoint_path, self.test_dir, expected_rows=2)

    def test_original_historical_fallback_exports_the_exact_package(self):
        _, checkpoint, out = self._prepare("fallback_refit", version=18)
        result = self._export("fallback_refit", checkpoint)
        self.assertEqual(result["kind"], "fallback_refit")
        self.assertEqual(result["linecount"], 2)
        for field in ("checkpoint_sha256", "csv_sha256", "zip_sha256"):
            self.assertEqual(result[field], self.selected["fallback"][field])
        self.assertTrue((out / "provenance.json").is_file())

    def test_changed_fallback_csv_zip_or_checkpoint_is_rejected(self):
        for mutation in ("csv", "zip", "checkpoint"):
            with self.subTest(mutation=mutation):
                value, checkpoint, out = self._prepare("fallback_refit", version=18)
                if mutation == "csv":
                    # Still a valid package with valid classes: only the frozen
                    # prediction bytes distinguish this incorrect fallback.
                    self._package("fallback_refit", flipped=True)
                elif mutation == "zip":
                    # A different container with identical CSV data is not the
                    # approved historical ZIP either.
                    self._package("fallback_refit", comment=b"changed-container")
                else:
                    value["model"]["small"] += 1
                    torch.save(value, checkpoint)
                with self.assertRaisesRegex(ValueError, "frozen historical full-refit package"):
                    self._export("fallback_refit", checkpoint)

    def test_accepted_v19_and_v20_validation_and_refit_can_export(self):
        for version in (19, 20):
            for kind in ("validation", "refit"):
                with self.subTest(version=version, kind=kind):
                    _, checkpoint, out = self._prepare(kind, version=version)
                    result = self._export(kind, checkpoint)
                    self.assertEqual(result["recipe"], self.selected["recipe"])
                    self.assertEqual(result["selection_sha256"], support.sha256(self.selection_path))
                    self.assertEqual(result["view_weights"], self.selected["calibration"]["view_weights"])
                    self.assertTrue((out / "provenance.json").is_file())

    def test_stage_format_classes_and_base_identity_must_match_selection(self):
        for kind in ("validation", "refit"):
            for field in ("stage", "format", "classes", "base", "accepted"):
                with self.subTest(kind=kind, field=field):
                    value, checkpoint, _ = self._prepare(kind)
                    if field == "stage": value["training_stage"] = "refit" if kind == "validation" else "validation"
                    elif field == "format": value["format_version"] = 19
                    elif field == "classes": value["class_names"] = ["0001", "0000"]
                    elif field == "base": value["config"]["base_model_identity"] = {"model.safetensors": "different"}
                    else: self.selected["refit_required"] = False
                    torch.save(value, checkpoint)
                    with self.assertRaisesRegex(ValueError, "delivery differs from selection"):
                        self._export(kind, checkpoint)

    def test_every_frozen_calibration_setting_and_bias_is_preserved(self):
        for kind in ("validation", "refit"):
            for field in ("view_weights", "alpha", "precision", "evaluation_protocol", "selected_epoch", "extra", "bias"):
                with self.subTest(kind=kind, field=field):
                    value, checkpoint, _ = self._prepare(kind)
                    if field == "view_weights": value["calibration"][field] = {"center": 1.}
                    elif field == "alpha": value["calibration"][field] = .3
                    elif field == "precision": value["calibration"][field] = "bf16"
                    elif field == "evaluation_protocol": value["calibration"][field] = {}
                    elif field == "selected_epoch": value["calibration"][field] = 14
                    elif field == "extra": value["calibration"]["unapproved_parameter"] = True
                    else: value["class_bias"][0] = .7
                    torch.save(value, checkpoint)
                    with self.assertRaisesRegex(ValueError, "delivery calibration"):
                        self._export(kind, checkpoint)
        value, checkpoint, _ = self._prepare("refit")
        value["calibration"].pop("frozen_before_refit")
        torch.save(value, checkpoint)
        with self.assertRaisesRegex(ValueError, "calibration settings"):
            self._export("refit", checkpoint)

    def test_scientific_training_settings_cannot_change_during_refit(self):
        for kind in ("validation", "refit"):
            for field, changed in (("lora_lr_b", .001), ("seed", 42), ("resource_sha256", "different")):
                with self.subTest(kind=kind, field=field):
                    value, checkpoint, _ = self._prepare(kind)
                    value["config"][field] = changed
                    torch.save(value, checkpoint)
                    with self.assertRaisesRegex(ValueError, "training settings changed"):
                        self._export(kind, checkpoint)

    def test_refit_uses_all_clean_rows_and_frozen_selection_contract(self):
        for field in ("stage", "stop_epoch", "selection", "pool", "holdout", "clean_count", "neighbor", "refit_hash"):
            with self.subTest(field=field):
                value, checkpoint, _ = self._prepare("refit")
                if field == "stage": value["config"]["stage"] = "validate"
                elif field == "stop_epoch": value["config"]["stop_epoch"] = 12
                elif field == "selection": value["config"]["selection_sha256"] = "changed"
                elif field == "pool": value["config"]["training_pool_signature"] = "missing-holdout-rows"
                elif field == "holdout": value["validation"]["indices"] = [1]
                elif field == "clean_count": value["quality_summary"]["n"] = 1
                elif field == "neighbor": value["config"]["neighbor_cache"] = str(self.root / "old-neighbors.npz")
                else: value["refit_selection_sha256"] = "changed"
                torch.save(value, checkpoint)
                with self.assertRaisesRegex(ValueError, "frozen full-data refit|did not freeze selection"):
                    self._export("refit", checkpoint)

    def test_validation_model_bytes_and_fallback_acceptance_are_frozen(self):
        value, checkpoint, _ = self._prepare("validation")
        value["model"]["small"] += 1
        torch.save(value, checkpoint)
        with self.assertRaisesRegex(ValueError, "validation checkpoint changed"):
            self._export("validation", checkpoint)
        _, checkpoint, _ = self._prepare("fallback_refit", version=18)
        self.selected["refit_required"] = True
        with self.assertRaisesRegex(ValueError, "frozen historical full-refit package"):
            self._export("fallback_refit", checkpoint)


if __name__ == "__main__":
    unittest.main()
