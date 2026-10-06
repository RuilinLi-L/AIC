"""Verify downloaded V17 fallback delivery and flatten its training diagnostics."""

import csv
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
RECIPES = ("agreement_recovery", "dynamic_prototype")
FIELDS = (
    "recovery_by_class", "repaired_by_class", "added_by_class",
    "withdrawn_by_class", "teacher_disagreement_by_class",
    "dynamic_only_selected_by_class", "dynamic_only_selected_by_target_class",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


selection_file = ROOT / "selection.json"
comparison_file = ROOT / "paired_comparison.json"
selection = json.loads(selection_file.read_text())
comparison = json.loads(comparison_file.read_text())
manifest = json.loads((ROOT / "test_filenames_digest.json").read_text())
assert manifest["expected_count"] == manifest["distinct_count"] == 37444
assert selection["format_version"] == comparison["format_version"] == 17
assert selection["recipe"] == "expanded_mlp"
assert selection["candidate_accepted"] is False
assert selection["refit_required"] is comparison["refit_required"] is False
assert comparison["winner_key"] == "baseline_v15"
assert selection["comparison"] == comparison
assert len(selection["class_names"]) == 750
classes = set(selection["class_names"])
assert all(len(name) == 4 and name.isdigit() for name in classes)

packages = {}
for kind, expected_stage in (("validation", "validation"), ("fallback_refit", "refit")):
    directory = ROOT / kind
    csv_file = directory / "pred_results.csv"
    zip_file = directory / "pred_results.zip"
    provenance = json.loads((directory / "provenance.json").read_text())
    assert provenance["linecount"] == 37444
    assert provenance["recipe"] == "expanded_mlp"
    assert provenance["source_format_version"] == 15
    assert provenance["stage"] == expected_stage
    assert provenance["online_score"] is None
    assert provenance["selection_sha256"] == sha256(selection_file)
    assert provenance["csv_sha256"] == sha256(csv_file)
    assert provenance["zip_sha256"] == sha256(zip_file)
    with zipfile.ZipFile(zip_file) as archive:
        assert archive.namelist() == ["pred_results.csv"]
        assert archive.testzip() is None
        assert archive.read("pred_results.csv") == csv_file.read_bytes()
    with csv_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, skipinitialspace=True))
    assert len(rows) == 37444
    assert all(len(row) == 2 and row[1] in classes for row in rows)
    names = [row[0] for row in rows]
    assert len(set(names)) == 37444
    assert hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest() == manifest["filename_sha256"]
    if kind == "validation":
        assert provenance["checkpoint_sha256"] == selection["source_checkpoint_sha256"]
    v15_kind = "v15_validation" if kind == "validation" else "v15_refit"
    original = ROOT.parent / "v15_delivery" / v15_kind
    original_provenance = json.loads((original / "provenance.json").read_text())
    assert provenance["checkpoint_sha256"] == original_provenance["checkpoint_sha256"]
    assert provenance["csv_sha256"] == original_provenance["csv_sha256"]
    assert provenance["zip_sha256"] == original_provenance["zip_sha256"]
    assert sha256(zip_file) == sha256(original / "pred_results.zip")
    packages[kind] = {"rows": len(rows), "unique_filenames": len(set(names)),
                      "only_zip_member": "pred_results.csv",
                      "byte_identical_to_existing_v15_zip": True, "provenance": provenance}

per_class_file = ROOT / "per_class_supervision.csv"
checks = {}
with per_class_file.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["recipe", "epoch", "source_class", "audit_previous_epoch", "audit_epoch",
                     "dynamic_prototype_epoch", "recovered_original_labels", "repaired_original_labels",
                     "repair_added", "repair_withdrawn", "teacher_disagreement",
                     "dynamic_only_repaired_original_labels", "dynamic_only_target_labels"])
    for recipe in RECIPES:
        metrics = json.loads((ROOT / "training" / recipe / "metrics.json").read_text())
        assert metrics["format_version"] == 17 and metrics["recipe"] == recipe
        assert metrics["stage"] == "validate" and metrics["classes"] == 750
        assert len(metrics["training_history"]) == 24
        assert metrics["selected_epoch"] == 16
        eval_file = ROOT / "evaluations" / recipe / "strict_eval.json"
        assert sha256(eval_file) == selection["evaluation_sha256"][
            "candidate_a" if recipe == "agreement_recovery" else "candidate_b"]
        evaluation = json.loads(eval_file.read_text())
        assert evaluation["selected"]["epoch"] == 16
        selected = {}
        for record in metrics["training_history"]:
            epoch = record["epoch"]
            diag = record["noise_diagnostics"]
            for name in FIELDS:
                assert len(diag.get(name, [0] * 750)) == 750
            recovered = diag["recovery_by_class"]
            repaired = diag["repaired_by_class"]
            dynamic_source = diag.get("dynamic_only_selected_by_class", [0] * 750)
            dynamic_target = diag.get("dynamic_only_selected_by_target_class", [0] * 750)
            assert sum(recovered) == diag["recovery_rows"]
            assert sum(repaired) == record["repair_plan_unique"]
            assert sum(dynamic_source) == sum(dynamic_target) == diag.get("dynamic_only_selected", 0)
            if recipe == "dynamic_prototype":
                assert sum(recovered) == 0
            else:
                assert sum(dynamic_source) == 0
            for index, class_name in enumerate(selection["class_names"]):
                writer.writerow([recipe, epoch, class_name, diag["audit_previous_epoch"],
                                 diag["audit_epoch"], diag["dynamic_prototype_epoch"],
                                 recovered[index], repaired[index],
                                 diag["added_by_class"][index], diag["withdrawn_by_class"][index],
                                 diag["teacher_disagreement_by_class"][index],
                                 dynamic_source[index], dynamic_target[index]])
            if epoch in (16, 24):
                selected[str(epoch)] = {"recovered": sum(recovered), "repaired": sum(repaired),
                                        "dynamic_only": sum(dynamic_source),
                                        "quality_base_mean": diag["quality_base_mean"],
                                        "quality_train_mean": diag["quality_train_mean"]}
        checks[recipe] = {"selected_epoch": 16, "epochs": 24, "per_class_rows": 24 * 750,
                          "summaries": selected, "strict_eval_sha256": sha256(eval_file)}

candidate_scores = {}
for key in ("candidate_a", "candidate_b"):
    candidate = comparison["candidates"][key]
    native = candidate["conditions"]["native"]
    assert candidate["candidate_accepted"] is False
    assert candidate["macro_gain_pp"] < 0.2
    candidate_scores[key] = {
        "macro_accuracy_pct": native["candidate_outer_metrics"]["macro_accuracy"] * 100,
        "tail_accuracy_pct": native["candidate_outer_metrics"]["tail_accuracy"] * 100,
        "macro_nll": native["candidate_outer_metrics"]["macro_nll"],
        "macro_gain_pp": candidate["macro_gain_pp"],
        "tail_gain_pp": candidate["tail_gain_pp"],
        "bootstrap_macro_delta_ci95_pp": [native["paired_class_bootstrap"]["ci95_low"] * 100,
                                           native["paired_class_bootstrap"]["ci95_high"] * 100],
    }

result = {"format_version": 17, "verification": "passed", "online_score": None,
          "selection_sha256": sha256(selection_file), "comparison_sha256": sha256(comparison_file),
          "baseline_macro_accuracy_pct": comparison["baseline_native_metrics"]["macro_accuracy"] * 100,
          "candidate_scores": candidate_scores, "packages": packages,
          "training": checks, "per_class_csv_sha256": sha256(per_class_file),
          "test_filenames_sha256": manifest["filename_sha256"]}
(ROOT / "delivery_verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps({"verification": result["verification"],
                  "baseline_macro_accuracy_pct": result["baseline_macro_accuracy_pct"],
                  "candidate_scores": candidate_scores,
                  "packages": {key: value["rows"] for key, value in packages.items()},
                  "per_class_rows": 24 * 750 * 2}, indent=2))
