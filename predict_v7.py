"""Generate and validate a calibrated four-view V7 competition submission."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_tta_v7 import build_classifier_from_checkpoint
from predict import class_id, load_class_map, output_name
from robust_clip import list_test_images, resolve_device
from train_v7 import FORMAT_VERSION
from v7_views import FourViewTestDataset, VIEW_NAMES, normalize_view_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-dir", default="clip-ViT-B-32")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zip-output", required=True)
    parser.add_argument("--class-map", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--filename-mode", choices=["basename", "relative"], default="basename")
    parser.add_argument("--index-offset", type=int, default=0)
    parser.add_argument("--logits-output", default=None)
    parser.add_argument("--expected-rows", type=int, default=37444)
    return parser.parse_args()


def _weighted_logits(classifier, batch, weights: dict[str, float], device: torch.device):
    need_center = any(name in weights for name in ("center", "hflip"))
    need_zoom = any(name in weights for name in ("zoom256", "zoom256_hflip"))
    pixels: dict[str, torch.Tensor] = {}
    if need_center:
        center = batch["center"].to(device, non_blocking=True)
        pixels["center"] = center
        pixels["hflip"] = torch.flip(center, dims=[3])
    if need_zoom:
        zoom = batch["zoom256"].to(device, non_blocking=True)
        pixels["zoom256"] = zoom
        pixels["zoom256_hflip"] = torch.flip(zoom, dims=[3])
    result = None
    for name in VIEW_NAMES:
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        logits = classifier(pixels[name], None)[0].float()
        result = weight * logits if result is None else result + weight * logits
    if result is None:
        raise ValueError("calibrated V7 checkpoint has no active TTA views")
    return result


def validate_submission(csv_path: Path, zip_path: Path, expected_rows: int) -> None:
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    if expected_rows > 0 and len(lines) != expected_rows:
        raise ValueError(f"submission has {len(lines)} rows, expected {expected_rows}")
    if any(len(line.rsplit(",", 1)) != 2 for line in lines):
        raise ValueError("submission contains a malformed row")
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        if names != ["pred_results.csv"]:
            raise ValueError(f"ZIP must contain only pred_results.csv, found {names}")
        zipped_lines = archive.read("pred_results.csv").decode("utf-8").splitlines()
    if zipped_lines != lines:
        raise ValueError("ZIP CSV does not match the generated submission")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.expected_rows < 0:
        raise ValueError("invalid batch-size, workers, or expected-rows")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("checkpoint is not V7")
    calibration = checkpoint.get("calibration")
    if not isinstance(calibration, dict) or "view_weights" not in calibration:
        raise ValueError("run evaluate_tta_v7.py before V7 prediction")
    weights = normalize_view_weights(calibration["view_weights"])
    class_names: Sequence[str] = checkpoint["class_names"]
    class_bias = checkpoint.get("class_bias")
    if class_bias is None:
        raise ValueError("calibrated V7 checkpoint is missing class_bias")
    class_bias = torch.as_tensor(class_bias, dtype=torch.float32, device=device)
    if class_bias.shape != (len(class_names),):
        raise ValueError("V7 class_bias has the wrong shape")
    classifier, processor = build_classifier_from_checkpoint(checkpoint, args.model_dir, device)
    test_root = Path(args.test_dir).resolve()
    paths = list_test_images(test_root)
    if args.expected_rows > 0 and len(paths) != args.expected_rows:
        raise ValueError(f"test directory has {len(paths)} images, expected {args.expected_rows}")
    names = [output_name(path, test_root, args.filename_mode) for path in paths]
    if args.filename_mode == "basename" and len(names) != len(set(names)):
        raise ValueError("duplicate test basenames; use --filename-mode relative")
    loader = DataLoader(
        FourViewTestDataset(paths, processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    predictions: list[int] = []
    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader, 1):
            logits = _weighted_logits(classifier, batch, weights, device) + class_bias
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            if args.logits_output:
                all_logits.append(logits.cpu().numpy())
            if batch_id % 100 == 0:
                print(f"v7_prediction_batches={batch_id}/{len(loader)}", flush=True)
    mapping = load_class_map(args.class_map)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        for name, index in zip(names, predictions):
            numeric_id = class_id(class_names[index], index, mapping, args.index_offset)
            handle.write(f"{name}, {numeric_id:04d}\n")
    if args.logits_output:
        logits_path = Path(args.logits_output)
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(logits_path, np.concatenate(all_logits, axis=0))
    zip_path = Path(args.zip_output)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output, arcname="pred_results.csv")
    validate_submission(output, zip_path, args.expected_rows)
    counts = np.bincount(np.asarray(predictions), minlength=len(class_names))
    print(
        f"v7_submission rows={len(predictions)} classes_predicted={int((counts > 0).sum())} "
        f"weights={weights} csv={output.resolve()} zip={zip_path.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
