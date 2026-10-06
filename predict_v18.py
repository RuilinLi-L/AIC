"""Generate and validate a calibrated single-model V13/V14/V15/V16/V17/V18 submission."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from v18_model import build_classifier_from_checkpoint, build_zoom_transform, checkpoint_resolution
from predict import class_id, load_class_map, output_name
from robust_clip import list_test_images, resolve_device, safe_open_image, _clip_image_transform
from v7_views import VIEW_NAMES, normalize_view_weights
from v18_runtime import autocast_context, loader_options, precision_name, Progress, pin_memory_enabled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-dir", default="clip-ViT-B-32")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zip-output", required=True)
    parser.add_argument("--class-map", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--filename-mode", choices=["basename", "relative"], default="basename")
    parser.add_argument("--index-offset", type=int, default=0)
    parser.add_argument("--logits-output", default=None)
    parser.add_argument("--expected-rows", type=int, default=37444)
    return parser.parse_args()


class RequiredViewTestDataset(Dataset):
    """Decode once and construct only base tensors used by calibrated views."""
    def __init__(self, paths, processor, weights):
        self.paths = list(paths)
        active = normalize_view_weights(weights)
        self.center_transform = (_clip_image_transform(processor, "none")
                                 if any(n in active for n in ("center", "hflip")) else None)
        self.zoom_transform = (build_zoom_transform(processor)
                               if any(n in active for n in ("zoom256", "zoom256_hflip")) else None)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        image = safe_open_image(path)
        result = {"path": str(path)}
        if self.center_transform is not None:
            result["center"] = self.center_transform(image)
        if self.zoom_transform is not None:
            result["zoom256"] = self.zoom_transform(image)
        return result


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
        with autocast_context(device):
            logits = classifier(pixels[name], None)[0]
        logits = logits.float()
        result = weight * logits if result is None else result + weight * logits
    if result is None:
        raise ValueError("calibrated V18 checkpoint has no active TTA views")
    return result


from predict_v13 import validate_submission


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.expected_rows < 1:
        raise ValueError("invalid batch-size, workers, or expected-rows")
    loader_options(args.workers, args.prefetch_factor)
    device = resolve_device(args.device)
    output_paths = [Path(args.output).resolve(), Path(args.zip_output).resolve()]
    if args.logits_output:
        actual_logits = args.logits_output if str(args.logits_output).endswith(".npy") else str(args.logits_output) + ".npy"
        output_paths.append(Path(actual_logits).resolve())
    if Path(args.checkpoint).resolve() in output_paths:
        raise ValueError("prediction output must not overwrite the checkpoint")
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("prediction CSV, ZIP and logits outputs must be distinct")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") not in (13, 14, 15, 16, 17, 18):
        raise ValueError("checkpoint is not V13, V14, V15, V16, V17 or V18")
    calibration = checkpoint.get("calibration")
    if not isinstance(calibration, dict) or "view_weights" not in calibration:
        raise ValueError("run the matching evaluation before prediction")
    if calibration.get("precision") != precision_name(device):
        raise ValueError("prediction precision differs from calibration; recalibrate on the target device")
    raw_weights = calibration["view_weights"]
    if (not isinstance(raw_weights, dict) or not raw_weights
            or any(not np.isfinite(value) or value < 0 for value in raw_weights.values())
            or abs(sum(raw_weights.values()) - 1.0) > 1e-6):
        raise ValueError("checkpoint calibration requires finite normalized nonnegative view weights")
    alpha = calibration.get("alpha")
    if alpha is None or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("checkpoint calibration has invalid alpha")
    if checkpoint["format_version"] in (17, 18):
        if (calibration.get("selected_epoch") != checkpoint.get("selected_epoch")
                or calibration.get("image_size") != checkpoint_resolution(checkpoint)):
            raise ValueError("V18 calibration epoch or resolution differs from the checkpoint")
    if checkpoint["format_version"] == 18:
        for field in ("zoom_shortest_edge", "lora_rank"):
            if calibration.get(field) != checkpoint["config"].get(field):
                raise ValueError(f"V18 calibration {field} differs from the checkpoint")
    weights = normalize_view_weights(calibration["view_weights"])
    class_names: Sequence[str] = checkpoint["class_names"]
    class_bias = checkpoint.get("class_bias")
    if class_bias is None:
        raise ValueError("calibrated V18 checkpoint is missing class_bias")
    class_bias = torch.as_tensor(class_bias, dtype=torch.float32, device=device)
    if class_bias.shape != (len(class_names),) or not torch.isfinite(class_bias).all():
        raise ValueError("V18 class_bias has the wrong shape")
    classifier, processor = build_classifier_from_checkpoint(checkpoint, args.model_dir, device)
    test_root = Path(args.test_dir).resolve()
    paths = list_test_images(test_root)
    if args.expected_rows > 0 and len(paths) != args.expected_rows:
        raise ValueError(f"test directory has {len(paths)} images, expected {args.expected_rows}")
    names = [output_name(path, test_root, args.filename_mode) for path in paths]
    if args.filename_mode == "basename" and len(names) != len(set(names)):
        raise ValueError("duplicate test basenames; use --filename-mode relative")
    loader = DataLoader(
        RequiredViewTestDataset(paths, processor, weights),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory_enabled(device),
        persistent_workers=False,
        **loader_options(args.workers, args.prefetch_factor),
    )
    predictions: list[int] = []
    all_logits: list[np.ndarray] = []
    progress = Progress("prediction", device, len(loader))
    with torch.no_grad():
        for batch_id, batch in enumerate(loader, 1):
            logits = _weighted_logits(classifier, batch, weights, device) + class_bias
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            if args.logits_output:
                all_logits.append(logits.cpu().numpy())
            progress.update(batch_id, len(logits))
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
    validate_submission(output, zip_path, args.expected_rows, names)
    counts = np.bincount(np.asarray(predictions), minlength=len(class_names))
    print(
        f"v18_submission rows={len(predictions)} classes_predicted={int((counts > 0).sum())} "
        f"weights={weights} csv={output.resolve()} zip={zip_path.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
