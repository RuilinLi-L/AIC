"""Generate the single-model prediction CSV required by the AIC challenge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from robust_clip import (
    RobustCLIPClassifier,
    TestImageDataset,
    inject_lora_from_config,
    load_classifier_state,
    list_test_images,
    load_clip,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="model.pt produced by train.py")
    parser.add_argument("--model-dir", default="clip-ViT-B-32", help="Local OpenAI CLIP ViT-B/32 directory")
    parser.add_argument("--test-dir", required=True, help="Official test image directory")
    parser.add_argument("--output", default="pred_results.csv")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping class name to submission id")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--filename-mode", choices=["basename", "relative"], default="basename")
    parser.add_argument("--index-offset", type=int, default=0)
    parser.add_argument("--logits-output", default=None, help="Optional .npy file for all test logits")
    parser.add_argument("--tta", choices=["none", "hflip"], default="none", help="Optional single-model test-time augmentation")
    parser.add_argument(
        "--original-weight",
        type=float,
        default=None,
        help="Original-view weight for hflip TTA; defaults to calibrated checkpoint metadata or 0.5",
    )
    parser.add_argument(
        "--base-logits-input",
        default=None,
        help="Existing uncalibrated original-view logits; with hflip, compute only the flipped view",
    )
    return parser.parse_args()


def load_class_map(path: str | None) -> Dict[str, int]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("class-map must be a JSON object: {\"class_name\": numeric_id}")
    return {str(key): int(item) for key, item in value.items()}


def class_id(class_name: str, index: int, mapping: Dict[str, int], index_offset: int) -> int:
    if class_name in mapping:
        return mapping[class_name]
    try:
        return int(class_name)
    except ValueError:
        return index + index_offset


def output_name(path: str, test_root: Path, mode: str) -> str:
    p = Path(path)
    if mode == "relative":
        return p.relative_to(test_root).as_posix()
    return p.name


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names: Sequence[str] = checkpoint["class_names"]
    model, processor = load_clip(args.model_dir, device)
    classifier = RobustCLIPClassifier(
        model,
        text_features=None,
        bottleneck=int(checkpoint.get("config", {}).get("bottleneck", 128)),
        classifier_init=torch.randn(len(class_names), int(model.config.projection_dim)),
        initial_logit_scale=float(checkpoint.get("config", {}).get("initial_logit_scale", 10.0)),
        max_logit_scale=float(checkpoint.get("config", {}).get("max_logit_scale", 100.0)),
        learn_logit_scale=bool(checkpoint.get("config", {}).get("learn_logit_scale", True)),
    ).to(device)
    lora_modules = inject_lora_from_config(classifier.clip, checkpoint.get("config", {}))
    if lora_modules:
        print(f"loaded_lora_modules={len(lora_modules)}")
    load_classifier_state(classifier, checkpoint["model"])
    classifier.eval()
    class_bias = checkpoint.get("class_bias")
    if class_bias is not None:
        class_bias = torch.as_tensor(class_bias, dtype=torch.float32, device=device)
        if class_bias.shape != (len(class_names),):
            raise ValueError(f"class_bias must have shape ({len(class_names)},), got {tuple(class_bias.shape)}")
        print(f"using_class_bias={checkpoint.get('calibration', {})}")
    if args.original_weight is not None and not 0.0 <= args.original_weight <= 1.0:
        raise ValueError("--original-weight must be in [0, 1]")
    calibration = checkpoint.get("calibration", {})
    metadata_weight = calibration.get("original_weight")
    view_weights = calibration.get("view_weights")
    if metadata_weight is None and isinstance(view_weights, dict):
        metadata_weight = view_weights.get("original")
    original_weight = float(args.original_weight if args.original_weight is not None else (metadata_weight if metadata_weight is not None else 0.5))
    if not 0.0 <= original_weight <= 1.0:
        raise ValueError("checkpoint original TTA weight must be in [0, 1]")
    if args.tta == "hflip":
        print(f"tta_weights original={original_weight:.4f} hflip={1.0 - original_weight:.4f}")

    test_root = Path(args.test_dir).resolve()
    paths = list_test_images(test_root)
    if args.base_logits_input and args.tta != "hflip":
        raise ValueError("--base-logits-input requires --tta hflip")
    base_logits = None
    if args.base_logits_input:
        base_logits = np.load(args.base_logits_input, mmap_mode="r")
        expected_shape = (len(paths), len(class_names))
        if base_logits.shape != expected_shape:
            raise ValueError(f"base logits shape is {base_logits.shape}, expected {expected_shape}")
        print(f"loaded_base_logits={Path(args.base_logits_input).resolve()} shape={base_logits.shape}")
    if args.filename_mode == "basename":
        names = [output_name(path, test_root, args.filename_mode) for path in paths]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate basenames detected; use --filename-mode relative")
    else:
        names = [output_name(path, test_root, args.filename_mode) for path in paths]
    dataset = TestImageDataset(paths, processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    predictions = []
    all_logits = []
    row_offset = 0
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            if args.tta == "hflip" and base_logits is not None:
                flipped_logits, _, _, _ = classifier(torch.flip(pixels, dims=[3]), None)
                batch_base = torch.tensor(
                    np.asarray(base_logits[row_offset : row_offset + len(pixels)]),
                    dtype=torch.float32,
                    device=device,
                )
                logits = original_weight * batch_base + (1.0 - original_weight) * flipped_logits
            else:
                logits, _, _, _ = classifier(pixels, None)
                if args.tta == "hflip":
                    flipped_logits, _, _, _ = classifier(torch.flip(pixels, dims=[3]), None)
                    logits = original_weight * logits + (1.0 - original_weight) * flipped_logits
            if class_bias is not None:
                logits = logits + class_bias
            if args.logits_output:
                all_logits.append(logits.float().cpu().numpy())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            row_offset += len(pixels)

    mapping = load_class_map(args.class_map)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        for name, index in zip(names, predictions):
            numeric_id = class_id(class_names[index], index, mapping, args.index_offset)
            handle.write(f"{name}, {numeric_id:04d}\n")
    if args.logits_output:
        logits_path = Path(args.logits_output)
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(logits_path, np.concatenate(all_logits, axis=0))
        print(f"saved_logits={logits_path.resolve()} shape={np.load(logits_path, mmap_mode='r').shape}")
    counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=len(class_names))
    print(
        f"prediction_counts min={int(counts.min())} median={float(np.median(counts)):.1f} "
        f"max={int(counts.max())} cv={float(counts.std() / max(counts.mean(), 1e-8)):.4f}"
    )
    print(f"wrote={out.resolve()} rows={len(predictions)}")


if __name__ == "__main__":
    main()
