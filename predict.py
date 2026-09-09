"""Generate the single-model prediction CSV required by the AIC challenge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader

from robust_clip import (
    RobustCLIPClassifier,
    TestImageDataset,
    encode_class_text,
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
    text_features = encode_class_text(model, processor, class_names, device)
    classifier = RobustCLIPClassifier(
        model,
        text_features,
        bottleneck=int(checkpoint.get("config", {}).get("bottleneck", 128)),
    ).to(device)
    classifier.load_state_dict(checkpoint["model"], strict=True)
    classifier.eval()

    test_root = Path(args.test_dir).resolve()
    paths = list_test_images(test_root)
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
    with torch.no_grad():
        for batch in loader:
            logits, _, _, _ = classifier(batch["pixel_values"].to(device, non_blocking=True), text_features)
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())

    mapping = load_class_map(args.class_map)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        for name, index in zip(names, predictions):
            numeric_id = class_id(class_names[index], index, mapping, args.index_offset)
            handle.write(f"{name}, {numeric_id:04d}\n")
    print(f"wrote={out.resolve()} rows={len(predictions)}")


if __name__ == "__main__":
    main()
