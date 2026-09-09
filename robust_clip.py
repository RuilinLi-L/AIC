"""First-version robust fine-tuning utilities for the AIC challenge.

The implementation deliberately keeps the OpenAI CLIP ViT-B/32 backbone frozen.
Only a small residual adapter and a cosine classifier are trained.  No test
images or external training data are used by this module.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_hidden_name(name: str) -> bool:
    return name.startswith(".") or name.startswith("__")


def _feature_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Unsupported CLIP feature output type: {type(output)!r}")


def effective_number_class_weights(class_counts: np.ndarray, beta: float, cap: float | None = None) -> np.ndarray:
    counts = np.maximum(np.asarray(class_counts, dtype=np.float64), 1.0)
    if beta <= 0.0 or beta >= 1.0:
        weights = np.ones_like(counts, dtype=np.float64)
    else:
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / weights.mean()
    if cap is not None and cap > 0:
        weights = np.clip(weights, 1.0 / cap, cap)
    return weights.astype(np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(root: str | Path) -> List[Tuple[str, int, str]]:
    """Return (path, integer label, class name) for folder-organized data."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir() and not _is_hidden_name(p.name)], key=lambda p: p.name)
    if not class_dirs:
        raise ValueError(f"Expected one subdirectory per class under {root}")
    rows: List[Tuple[str, int, str]] = []
    for label, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not any(
                _is_hidden_name(part) for part in path.relative_to(class_dir).parts
            ):
                rows.append((str(path), label, class_dir.name))
    if not rows:
        raise ValueError(f"No supported image files found under {root}")
    return rows


def list_test_images(root: str | Path) -> List[str]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {root}")
    paths = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(_is_hidden_name(part) for part in p.relative_to(root).parts)
    ]
    if not paths:
        raise ValueError(f"No supported image files found under {root}")
    return [str(p) for p in paths]


def stratified_split(rows: Sequence[Tuple[str, int, str]], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, (_, label, _) in enumerate(rows):
        by_class.setdefault(label, []).append(idx)
    train_idx, val_idx = [], []
    for indices in by_class.values():
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 0
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def safe_open_image(path: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:  # Pillow can read some truncated files but not all.
        print(f"[warning] failed to decode {path}: {exc}; using a black image")
        return Image.new("RGB", (224, 224), color=(0, 0, 0))


class FolderImageDataset(Dataset):
    def __init__(self, rows: Sequence[Tuple[str, int, str]], indices: Sequence[int], processor: CLIPProcessor):
        self.rows = rows
        self.indices = list(indices)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, object]:
        row_index = self.indices[item]
        path, label, class_name = self.rows[row_index]
        image = safe_open_image(path)
        encoded = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "row_index": torch.tensor(row_index, dtype=torch.long),
            "path": path,
            "class_name": class_name,
        }


class TestImageDataset(Dataset):
    def __init__(self, paths: Sequence[str], processor: CLIPProcessor):
        self.paths = list(paths)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, item: int) -> Dict[str, object]:
        path = self.paths[item]
        image = safe_open_image(path)
        encoded = self.processor(images=image, return_tensors="pt")
        return {"pixel_values": encoded["pixel_values"].squeeze(0), "path": path}


def make_prompts(class_names: Sequence[str]) -> List[str]:
    templates = [
        "a photo of a {}",
        "a close-up photo of a {}",
        "a wildlife photo of a {}",
        "a botanical photo of a {}",
        "a specimen photo of a {}",
    ]
    prompts = []
    for name in class_names:
        # Keep the provided folder name intact: many fine-grained labels are
        # scientific Latin names or official numeric identifiers.
        prompts.append(" || ".join(template.format(name) for template in templates))
    return prompts


@torch.no_grad()
def encode_class_text(model: CLIPModel, processor: CLIPProcessor, class_names: Sequence[str], device: torch.device) -> torch.Tensor:
    """Encode an averaged prompt ensemble, returning [num_classes, 512]."""
    features = []
    for prompt in make_prompts(class_names):
        texts = prompt.split(" || ")
        encoded = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = _feature_tensor(model.get_text_features(**encoded)).float()
        text_features = F.normalize(text_features, dim=-1).mean(dim=0)
        features.append(F.normalize(text_features, dim=0))
    return torch.stack(features, dim=0)


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int = 512, bottleneck: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features + self.scale.tanh() * self.net(features), dim=-1)


class RobustCLIPClassifier(nn.Module):
    def __init__(self, model: CLIPModel, text_features: torch.Tensor, bottleneck: int = 128):
        super().__init__()
        self.clip = model
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.clip.eval()
        dim = int(text_features.shape[-1])
        self.adapter = ResidualAdapter(dim=dim, bottleneck=bottleneck)
        self.classifier = nn.Parameter(F.normalize(text_features.clone(), dim=-1))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  # never update CLIP BatchNorm/dropout-like state
        return self

    def encode_image(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base = F.normalize(_feature_tensor(self.clip.get_image_features(pixel_values=pixel_values)).float(), dim=-1)
        adapted = self.adapter(base)
        return base, adapted

    def forward(self, pixel_values: torch.Tensor, text_features: torch.Tensor | None = None):
        base, adapted = self.encode_image(pixel_values)
        weights = F.normalize(self.classifier, dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        logits = scale * adapted @ weights.t()
        zero_logits = None
        if text_features is not None:
            zero_logits = scale.detach() * base @ F.normalize(text_features, dim=-1).t()
        return logits, zero_logits, base, adapted


@dataclass
class TrainConfig:
    model_dir: str
    train_dir: str
    output_dir: str = "outputs/robust_clip_v1"
    val_ratio: float = 0.1
    seed: int = 2026
    batch_size: int = 64
    workers: int = 4
    epochs: int = 8
    rounds: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-4
    bottleneck: int = 128
    clean_fraction: float = 0.8
    clean_fraction_schedule: Tuple[float, ...] | None = None
    score_momentum: float = 0.7
    weight_floor: float = 0.25
    weight_cap: float = 2.0
    class_weight_beta: float = 0.9999
    mae_weight: float = 0.25
    distill_weight: float = 0.1
    temperature: float = 2.0
    device: str = "auto"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def collect_zero_shot_scores(
    classifier: RobustCLIPClassifier,
    loader: Iterable[Dict[str, object]],
    text_features: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    classifier.eval()
    n = max((int(batch["row_index"].max().item()) for batch in loader), default=-1) + 1
    # The loader is consumed once; callers use a fresh loader when needed.
    scores = np.zeros(n, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["label"].numpy()
        idx = batch["row_index"].numpy()
        base = F.normalize(_feature_tensor(classifier.clip.get_image_features(pixel_values=pixels)).float(), dim=-1)
        logits = base @ F.normalize(text_features, dim=-1).t()
        probs = logits.softmax(dim=-1).cpu().numpy()
        scores[idx] = probs[np.arange(len(y)), y]
        labels[idx] = y
    return scores, labels


@torch.no_grad()
def collect_model_scores(
    classifier: RobustCLIPClassifier,
    loader: Iterable[Dict[str, object]],
    text_features: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    classifier.eval()
    rows = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["label"].to(device)
        idx = batch["row_index"].cpu().numpy()
        logits, _, _, _ = classifier(pixels, text_features)
        probs = logits.softmax(dim=-1)
        rows.extend(
            zip(
                idx.tolist(),
                probs.gather(1, y[:, None]).squeeze(1).cpu().numpy().tolist(),
                y.cpu().numpy().tolist(),
            )
        )
    if not rows:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    max_idx = max(row[0] for row in rows)
    scores = np.zeros(max_idx + 1, dtype=np.float32)
    labels = np.zeros(max_idx + 1, dtype=np.int64)
    for idx, score, label in rows:
        scores[idx] = score
        labels[idx] = label
    return scores, labels


def classwise_quality(
    rows: Sequence[Tuple[str, int, str]],
    scores: np.ndarray,
    clean_fraction: float,
    weight_floor: float,
    weight_gamma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert scores into class-balanced continuous weights and a clean mask."""
    if not 0.0 < clean_fraction <= 1.0:
        raise ValueError("clean_fraction must be in (0, 1]")
    if not 0.0 <= weight_floor < 1.0:
        raise ValueError("weight_floor must be in [0, 1)")
    labels = np.array([label for _, label, _ in rows], dtype=np.int64)
    quality = np.full(len(rows), weight_floor, dtype=np.float32)
    clean = np.zeros(len(rows), dtype=np.bool_)
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        order = indices[np.argsort(scores[indices])[::-1]]
        keep = max(1, int(math.ceil(len(order) * clean_fraction)))
        clean[order[:keep]] = True
        # Rank-based weights avoid a single global threshold favoring head classes.
        ranks = np.linspace(1.0, 0.0, num=len(order), endpoint=False, dtype=np.float32)
        quality[order] = weight_floor + (1.0 - weight_floor) * np.power(ranks, weight_gamma)
    return quality, clean


def build_optimizer(classifier: RobustCLIPClassifier, config: TrainConfig):
    params = [p for p in classifier.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def train_round(
    classifier: RobustCLIPClassifier,
    loader: Iterable[Dict[str, object]],
    text_features: torch.Tensor,
    quality: np.ndarray,
    class_counts: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    classifier.train()
    optimizer = build_optimizer(classifier, config)
    class_weight = torch.tensor(
        effective_number_class_weights(class_counts, config.class_weight_beta, config.weight_cap),
        dtype=torch.float32,
        device=device,
    )
    total_loss = total_correct = total_count = 0.0
    for _ in range(config.epochs):
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["label"].to(device)
            idx = batch["row_index"].numpy()
            w = torch.tensor(quality[idx], dtype=torch.float32, device=device)
            logits, zero_logits, _, _ = classifier(pixels, text_features)
            ce = F.cross_entropy(logits, y, reduction="none")
            p_y = logits.softmax(dim=-1).gather(1, y[:, None]).squeeze(1)
            mae = 1.0 - p_y
            weighted = w * class_weight[y]
            loss_cls = (weighted * (ce + config.mae_weight * mae)).sum() / weighted.sum().clamp_min(1e-6)
            loss_distill = torch.zeros((), device=device)
            if zero_logits is not None and config.distill_weight > 0:
                t = config.temperature
                loss_distill = F.kl_div(
                    F.log_softmax(logits / t, dim=-1),
                    F.softmax(zero_logits / t, dim=-1),
                    reduction="batchmean",
                ) * (t * t)
            loss = loss_cls + config.distill_weight * loss_distill
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in classifier.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            total_correct += float((logits.argmax(dim=-1) == y).sum().cpu())
            total_count += len(y)
    return {"loss": total_loss / max(total_count, 1.0), "noisy_label_accuracy": total_correct / max(total_count, 1.0)}


def save_checkpoint(
    path: str | Path,
    classifier: RobustCLIPClassifier,
    class_names: Sequence[str],
    config: TrainConfig,
    quality: np.ndarray,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": classifier.state_dict(),
            "class_names": list(class_names),
            "config": asdict(config),
            "quality_summary": {
                "n": int(len(quality)),
                "mean": float(quality.mean()) if len(quality) else 0.0,
                "p10": float(np.quantile(quality, 0.1)) if len(quality) else 0.0,
            },
        },
        path,
    )


def load_clip(model_dir: str, device: torch.device) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load the official CLIP checkpoint from either HF or SBERT layout."""
    root = Path(model_dir)
    nested = root / "0_CLIPModel"
    if (root / "config.json").is_file():
        load_dir = root
    elif (nested / "config.json").is_file():
        load_dir = nested
    else:
        raise FileNotFoundError(
            f"Cannot find CLIP config.json under {root} or {nested}. "
            "Pass the local openai/clip-vit-base-patch32 directory."
        )
    model = CLIPModel.from_pretrained(str(load_dir), local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(str(load_dir), local_files_only=True)
    model.eval()
    print(f"loaded_clip={load_dir} projection_dim={model.config.projection_dim} device={device}")
    return model, processor


def dump_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
