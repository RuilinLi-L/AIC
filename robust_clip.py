"""Robust fine-tuning utilities for the AIC noisy-label challenge.

The default V3 path keeps CLIP frozen.  The V4 path can additionally insert
small LoRA branches into the final vision-transformer blocks while preserving
the official OpenAI CLIP ViT-B/32 backbone and checkpoint.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms


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


def _clip_image_transform(processor: CLIPProcessor, augmentation: bool | str = False):
    """Build the float32 torchvision transform used by all image datasets."""
    augmentation = "light" if augmentation is True else ("none" if not augmentation else str(augmentation))
    image_processor = processor.image_processor
    mean = image_processor.image_mean
    std = image_processor.image_std
    normalize = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    if augmentation == "none":
        shortest = int(image_processor.size.get("shortest_edge", 224))
        crop_height = int(image_processor.crop_size.get("height", shortest))
        crop_width = int(image_processor.crop_size.get("width", shortest))
        return transforms.Compose(
            [
                transforms.Resize(shortest, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop((crop_height, crop_width)),
                *normalize,
            ]
        )
    if augmentation == "light":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.90, 1.0),
                    ratio=(0.95, 1.05),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                *normalize,
            ]
        )
    if augmentation == "strong":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.70, 1.0),
                    ratio=(0.85, 1.15),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                *normalize,
            ]
        )
    raise ValueError(f"Unknown augmentation preset: {augmentation}")


class FolderImageDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Tuple[str, int, str]],
        indices: Sequence[int],
        processor: CLIPProcessor,
        augment: bool | str = False,
    ):
        self.rows = rows
        self.indices = list(indices)
        self.transform = _clip_image_transform(processor, augment)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, object]:
        row_index = self.indices[item]
        path, label, class_name = self.rows[row_index]
        image = safe_open_image(path)
        return {
            # torchvision stays in float32 throughout; the slow HF processor
            # temporarily upcasts every image to float64 and can exhaust RAM
            # in long Windows multi-worker runs.
            "pixel_values": self.transform(image),
            "label": torch.tensor(label, dtype=torch.long),
            "row_index": torch.tensor(row_index, dtype=torch.long),
            "path": path,
            "class_name": class_name,
        }


class PairedFolderImageDataset(Dataset):
    """Return two independently augmented views of each folder image."""

    def __init__(
        self,
        rows: Sequence[Tuple[str, int, str]],
        indices: Sequence[int],
        processor: CLIPProcessor,
        augment: bool | str = "light",
    ):
        self.rows = rows
        self.indices = list(indices)
        # Separate transform objects make the independence of random crops
        # explicit and also keep this dataset safe for Windows worker spawn.
        self.transform_a = _clip_image_transform(processor, augment)
        self.transform_b = _clip_image_transform(processor, augment)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, object]:
        row_index = self.indices[item]
        path, label, class_name = self.rows[row_index]
        image = safe_open_image(path)
        return {
            "pixel_values_a": self.transform_a(image),
            "pixel_values_b": self.transform_b(image),
            "label": torch.tensor(label, dtype=torch.long),
            "row_index": torch.tensor(row_index, dtype=torch.long),
            "path": path,
            "class_name": class_name,
        }


class FeatureDataset(Dataset):
    """Dataset over cached frozen CLIP features."""

    def __init__(
        self,
        rows: Sequence[Tuple[str, int, str]],
        indices: Sequence[int],
        features: np.ndarray,
    ):
        if len(features) != len(rows):
            raise ValueError("features and rows must have the same length")
        self.rows = rows
        self.indices = list(indices)
        self.features = features

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, object]:
        row_index = self.indices[item]
        path, label, class_name = self.rows[row_index]
        return {
            # np.memmap slices are read-only; tensor() makes a small writable
            # copy and avoids undefined-behavior warnings from torch.as_tensor.
            "features": torch.tensor(self.features[row_index], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "row_index": torch.tensor(row_index, dtype=torch.long),
            "path": path,
            "class_name": class_name,
        }


class FeatureOnlyBackbone(nn.Module):
    """Minimal backbone placeholder when all image features are cached."""

    def __init__(self, projection_dim: int):
        super().__init__()
        self.config = SimpleNamespace(projection_dim=int(projection_dim))

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values


class TestImageDataset(Dataset):
    def __init__(self, paths: Sequence[str], processor: CLIPProcessor):
        self.paths = list(paths)
        image_processor = processor.image_processor
        shortest = int(image_processor.size.get("shortest_edge", 224))
        crop_height = int(image_processor.crop_size.get("height", shortest))
        crop_width = int(image_processor.crop_size.get("width", shortest))
        self.transform = transforms.Compose(
            [
                transforms.Resize(shortest, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop((crop_height, crop_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, item: int) -> Dict[str, object]:
        path = self.paths[item]
        image = safe_open_image(path)
        return {"pixel_values": self.transform(image), "path": path}


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


class LoRALinear(nn.Module):
    """Frozen linear projection plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = float(alpha) / float(rank)
        self.dropout = float(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual_inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        residual = F.linear(F.linear(residual_inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + self.scaling * residual


def inject_visual_lora(
    model: CLIPModel,
    last_n_layers: int = 4,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: Sequence[str] = ("q_proj", "v_proj"),
) -> List[str]:
    """Insert LoRA into selected projections of the last CLIP vision blocks."""
    layers = model.vision_model.encoder.layers
    if not 1 <= last_n_layers <= len(layers):
        raise ValueError(f"last_n_layers must be in [1, {len(layers)}]")
    supported = {"q_proj", "k_proj", "v_proj", "out_proj"}
    unknown = set(targets) - supported
    if unknown:
        raise ValueError(f"Unsupported LoRA targets: {sorted(unknown)}")
    replaced: List[str] = []
    start = len(layers) - last_n_layers
    for layer_id in range(start, len(layers)):
        attention = layers[layer_id].self_attn
        for target in targets:
            base = getattr(attention, target)
            if isinstance(base, LoRALinear):
                continue
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected Linear at vision layer {layer_id}.{target}, got {type(base)!r}")
            setattr(attention, target, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(f"vision_model.encoder.layers.{layer_id}.self_attn.{target}")
    return replaced


def inject_lora_from_config(model: CLIPModel, config: Dict[str, object]) -> List[str]:
    """Recreate the LoRA structure recorded in a checkpoint config."""
    rank = int(config.get("lora_rank", 0))
    if rank <= 0:
        return []
    raw_targets = config.get("lora_targets", ("q_proj", "v_proj"))
    if isinstance(raw_targets, str):
        targets = tuple(part.strip() for part in raw_targets.split(",") if part.strip())
    else:
        targets = tuple(raw_targets)
    return inject_visual_lora(
        model,
        last_n_layers=int(config.get("lora_layers", 4)),
        rank=rank,
        alpha=float(config.get("lora_alpha", rank * 2)),
        dropout=float(config.get("lora_dropout", 0.0)),
        targets=targets,
    )


class RobustCLIPClassifier(nn.Module):
    def __init__(
        self,
        model: CLIPModel,
        text_features: torch.Tensor | None = None,
        bottleneck: int = 128,
        classifier_init: torch.Tensor | None = None,
        initial_logit_scale: float = 30.0,
        max_logit_scale: float = 100.0,
        learn_logit_scale: bool = True,
    ):
        super().__init__()
        self.clip = model
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.clip.eval()
        if classifier_init is not None:
            dim = int(classifier_init.shape[-1])
        elif text_features is not None:
            dim = int(text_features.shape[-1])
        else:
            dim = int(model.config.projection_dim)
        self.adapter = ResidualAdapter(dim=dim, bottleneck=bottleneck)
        if classifier_init is None:
            if text_features is None:
                classifier_init = torch.randn(1, dim)
            else:
                classifier_init = text_features
        self.classifier = nn.Parameter(F.normalize(classifier_init.clone(), dim=-1))
        if not 1.0 <= initial_logit_scale <= max_logit_scale:
            raise ValueError("initial_logit_scale must be in [1, max_logit_scale]")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(initial_logit_scale)))
        self.logit_scale.requires_grad_(learn_logit_scale)
        self.max_logit_scale = float(max_logit_scale)

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  # preserve the pretrained model's deterministic state
        # LoRA dropout remains independently controllable even though the
        # frozen backbone itself stays in eval mode.
        for module in self.clip.modules():
            if isinstance(module, LoRALinear):
                module.train(mode)
        return self

    def has_trainable_backbone(self) -> bool:
        return any(parameter.requires_grad for parameter in self.clip.parameters())

    def encode_image(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.has_trainable_backbone():
            base = F.normalize(_feature_tensor(self.clip.get_image_features(pixel_values=pixel_values)).float(), dim=-1)
        else:
            with torch.no_grad():
                base = F.normalize(_feature_tensor(self.clip.get_image_features(pixel_values=pixel_values)).float(), dim=-1)
        adapted = self.adapter(base)
        return base, adapted

    def forward_features(self, base: torch.Tensor, text_features: torch.Tensor | None = None):
        base = F.normalize(base.float(), dim=-1)
        adapted = self.adapter(base)
        weights = F.normalize(self.classifier, dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, self.max_logit_scale)
        logits = scale * adapted @ weights.t()
        zero_logits = None
        if text_features is not None:
            zero_logits = scale.detach() * base @ F.normalize(text_features, dim=-1).t()
        return logits, zero_logits, base, adapted

    def forward(self, pixel_values: torch.Tensor, text_features: torch.Tensor | None = None):
        if self.has_trainable_backbone():
            base = F.normalize(_feature_tensor(self.clip.get_image_features(pixel_values=pixel_values)).float(), dim=-1)
        else:
            with torch.no_grad():
                base = F.normalize(_feature_tensor(self.clip.get_image_features(pixel_values=pixel_values)).float(), dim=-1)
        return self.forward_features(base, text_features)


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
    low_confidence_multiplier: float = 1.0
    weight_cap: float = 2.0
    class_weight_beta: float = 0.9999
    mae_weight: float = 0.25
    loss_mode: str = "soft_ce"
    gce_q: float = 0.7
    distill_weight: float = 0.0
    drift_weight: float = 0.0
    augmentation: str = "none"
    prototype_keep_fraction: float = 0.7
    prototype_iterations: int = 2
    initial_logit_scale: float = 30.0
    max_logit_scale: float = 50.0
    learn_logit_scale: bool = False
    temperature: float = 2.0
    device: str = "auto"
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_layers: int = 4
    lora_targets: Tuple[str, ...] = ("q_proj", "v_proj")
    lora_lr: float = 2e-5
    head_lr: float = 1e-4
    anchor_weight: float = 0.1
    label_smoothing: float = 0.03
    pseudo_start_epoch: int = 2
    pseudo_threshold: float = 0.7
    pseudo_weight: float = 0.6
    temporal_decay: float = 0.8
    ema_decay: float = 0.995
    gradient_accumulation: int = 1
    tune_layernorm: bool = False
    tune_visual_projection: bool = False
    trusted_val_fraction: float = 0.7
    consistency_weight: float = 0.05
    consistency_temperature: float = 2.0
    prototype_pseudo_start_epoch: int = 2
    prototype_pseudo_margin: float = 0.08
    prototype_pseudo_weight: float = 0.35
    prototype_teacher_mix: float = 0.5
    prototype_temperature: float = 0.07
    full_refit_epochs: int = 2


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


@torch.no_grad()
def collect_frozen_features(
    model: CLIPModel,
    loader: Iterable[Dict[str, object]],
    num_samples: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode images once with frozen CLIP, preserving global row indices."""
    dim = int(model.config.projection_dim)
    features = np.zeros((num_samples, dim), dtype=np.float16)
    labels = np.full(num_samples, -1, dtype=np.int64)
    seen = np.zeros(num_samples, dtype=np.bool_)
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        idx = batch["row_index"].cpu().numpy()
        y = batch["label"].cpu().numpy()
        base = F.normalize(_feature_tensor(model.get_image_features(pixel_values=pixels)).float(), dim=-1)
        features[idx] = base.cpu().numpy().astype(np.float16)
        labels[idx] = y
        seen[idx] = True
    if not seen.all():
        missing = int((~seen).sum())
        raise ValueError(f"Frozen feature loader did not cover {missing} rows")
    return features, labels


def robust_visual_prototypes_and_scores(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    candidate_indices: Sequence[int] | None = None,
    keep_fraction: float = 0.7,
    iterations: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build trimmed visual prototypes and per-sample cosine scores.

    The challenge folders expose numeric IDs rather than species names, so a
    text prompt such as ``a photo of a 0001`` is not a useful semantic prior.
    Iterative within-class trimming reduces contamination from noisy folders.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    feat = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    candidates = np.arange(len(labels), dtype=np.int64) if candidate_indices is None else np.asarray(candidate_indices, dtype=np.int64)
    selected = candidates.copy()
    scores = np.full(len(labels), np.nan, dtype=np.float32)
    prototypes = np.zeros((num_classes, feat.shape[1]), dtype=np.float32)
    for iteration in range(iterations):
        prototypes.fill(0.0)
        counts = np.zeros(num_classes, dtype=np.int64)
        np.add.at(prototypes, labels[selected], feat[selected])
        np.add.at(counts, labels[selected], 1)
        if (counts == 0).any():
            missing = np.where(counts == 0)[0].tolist()
            raise ValueError(f"No prototype candidates for classes: {missing[:10]}")
        prototypes /= counts[:, None]
        prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
        scores[candidates] = np.sum(feat[candidates] * prototypes[labels[candidates]], axis=1)
        if iteration + 1 < iterations and keep_fraction < 1.0:
            retained = []
            for label in range(num_classes):
                indices = candidates[labels[candidates] == label]
                keep = max(1, int(math.ceil(len(indices) * keep_fraction)))
                order = indices[np.argsort(scores[indices])[::-1]]
                retained.append(order[:keep])
            selected = np.concatenate(retained)
    return prototypes, scores


def prototype_pseudo_targets(
    features: np.ndarray,
    prototypes: np.ndarray,
    temperature: float = 0.07,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prototype top-1 labels, confidence and top-1/top-2 margin.

    Prototypes are expected to be built from the training split only.  The
    helper returns compact screening signals; V5 computes a soft teacher for
    eligible mini-batches on the GPU instead of storing a huge probability
    matrix for the complete training set.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    feat = np.asarray(features, dtype=np.float32)
    proto = np.asarray(prototypes, dtype=np.float32)
    if feat.ndim != 2 or proto.ndim != 2 or feat.shape[1] != proto.shape[1]:
        raise ValueError("features and prototypes must be rank-2 arrays with matching dimensions")
    scores = (feat @ proto.T).astype(np.float32) / float(temperature)
    pseudo = scores.argmax(axis=1).astype(np.int64)
    top = np.partition(scores, -2, axis=1)[:, -2:]
    top.sort(axis=1)
    top1 = top[:, 1]
    top2 = top[:, 0]
    shifted = scores - top1[:, None]
    confidence = (1.0 / np.exp(shifted).sum(axis=1)).astype(np.float32)
    margin = (top1 - top2).astype(np.float32)
    return pseudo, confidence, margin


@torch.no_grad()
def collect_visual_prototypes_and_scores(
    model: CLIPModel,
    loader: Iterable[Dict[str, object]],
    num_classes: int,
    num_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Compatibility wrapper using all rows and an untrimmed prototype."""
    features, labels = collect_frozen_features(model, loader, num_samples, device)
    prototypes, scores = robust_visual_prototypes_and_scores(
        features, labels, num_classes, keep_fraction=1.0, iterations=1
    )
    return torch.from_numpy(prototypes).to(device=device, dtype=torch.float32), scores


def classwise_rank_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    candidate_indices: Sequence[int],
) -> np.ndarray:
    """Normalize heterogeneous screening signals to comparable within-class ranks."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    ranked = np.full(len(labels), np.nan, dtype=np.float32)
    for label in np.unique(labels[candidates]):
        indices = candidates[labels[candidates] == label]
        order = indices[np.argsort(scores[indices])]
        ranked[order] = np.linspace(0.0, 1.0, len(order), dtype=np.float32)
    return ranked


def classwise_quality(
    rows: Sequence[Tuple[str, int, str]],
    scores: np.ndarray,
    clean_fraction: float,
    weight_floor: float,
    weight_gamma: float = 2.0,
    candidate_indices: Sequence[int] | None = None,
    low_confidence_multiplier: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert scores into class-balanced continuous weights and a clean mask."""
    if not 0.0 < clean_fraction <= 1.0:
        raise ValueError("clean_fraction must be in (0, 1]")
    if not 0.0 <= weight_floor < 1.0:
        raise ValueError("weight_floor must be in [0, 1)")
    if not 0.0 <= low_confidence_multiplier <= 1.0:
        raise ValueError("low_confidence_multiplier must be in [0, 1]")
    labels = np.array([label for _, label, _ in rows], dtype=np.int64)
    quality = np.zeros(len(rows), dtype=np.float32)
    clean = np.zeros(len(rows), dtype=np.bool_)
    candidates = np.arange(len(rows), dtype=np.int64) if candidate_indices is None else np.asarray(candidate_indices, dtype=np.int64)
    for label in np.unique(labels[candidates]):
        indices = candidates[labels[candidates] == label]
        order = indices[np.argsort(scores[indices])[::-1]]
        keep = max(1, int(math.ceil(len(order) * clean_fraction)))
        clean[order[:keep]] = True
        # Rank-based weights avoid a single global threshold favoring head classes.
        ranks = np.linspace(1.0, 0.0, num=len(order), endpoint=False, dtype=np.float32)
        quality[order] = weight_floor + (1.0 - weight_floor) * np.power(ranks, weight_gamma)
        # Optionally reduce low-confidence gradient mass. The conservative
        # default keeps it at 1; hard loss modes still use the clean mask.
        quality[order[keep:]] *= low_confidence_multiplier
    return quality, clean


def build_optimizer(classifier: RobustCLIPClassifier, config: TrainConfig):
    params = [p for p in classifier.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def generalized_cross_entropy(logits: torch.Tensor, y: torch.Tensor, q: float) -> torch.Tensor:
    if not 0.0 < q <= 1.0:
        raise ValueError("gce_q must be in (0, 1]")
    p_y = logits.softmax(dim=-1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
    return (1.0 - p_y.pow(q)) / q


def symmetric_kl_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Return a per-sample symmetric KL divergence for two views."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scale = float(temperature)
    log_prob_a = F.log_softmax(logits_a / scale, dim=-1)
    log_prob_b = F.log_softmax(logits_b / scale, dim=-1)
    prob_a = log_prob_a.exp()
    prob_b = log_prob_b.exp()
    forward = F.kl_div(log_prob_a, prob_b, reduction="none").sum(dim=-1)
    reverse = F.kl_div(log_prob_b, prob_a, reduction="none").sum(dim=-1)
    return 0.5 * (forward + reverse) * (scale * scale)


def train_round(
    classifier: RobustCLIPClassifier,
    loader: Iterable[Dict[str, object]],
    text_features: torch.Tensor | None,
    quality: np.ndarray,
    clean: np.ndarray | None,
    class_counts: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epochs: int | None = None,
) -> Dict[str, float]:
    classifier.train()
    if optimizer is None:
        optimizer = build_optimizer(classifier, config)
    class_weight = torch.tensor(
        effective_number_class_weights(class_counts, config.class_weight_beta, config.weight_cap),
        dtype=torch.float32,
        device=device,
    )
    total_loss = total_correct = total_count = 0.0
    num_epochs = config.epochs if epochs is None else epochs
    for _ in range(num_epochs):
        for batch in loader:
            y = batch["label"].to(device)
            idx = batch["row_index"].numpy()
            w = torch.as_tensor(quality[idx], dtype=torch.float32, device=device)
            clean_batch = torch.ones(len(idx), dtype=torch.bool, device=device)
            if clean is not None:
                clean_batch = torch.as_tensor(clean[idx], dtype=torch.bool, device=device)
            if "features" in batch:
                base_features = batch["features"].to(device, non_blocking=True)
                logits, zero_logits, base, adapted = classifier.forward_features(base_features, text_features)
            else:
                pixels = batch["pixel_values"].to(device, non_blocking=True)
                logits, zero_logits, base, adapted = classifier(pixels, text_features)
            ce = F.cross_entropy(logits, y, reduction="none")
            p_y = logits.softmax(dim=-1).gather(1, y[:, None]).squeeze(1)
            mae = 1.0 - p_y
            weighted = w * class_weight[y]
            if config.loss_mode == "soft_ce":
                # Keep useful gradients for every sample; screening acts through
                # continuous weights rather than a brittle hard loss switch.
                sample_loss = ce + config.mae_weight * mae
            elif config.loss_mode == "hard_gce":
                gce = (1.0 - p_y.clamp_min(1e-7).pow(config.gce_q)) / config.gce_q
                sample_loss = torch.where(clean_batch, ce + config.mae_weight * mae, gce)
            elif config.loss_mode == "hard_mae":
                sample_loss = torch.where(clean_batch, ce + config.mae_weight * mae, mae)
            else:
                raise ValueError(f"Unknown loss_mode: {config.loss_mode}")
            loss_cls = (weighted * sample_loss).sum() / weighted.sum().clamp_min(1e-6)
            loss_distill = torch.zeros((), device=device)
            if zero_logits is not None and config.distill_weight > 0:
                t = config.temperature
                loss_distill = F.kl_div(
                    F.log_softmax(logits / t, dim=-1),
                    F.softmax(zero_logits / t, dim=-1),
                    reduction="batchmean",
                ) * (t * t)
            loss_drift = (1.0 - (base.detach() * adapted).sum(dim=-1)).mean()
            loss = loss_cls + config.distill_weight * loss_distill + config.drift_weight * loss_drift
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in classifier.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            total_correct += float((logits.argmax(dim=-1) == y).sum().cpu())
            total_count += len(y)
    return {"loss": total_loss / max(total_count, 1.0), "noisy_label_accuracy": total_correct / max(total_count, 1.0)}


@torch.no_grad()
def evaluate_classifier(
    classifier: RobustCLIPClassifier,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
) -> Dict[str, float]:
    classifier.eval()
    total_loss = total_correct = total_count = 0.0
    for batch in loader:
        y = batch["label"].to(device)
        if "features" in batch:
            logits, _, _, _ = classifier.forward_features(batch["features"].to(device, non_blocking=True), None)
        else:
            logits, _, _, _ = classifier(batch["pixel_values"].to(device, non_blocking=True), None)
        total_loss += float(F.cross_entropy(logits, y, reduction="sum").cpu())
        total_correct += float((logits.argmax(dim=-1) == y).sum().cpu())
        total_count += len(y)
    return {
        "loss": total_loss / max(total_count, 1.0),
        "accuracy": total_correct / max(total_count, 1.0),
    }


def trainable_state_dict(classifier: RobustCLIPClassifier) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, value in classifier.named_parameters() if value.requires_grad}
    return {
        name: value.detach().cpu().clone()
        for name, value in classifier.state_dict().items()
        if not name.startswith("clip.") or name in trainable_names
    }


def load_classifier_state(classifier: RobustCLIPClassifier, state: Dict[str, torch.Tensor]) -> None:
    incompatible = classifier.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_clip = [name for name in incompatible.missing_keys if not name.startswith("clip.")]
    if unexpected or missing_non_clip:
        raise RuntimeError(
            f"Checkpoint mismatch: unexpected={unexpected[:5]} missing_non_clip={missing_non_clip[:5]}"
        )


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
            "format_version": 3 if config.lora_rank > 0 else 2,
            "model": trainable_state_dict(classifier),
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
