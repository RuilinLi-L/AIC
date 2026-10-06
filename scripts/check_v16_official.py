"""Two isolated updates and a strict round trip on the official V16 backbone.

Uses synthetic pixels only for engineering checks. No trained weights are saved
outside a temporary directory, and this is not an accuracy measurement.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from robust_clip import LoRALinear, load_clip, trainable_state_dict
from v16_core import RECIPES, build_optimizer, model_identity, recipe_config, layernorm_parameter_names
from v16_model import (build_classifier, build_classifier_from_checkpoint,
                       load_classifier_state, trainable_parameter_names)


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.clip.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--recipe', choices=RECIPES, default='mlp_light')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(2026)
    config = {**recipe_config(args.recipe), 'image_size': 320,
              'zoom_shortest_edge': 366, 'interpolate_pos_encoding': True,
              'base_model_identity': model_identity(args.model_dir)}
    base, _ = load_clip(args.model_dir, device)
    prototypes = np.random.default_rng(2026).normal(size=(750, 512)).astype(np.float32)
    model = build_classifier(base, prototypes, device, config)
    modules = [module for module in model.clip.modules() if isinstance(module, LoRALinear)]
    assert len(modules) == 72
    visual_count = sum(p.numel() for p in model.clip.parameters() if p.requires_grad)
    norm_names = layernorm_parameter_names(config)
    norm_before = {name: value.detach().clone() for name, value in model.named_parameters() if name in norm_names}
    assert len(norm_before) == (16 if args.recipe == 'mlp_light_ln' else 0)
    assert sum(value.numel() for value in norm_before.values()) == (12288 if norm_names else 0)
    assert visual_count == 2654208 + (12288 if norm_names else 0), visual_count
    digest_before = frozen_digest(model)
    optimizer, _, _ = build_optimizer(model, config)
    pixels = torch.randn(2, 3, 320, 320, device=device)
    targets = torch.tensor([0, 1], device=device)
    model.train()
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(pixels)[0].float(), targets)
        loss.backward()
        assert torch.isfinite(loss)
        for name, parameter in model.clip.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                if step == 1:
                    assert parameter.grad.abs().sum() > 0, name
            else:
                assert parameter.grad is None, name
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert digest_before == frozen_digest(model), 'official frozen weights changed'
    for name, before in norm_before.items():
        assert not torch.equal(before, dict(model.named_parameters())[name]), f'LayerNorm did not update: {name}'
    model.eval()
    with torch.no_grad():
        expected = model(pixels)[0]
    checkpoint = {'format_version': 16, 'config': config,
                  'class_names': [f'{i:04d}' for i in range(750)],
                  'trainable_parameter_names': trainable_parameter_names(model),
                  'model': trainable_state_dict(model)}
    with tempfile.TemporaryDirectory(prefix='v16-official-check-') as directory:
        path = Path(directory) / 'model.pt'
        torch.save(checkpoint, path)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        restored, _ = build_classifier_from_checkpoint(saved, args.model_dir, device)
        with torch.no_grad():
            actual = restored(pixels)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        missing_keys = [next(key for key in saved['model'] if '.mlp.fc1.lora_b' in key)]
        missing_keys += sorted(norm_names)
        for key in missing_keys:
            missing = dict(saved['model'])
            del missing[key]
            try:
                load_classifier_state(restored, missing)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'missing trainable weight was accepted: {key}')
    report = {'success': True, 'recipe': args.recipe, 'lora_modules': 72, 'visual_trainable_parameters': visual_count,
              'layernorm_tensors': len(norm_names), 'layernorm_parameters': sum(v.numel() for v in norm_before.values()),
              'all_selected_layernorms_updated': True,
              'optimizer_steps': 2, 'all_attention_and_mlp_gradients_nonzero': True,
              'frozen_backbone_unchanged': True, 'roundtrip_logits_exact': True,
              'missing_mlp_weight_rejected': True, 'losses': losses,
              'device': str(device), 'scope': 'engineering check; synthetic pixels; no accuracy claim'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
