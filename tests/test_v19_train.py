"""V19 behavioral checks: routing, noise correction, LoRA and selection isolation."""
from copy import deepcopy
import json
import random
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPConfig, CLIPModel

from robust_clip import FolderImageDataset, trainable_state_dict
from train_v5 import clone_trainable
from train_v6 import build_optimizer as legacy_optimizer, make_v6_scheduler
from train_v19 import fit, run_epoch, load_resume, _restore_rng, save_epoch_logits
from v6_core import PrototypeSignals
from v8_neighbors import NeighborEvidence
from v12_runtime import BatchedFeatureQueue
from v19_core import (recipe_config, build_optimizer, stage_indices, pool_signature, should_audit,
                      dynamic_repair_plan, repaired_soft_loss, repair_weight, file_sha256)
from v19_model import (build_classifier, build_classifier_from_checkpoint, resolution_processor,
                       load_classifier_state, trainable_parameter_names)
from test_v11 import processor
from test_v12_runtime import Tiny


class DropoutTiny(Tiny):
    def forward(self, pixels, unused=None):
        return super().forward(F.dropout(pixels, p=.05, training=self.training), unused)


def clip():
    return CLIPModel(CLIPConfig(projection_dim=16,
        vision_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 12,
                       'num_attention_heads': 4, 'patch_size': 32, 'image_size': 224},
        text_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1,
                     'num_attention_heads': 4, 'vocab_size': 20}))


class V19TrainingTests(unittest.TestCase):
    def test_epoch_cache_records_both_dropout_values(self):
        values = recipe_config('rank32_dropout')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'epoch_01.pt').write_bytes(b'frozen-checkpoint')
            logits = np.eye(3, dtype=np.float32)
            save_epoch_logits(root, 1, logits, logits, np.arange(3), [7, 8, 9], config=values)
            with np.load(root / 'epoch_01_logits.npz', allow_pickle=False) as cached:
                self.assertEqual(int(cached['format_version']), 19)
                self.assertEqual(float(cached['lora_dropout']), .05)
                self.assertEqual(float(cached['mlp_lora_dropout']), .05)
                self.assertEqual(str(cached['checkpoint_sha256']), file_sha256(root / 'epoch_01.pt'))

    def test_resource_infeasible_or_mismatched_science_is_rejected_before_training(self):
        from train_v19 import bind_resource_plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resource = root / 'resource.json'; resource.write_text('{}')
            args = SimpleNamespace(resource_json=str(resource), recipe='rank32_dropout',
                                   batch_size=256, gradient_accumulation=1, workers=8,
                                   stage='validate', output_dir=str(root / 'run'))
            entry = {**recipe_config(args.recipe), 'workers': 8, 'status': 'runnable',
                     'run_dir': args.output_dir}
            with patch('v19_pipeline_support.read_resource_plan', return_value={'ok': True}), \
                 patch('v19_pipeline_support.resource_for_recipe', return_value=entry):
                self.assertEqual(bind_resource_plan(args), {'ok': True})
            for key, value in [('status', 'resource_infeasible'), ('lora_dropout', 0.),
                               ('mlp_lora_dropout', 0.), ('image_size', 384), ('lora_alpha', 32.)]:
                with self.subTest(key=key), \
                     patch('v19_pipeline_support.read_resource_plan', return_value={}), \
                     patch('v19_pipeline_support.resource_for_recipe', return_value={**entry, key: value}), \
                     self.assertRaises(ValueError):
                    bind_resource_plan(args)

    def test_training_update_preserves_v15_loss_and_ema_behavior(self):
        from train_v15 import run_epoch as baseline_epoch
        from benchmark_v19 import isolated_state

        n, dim, classes = 6, 8, 3
        torch.manual_seed(92)
        initial = Tiny()
        pixels = torch.randn(n, 3, 8, 8)
        labels = np.arange(n) % classes
        ambiguous = np.array([True, False, False, False, False, False])
        candidates = [(0, 1)] + [(int(y),) for y in labels[1:]]
        results = []
        for implementation in (baseline_epoch, run_epoch):
            torch.manual_seed(93)
            model = deepcopy(initial)
            optimizer, _, _ = legacy_optimizer(model)
            scheduler = make_v6_scheduler(optimizer, 24)
            queue = BatchedFeatureQueue(classes, 2, dim, torch.device('cpu'))
            queue.enqueue(torch.eye(classes, dim).repeat_interleave(2, 0),
                          torch.arange(classes).repeat_interleave(2), torch.ones(6, dtype=torch.bool))
            ema = clone_trainable(model)
            state = isolated_state(labels, ambiguous, classes)
            batch = {'pixel_values_a': pixels, 'pixel_values_b': pixels.flip(-1),
                     'label': torch.as_tensor(labels), 'row_index': torch.arange(n)}
            stats, step = implementation(
                classifier=model, loader=[batch], optimizer=optimizer, scheduler=scheduler,
                ema=ema, queue=queue, device=torch.device('cpu'), epoch=8,
                labels=labels, class_prior=torch.tensor([.2, .3, .5]),
                center_features=np.eye(classes, dim, dtype=np.float32)[labels],
                prototypes=torch.eye(classes, dim), ambiguous_mask=ambiguous,
                candidate_labels=candidates, gradient_accumulation=1, soft_teacher=True,
                dynamic_noise=True, schedule_epochs=24, optimizer_step=0, **state)
            results.append((deepcopy(model.state_dict()), ema, queue.state_dict(), stats, step))
        for index in (0, 1, 2):
            for key in results[0][index]:
                a, b = results[0][index][key], results[1][index][key]
                if torch.is_tensor(a):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                else:
                    self.assertEqual(a, b)
        self.assertEqual(results[0][3:], results[1][3:])

    def test_repaired_rows_ignore_old_label_and_detach_teacher(self):
        n, dim, classes = 3, 8, 3
        torch.manual_seed(5)
        initial = Tiny()
        pixels = torch.randn(n, 3, 8, 8)
        teacher = torch.tensor([[.05, .90, .05]] * n, requires_grad=True)
        a = torch.randn(n, classes, requires_grad=True)
        b = torch.randn(n, classes, requires_grad=True)
        repaired_soft_loss(a, b, teacher, torch.ones(n, dtype=torch.bool), .5).sum().backward()
        self.assertIsNone(teacher.grad)
        results = []
        for old_label in (0, 2):
            network = deepcopy(initial)
            optimizer, _, _ = legacy_optimizer(network)
            scheduler = make_v6_scheduler(optimizer, 24)
            queue = BatchedFeatureQueue(classes, 2, dim, torch.device('cpu'))
            batch = {'pixel_values_a': pixels, 'pixel_values_b': pixels,
                     'label': torch.full((n,), old_label), 'row_index': torch.arange(n)}
            stats, _ = run_epoch(network, [batch], optimizer, scheduler, clone_trainable(network), queue,
                torch.device('cpu'), 8, np.full(n, old_label), np.full(n, .3), np.ones(n, bool),
                torch.ones(classes) / classes, np.eye(n, dim, dtype=np.float32), torch.eye(classes, dim),
                np.ones(n, bool), teacher.detach().numpy().copy(), np.ones(n, bool), np.zeros(n, dtype=np.int64),
                np.zeros(n), np.zeros(n, dtype=np.int16), np.zeros(n), np.zeros(n, dtype=np.int16), 0,
                np.zeros(n, bool), [(old_label,)] * n, 1, True, dynamic_noise=True,
                audit_probabilities=teacher.detach().numpy(), schedule_epochs=24)
            self.assertEqual(stats['repaired_rows'], n)
            self.assertEqual(stats['soft_teacher_rows'], 0)
            self.assertFalse(queue.valid.any())
            results.append(network.state_dict())
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)


    def test_periodic_audit_and_exact_epoch_resume(self):
        cpu, n = torch.device('cpu'), 9
        labels = np.arange(n) % 3
        ones = np.ones(n, np.float32)
        proto = np.eye(3, 8, dtype=np.float32)
        signals = PrototypeSignals(proto, np.zeros(n), ones, ones, ones, labels.copy(), ones, ones)
        neighbor = NeighborEvidence(ones, labels.copy(), ones, ones, np.zeros(n))
        config = {**recipe_config('rank32_dropout'), 'version': 'v19', 'stage': 'validate', 'soft_teacher': True,
                  'precision': 'fp32', 'workers': 0, 'batch_size': 256, 'gradient_accumulation': 1,
                  'prefetch_factor': 2, 'eval_batch_size': 3}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = []
            for i in range(n):
                path = root / f'{i}.png'; Image.new('RGB', (330, 350), (i*20, 90, 30)).save(path)
                rows.append((str(path), int(labels[i]), f'{labels[i]:04d}'))
            prep = resolution_processor(processor(), config['image_size'])
            common = dict(processor=prep, rows=rows, train_indices=list(range(6)), supervised_indices=list(range(6)),
                labels=labels, view_features=np.stack([proto[labels]] * 4), signals=signals, quality=ones.copy()*.8,
                validation_loader=DataLoader(FolderImageDataset(rows, [6, 7, 8], prep), batch_size=3),
                trusted_validation=np.ones(n, bool), class_counts=np.array([2, 2, 2]),
                class_names=['0000', '0001', '0002'], config=config, dataset_digest='tiny', device=cpu,
                batch_size=256, workers=0, stage='validate', validation_indices=[6, 7, 8],
                ambiguous_mask=np.zeros(n, bool), candidate_labels=[(int(x),) for x in labels],
                sampler_mode='shuffle', row_repeat_factors=np.ones(n), gradient_accumulation=1, neighbor_evidence=neighbor)
            torch.manual_seed(77); initial = DropoutTiny()
            with patch('train_v19.build_optimizer', side_effect=lambda model, cfg: legacy_optimizer(model)):
                full = root / 'full'; full.mkdir()
                torch.manual_seed(123); random.seed(123)
                result = fit(classifier=deepcopy(initial), output_dir=full, resume=None, epochs=6, **deepcopy(common))
                split = root / 'split'; split.mkdir()
                torch.manual_seed(123); random.seed(123)
                fit(classifier=deepcopy(initial), output_dir=split, resume=None, epochs=4, **deepcopy(common))
                state = load_resume(str(split / 'resume_latest.pt'), config, 'tiny', cpu)
                resumed = fit(classifier=deepcopy(initial), output_dir=split, resume=state, epochs=6, **deepcopy(common))
                import train_v19
                save_complete = train_v19.save_resume
                interrupted = root / 'interrupted_first_epoch'; interrupted.mkdir()
                def fail_first_epoch(path, **kwargs):
                    if kwargs['epoch'] == 1:
                        raise RuntimeError('injected interruption before epoch-one resume commit')
                    return save_complete(path, **kwargs)
                torch.manual_seed(123); random.seed(123)
                with patch('train_v19.save_resume', side_effect=fail_first_epoch), \
                     self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                    fit(classifier=deepcopy(initial), output_dir=interrupted,
                        resume=None, epochs=6, **deepcopy(common))
                self.assertTrue((interrupted / 'best_model.pt').exists())
                epoch_zero = load_resume(str(interrupted / 'resume_latest.pt'), config, 'tiny', cpu)
                self.assertEqual(epoch_zero['epoch'], 0)
                restarted = fit(classifier=deepcopy(initial), output_dir=interrupted,
                                resume=epoch_zero, epochs=6, **deepcopy(common))
            for name, value in result[0].items():
                torch.testing.assert_close(value, resumed[0][name], rtol=0, atol=0)
                torch.testing.assert_close(value, restarted[0][name], rtol=0, atol=0)
            latest = torch.load(split / 'resume_latest.pt', map_location='cpu', weights_only=False)
            self.assertTrue(latest['audit_epoch2_seen'][:6].all())
            self.assertFalse(latest['audit_epoch2_seen'][6:].any())
            self.assertTrue((latest['audit_current_prob'][6:] == 0).all())
            self.assertTrue((latest['audit_previous_prob'][6:] == 0).all())
            self.assertEqual([h['epoch'] for h in resumed[-1] if h['quality_refreshed']], [3, 5])
            self.assertEqual(latest['trainable_parameter_names'], trainable_parameter_names(initial))
            for state_key in ('model', 'ema', 'best_state'):
                damaged = deepcopy(latest)
                del damaged[state_key][latest['trainable_parameter_names'][0]]
                with self.subTest(state=state_key), \
                     patch('train_v19.build_optimizer', side_effect=lambda model, cfg: legacy_optimizer(model)), \
                     self.assertRaisesRegex(RuntimeError, 'trainable state mismatch'):
                    fit(classifier=deepcopy(initial), output_dir=split, resume=damaged, epochs=7, **deepcopy(common))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_resume(str(split / 'resume_latest.pt'), {**config, 'stage': 'refit'}, 'tiny', cpu)
            with self.assertRaisesRegex(ValueError, 'configuration|batch_size'):
                load_resume(str(split / 'resume_latest.pt'),
                            {**config, 'batch_size': 128, 'gradient_accumulation': 2}, 'tiny', cpu)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_resume(str(split / 'resume_latest.pt'), {**config, 'workers': 8}, 'tiny', cpu)

    def test_refit_rejects_fallback_and_control_before_loading_data(self):
        import train_v19
        args = SimpleNamespace(batch_size=256, workers=0, gradient_accumulation=1,
                               prefetch_factor=1, eval_batch_size=256, stage='refit',
                               selection_json='selection.json', recipe='resolution384_rank32',
                               neighbor_cache=None, train_dir='data', output_dir='out',
                               conflict_policy='partial', sampler='repeat-factor')
        for recipe in ('expanded_mlp', 'matched_control', 'rank32', 'resolution384'):
            with patch('train_v19.parse_args', return_value=args), \
                 patch('select_v19.load_selection', return_value={'recipe': recipe}), \
                 patch('train_v19.load_manifest_dataset') as manifest, \
                 self.assertRaisesRegex(ValueError, 'winning candidate'):
                train_v19.main()
            manifest.assert_not_called()



if __name__ == "__main__": unittest.main()
