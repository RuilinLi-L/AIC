"""Independent scheduler integration, queue boundaries and restart behavior."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from v20_pipeline import Pipeline
from v20_pipeline_support import benchmark_status, official_check, RESOURCE_SPECS


def completed(run):
    run.mkdir(parents=True, exist_ok=True)
    (run / 'validation_epochs').mkdir(exist_ok=True)
    (run / 'validation_epochs' / 'epoch_24.pt').touch()
    (run / 'metrics.json').write_text(json.dumps({'stage': 'validate', 'training_history': [{'epoch': n} for n in range(1, 25)]}))


class RecordingPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.leases = []
    @contextmanager
    def lease(self, label):
        self.leases.append(label)
        yield 3
    def run(self, args, log, gpu=None):
        self.calls.append((args, Path(log), gpu))
        if args[0] in ('train_v20.py', 'refit_legacy_v20.py'):
            out = Path(args[args.index('--output-dir') + 1])
            if args[args.index('--stage') + 1] == 'validate':
                completed(out)
            else:
                out.mkdir(parents=True, exist_ok=True)
                (out / 'model.pt').touch()
        elif args[0] == 'evaluate_tta_v20.py':
            out = Path(args[args.index('--output-dir') + 1])
            assert not out.exists(), 'log creation must not claim the evaluation output directory'
            assert not Path(log).is_relative_to(out)
            out.mkdir(parents=True)
            (out / 'strict_eval.json').write_text('{}')
            (out / 'model.pt').touch()
        elif args[0] == 'select_v20.py':
            Path(args[args.index('--output') + 1]).write_text('{}')


class SchedulerTests(unittest.TestCase):
    def setup_pipeline(self, directory):
        root = Path(directory)
        project, work = root / 'snapshot', root / 'v20'
        project.mkdir(); work.mkdir(); (work / 'pipeline').mkdir()
        (work / 'pipeline' / 'resource_plan.json').write_text('{}')
        env = dict(AIC_PROJECT_DIR=str(project), AIC_OUTPUT_ROOT=str(root), AIC_V20_ROOT=str(work),
                   AIC_V19_RESOURCE_JSON=str(root / 'v19.json'), AIC_MODEL_DIR=str(root / 'official'),
                   AIC_DATA_DIR=str(root / 'data'), AIC_MANIFEST=str(root / 'manifest.json'),
                   AIC_FEATURE_CACHE=str(root / 'frozen.npy'), AIC_BENCHMARK_METADATA=str(root / 'metadata.pt'))
        return root, work, env
    def plan(self, root, work):
        candidates = {key: {'recipe': recipe, 'run_dir': str(work / recipe), 'status': 'runnable'}
                      for key, recipe in [('candidate_a', 'dora_rank32'), ('candidate_b', 'loraplus_rank32')]}
        deps = {recipe: {'recipe': recipe, 'run_dir': str(root / 'v19' / recipe), 'status': 'runnable'}
                for recipe in ('resolution384_rank32', 'rank32_dropout')}
        control = {'recipe': 'rank32_control', 'run_dir': str(root / 'v19' / 'rank32_control'), 'status': 'runnable'}
        for entry in [control, *deps.values()]: completed(Path(entry['run_dir']))
        return {'candidates': candidates, 'dependencies': deps, 'control': control,
                'historical_refit': {'directory': str(root / 'historical')}}
    def test_two_candidates_all_three_baselines_and_only_winner_refit(self):
        for version, recipe in ((20, 'loraplus_rank32'), (19, 'rank32_dropout')):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root, work, env = self.setup_pipeline(directory)
                plan = self.plan(root, work)
                selected = {'refit_required': True, 'source_checkpoint': str(work / 'selected.pt'),
                            'source_format_version': version, 'recipe': recipe,
                            'training_config': {'resource_json': str(root / 'v19.json')}}
                with patch.dict(os.environ, env), patch('v20_pipeline.read_resource_plan', return_value=plan), \
                     patch('select_v20.read_decision', return_value=selected), patch('v20_pipeline.export_submission') as export:
                    pipeline = RecordingPipeline(); pipeline.execute()
                    trains = [args for args, _, _ in pipeline.calls if args[0] in ('train_v20.py', 'refit_legacy_v20.py')]
                    self.assertEqual(len(trains), 3)
                    refits = [args for args in trains if args[args.index('--stage') + 1] == 'refit']
                    self.assertEqual(len(refits), 1)
                    self.assertEqual(refits[0][0], 'train_v20.py' if version == 20 else 'refit_legacy_v20.py')
                    self.assertEqual(refits[0][refits[0].index('--recipe') + 1], recipe)
                    self.assertEqual(len([args for args, _, _ in pipeline.calls if args[0] == 'evaluate_tta_v20.py']), 5)
                    self.assertEqual(export.call_count, 2)
                    before = len(pipeline.calls); pipeline.execute()
                    # Existing frozen selection bypasses all training/evaluation and selection rewriting.
                    self.assertTrue(all(args[0] == 'predict_v20.py' for args, _, _ in pipeline.calls[before:]))
                    self.assertEqual(json.loads((work / 'pipeline' / 'status.json').read_text())['status'], 'complete')
    def test_missing_v19_dependency_never_silently_disappears(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work, env = self.setup_pipeline(directory)
            with patch.dict(os.environ, env):
                pipeline = RecordingPipeline(); pipeline.timeout = 0
                with self.assertRaisesRegex(TimeoutError, 'no silent omission'):
                    pipeline.legacy_evaluation({'recipe': 'rank32_dropout', 'run_dir': str(root / 'unfinished'), 'status': 'runnable'})
                self.assertFalse(pipeline.calls)
    def test_gpu_lease_checks_only_0_to_3_and_requires_70gib(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work, env = self.setup_pipeline(directory)
            visited = []
            def smi(args, **kwargs):
                gpu = int(next(arg[5:] for arg in args if arg.startswith('--id=')))
                visited.append(gpu)
                return SimpleNamespace(stdout='71680\n' if gpu == 3 else '71679\n')
            with patch.dict(os.environ, env), patch('v20_pipeline.subprocess.run', side_effect=smi):
                pipeline = Pipeline()
                with pipeline.lease('test') as gpu: self.assertEqual(gpu, 3)
                self.assertEqual(visited, [0, 1, 2, 3])
    def test_no_smaller_batch_retry_after_preflight_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work, env = self.setup_pipeline(directory)
            with patch.dict(os.environ, env):
                pipeline = RecordingPipeline()
                with patch.object(pipeline, 'run', side_effect=subprocess.CalledProcessError(1, 'probe')) as run:
                    with self.assertRaises(subprocess.CalledProcessError):
                        pipeline.guarded(pipeline.preflight, 'dora_rank32_preflight', 'dora_rank32')
                    self.assertEqual(run.call_count, 1)
                    status = json.loads((work / 'pipeline' / 'dora_rank32_preflight_status.json').read_text())
                    self.assertEqual(status['status'], 'failed')
    def test_completion_requires_whole_history_and_final_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory); completed(run)
            self.assertTrue(Pipeline.complete_training(run))
            (run / 'validation_epochs' / 'epoch_24.pt').unlink()
            self.assertFalse(Pipeline.complete_training(run))


class ProbeContractTests(unittest.TestCase):
    def report(self, recipe):
        return {'format_version': 20, 'recipe': recipe, **RESOURCE_SPECS[recipe],
                'initialization': 'official_base_only', 'trained_checkpoint_weights_loaded': False,
                'batch_size': 256, 'gradient_accumulation': 1, 'success': True,
                'chosen_workers': 8, 'pin_memory': False, 'prefetch_factor': 1, 'activation_checkpointing': False}
    def test_only_typed_actual_batch_oom_is_resource_infeasible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'benchmark.json'
            report = self.report('dora_rank32'); path.write_text(json.dumps(report))
            self.assertEqual(benchmark_status(path, 'dora_rank32'), 'runnable')
            report.update(success=False, status='cuda_oom', configurations=[{'batch_size': 256, 'status': 'cuda_oom'}])
            path.write_text(json.dumps(report))
            self.assertEqual(benchmark_status(path, 'dora_rank32'), 'resource_infeasible')
            for change in ({'batch_size': 128}, {'status': 'worker_error'}, {'configurations': []}):
                path.write_text(json.dumps({**report, **change}))
                with self.assertRaises(ValueError): benchmark_status(path, 'dora_rank32')
    def test_dora_preflight_requires_magnitude_state_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'check.json'
            report = {'format_version': 20, 'recipe': 'dora_rank32', **RESOURCE_SPECS['dora_rank32'],
                      'success': True, 'lora_modules': 72, 'optimizer_steps': 2,
                      'frozen_backbone_unchanged': True, 'roundtrip_logits_exact': True,
                      'all_attention_and_mlp_gradients_nonzero': True, 'missing_mlp_weight_rejected': True}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'magnitude'): official_check(path, 'dora_rank32')
            report['missing_magnitude_weight_rejected'] = True
            path.write_text(json.dumps(report)); official_check(path, 'dora_rank32')
