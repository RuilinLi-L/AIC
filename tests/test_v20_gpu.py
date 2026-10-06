"""Shared GPU admission preserves memory caps and V20 mutual exclusion."""
import fcntl
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from v20_gpu import SHARED, gpu_policy, task_limit
from v20_pipeline import Pipeline
from test_v20_pipeline import SchedulerTests, RecordingPipeline


class SharedGpuTests(unittest.TestCase):
    def test_shared_gpu3_ignores_v19_mutex_but_preserves_v20_mutex_and_caps(self):
        fixture = SchedulerTests()
        for recipe, cap in (('loraplus_rank32', 32), ('dora_rank32', 48)):
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as directory:
                root, work, env = fixture.setup_pipeline(directory)
                env['AIC_V20_GPU_POLICY'] = SHARED
                with (root / 'gpu_3.lock').open('a') as old_lock:
                    fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    def smi(args, **kwargs):
                        return SimpleNamespace(stdout='55653\n' if '--id=3' in args else '0\n')
                    with patch.dict(os.environ, env), patch('v20_pipeline.subprocess.run', side_effect=smi):
                        pipeline = Pipeline()
                        with pipeline.lease(recipe) as gpu:
                            self.assertEqual(gpu, 3)
                            self.assertEqual(pipeline.gpu_context.cap_gib, cap)
                            other = Pipeline(); other.timeout = 0
                            with self.assertRaises(TimeoutError):
                                with other.lease('loraplus_rank32'): self.fail('second V20 occupied same GPU')
                        self.assertIsNone(pipeline.gpu_context.cap_gib)
                self.assertEqual(task_limit(gpu_policy(SHARED), recipe)['cap_gib'], cap)
        with self.assertRaises(ValueError): task_limit(gpu_policy(SHARED), 'unidentified_task')

    def test_shared_pipeline_prioritizes_loraplus_and_still_refits_only_winner(self):
        fixture = SchedulerTests()
        with tempfile.TemporaryDirectory() as directory:
            root, work, env = fixture.setup_pipeline(directory)
            env['AIC_V20_GPU_POLICY'] = SHARED
            plan = fixture.plan(root, work); plan['gpu_runtime'] = gpu_policy(SHARED)
            selected = {'refit_required': True, 'source_checkpoint': str(work / 'selected.pt'),
                        'source_format_version': 20, 'recipe': 'loraplus_rank32', 'training_config': {}}
            with patch.dict(os.environ, env), patch('v20_pipeline.read_resource_plan', return_value=plan), \
                 patch('select_v20.read_decision', return_value=selected), patch('v20_pipeline.export_submission'):
                pipeline = RecordingPipeline(); pipeline.execute()
                self.assertEqual(pipeline.leases[0], 'loraplus_rank32')
                self.assertIn('loraplus_rank32_refit', pipeline.leases)
                refits = [args for args, _, _ in pipeline.calls if args[0]=='train_v20.py' and '--stage' in args and args[args.index('--stage')+1]=='refit']
                self.assertEqual(len(refits), 1)

    def test_wrapper_applies_ceiling_before_running_unmodified_training_entrypoint(self):
        from run_cuda_capped_v20 import main
        with patch('sys.argv', ['runner', '--cap-gib', '32', 'train_v20.py', '--recipe', 'loraplus_rank32']), \
             patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=80*2**30)), \
             patch('torch.cuda.set_per_process_memory_fraction') as cap, patch('runpy.run_path') as run:
            main()
            cap.assert_called_once_with(.4, device=0)
            self.assertEqual(run.call_args.args[0], str(Path('train_v20.py').resolve()))
