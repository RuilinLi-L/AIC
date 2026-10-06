"""Run the approved V20 experiment without modifying the V19 supervisor."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from v20_pipeline_support import (atomic_json, sha256, state, initialize,
    freeze_resources, read_resource_plan, resource_for_recipe, export_submission)

RECIPES = ('dora_rank32', 'loraplus_rank32')


class Pipeline:
    def __init__(self):
        self.project = Path(os.environ['AIC_PROJECT_DIR']).resolve()
        self.python = os.environ.get('AIC_PYTHON', sys.executable)
        self.root = Path(os.environ.get('AIC_OUTPUT_ROOT', '/data/mcxu/AIC/outputs'))
        self.work = Path(os.environ['AIC_V20_ROOT']).resolve()
        self.pipeline = self.work / 'pipeline'
        self.pipeline.mkdir(parents=True, exist_ok=True)
        os.environ['AIC_PIPELINE_DIR'] = str(self.pipeline)
        os.environ['AIC_SELECTION'] = str(self.work / 'selection.json')
        os.environ['AIC_DELIVERY_DIR'] = str(self.work / 'delivery')
        self.resources = self.pipeline / 'resource_plan.json'
        self.old_resources = Path(os.environ['AIC_V19_RESOURCE_JSON']).resolve()
        self.model = os.environ['AIC_MODEL_DIR']
        self.data = Path(os.environ['AIC_DATA_DIR'])
        self.manifest = os.environ['AIC_MANIFEST']
        self.features = os.environ['AIC_FEATURE_CACHE']
        self.metadata = os.environ['AIC_BENCHMARK_METADATA']
        self.timeout = int(os.environ.get('AIC_GPU_TIMEOUT', '172800'))
        self.interval = int(os.environ.get('AIC_POLL_INTERVAL', '30'))
        self.started = initialize(self.root)['started_at_unix']
        os.environ['AIC_PIPELINE_STARTED_AT'] = str(self.started)

    @contextmanager
    def lease(self, label):
        started = time.monotonic()
        locks = self.root / 'v20_gpu_leases'
        locks.mkdir(parents=True, exist_ok=True)
        while True:
            for gpu in range(4):
                handles = []
                try:
                    # Honor the V19 mutex too: its existing recovery worker must
                    # not race a V20 job onto the same newly freed card.
                    for directory in (locks, self.old_resources.parent):
                        handle = (directory / f'gpu_{gpu}.lock').open('a')
                        handles.append(handle)
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    result = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
                        '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
                    free = int(result.stdout.strip())
                    if free >= 71680:
                        state(self.root, label, 'gpu_ready', f'{label}_status.json', {'physical_gpu': gpu, 'free_mib': free})
                        print(f'v20_gpu_ready task={label} physical_gpu={gpu} free_mib={free}', flush=True)
                        yield gpu
                        return
                except BlockingIOError:
                    continue
                finally:
                    for handle in reversed(handles):
                        fcntl.flock(handle, fcntl.LOCK_UN)
                        handle.close()
            state(self.root, label, 'waiting_gpu', f'{label}_status.json', {'minimum_free_mib': 71680, 'allowed_gpus': [0, 1, 2, 3]})
            if time.monotonic() - started >= self.timeout:
                raise TimeoutError(f'{label}: no allowed GPU reached 71680 MiB')
            time.sleep(self.interval)

    def run(self, args, log, gpu=None):
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'AIC_PIN_MEMORY': '0', 'PYTHONDONTWRITEBYTECODE': '1'}
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open('a', buffering=1) as handle:
            handle.write(f'command={json.dumps(args)} physical_gpu={gpu}\n')
            subprocess.run([self.python, '-B', '-u', *args], cwd=self.project, env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)

    def preflight(self, recipe):
        run = self.work / recipe
        run.mkdir(parents=True, exist_ok=True)
        with self.lease(f'{recipe}_preflight') as gpu:
            state(self.root, 'preflight', 'running', f'{recipe}_status.json')
            if not (run / 'official_check.json').exists():
                self.run(['scripts/check_v20_official.py', '--recipe', recipe, '--model-dir', self.model,
                          '--output', str(run / 'official_check.json'), '--device', 'cuda'], run / 'official_check.log', gpu)
            if not (run / 'benchmark.json').exists():
                self.run(['benchmark_v20.py', '--recipe', recipe, '--model-dir', self.model,
                          '--train-dir', str(self.data / 'train'), '--data-manifest', self.manifest,
                          '--checkpoint', self.metadata, '--feature-cache', self.features,
                          '--output', str(run / 'benchmark.json'), '--device', 'cuda',
                          '--batch-size', '256', '--gradient-accumulation', '1',
                          '--warmup-steps', '2', '--steps', '8'], run / 'benchmark.log', gpu)
        state(self.root, 'preflight_complete', 'complete', f'{recipe}_status.json')

    def train_args(self, recipe, run, stage, resource):
        args = ['--recipe', recipe, '--stage', stage, '--resource-json', str(resource),
                '--train-dir', str(self.data / 'train'), '--data-manifest', self.manifest,
                '--model-dir', self.model, '--output-dir', str(run), '--device', 'cuda',
                '--batch-size', '256', '--gradient-accumulation', '1', '--workers', '8',
                '--eval-batch-size', '128', '--prefetch-factor', '1', '--feature-cache', self.features]
        if stage == 'refit':
            args += ['--selection-json', os.environ['AIC_SELECTION']]
        if (run / 'resume_latest.pt').exists():
            args += ['--resume', str(run / 'resume_latest.pt')]
        return args

    def evaluate(self, recipe, run, gpu):
        out = self.work / 'joint_evaluations' / recipe
        if (out / 'strict_eval.json').is_file() and (out / 'model.pt').is_file():
            from select_v14 import read_evaluation
            from compare_v20 import _validate_source
            from compare_v10 import load_evaluation
            resource = read_resource_plan(self.resources)
            entry = (resource_for_recipe(resource, recipe) if recipe in RECIPES else
                     resource['control'] if recipe == 'rank32_control' else resource['dependencies'][recipe])
            _validate_source(read_evaluation(out), entry, resource, self.resources)
            load_evaluation(out)
            return out
        self.run(['evaluate_tta_v20.py', '--run-dir', str(run), '--train-dir', str(self.data / 'train'),
                  '--data-manifest', self.manifest, '--model-dir', self.model,
                  '--output-dir', str(out), '--output-checkpoint', str(out / 'model.pt'),
                  '--batch-size', '128', '--workers', '8', '--prefetch-factor', '1', '--device', 'cuda'],
                 self.pipeline / f'{recipe}_joint_evaluation.log', gpu)
        if not (out / 'strict_eval.json').is_file() or not (out / 'model.pt').is_file():
            raise RuntimeError(f'{recipe} joint evaluation returned without products')
        return out

    def candidate(self, recipe):
        resource = read_resource_plan(self.resources)
        entry = resource_for_recipe(resource, recipe)
        if entry['status'] == 'resource_infeasible':
            state(self.root, 'resource_infeasible', 'skipped', f'{recipe}_status.json')
            return None
        run = Path(entry['run_dir'])
        with self.lease(recipe) as gpu:
            if not self.complete_training(run):
                state(self.root, 'validation_training', 'running', f'{recipe}_status.json')
                self.run(['train_v20.py', *self.train_args(recipe, run, 'validate', self.resources)], run / 'train.log', gpu)
            if not self.complete_training(run):
                raise RuntimeError(f'{recipe} did not produce complete 24-epoch validation')
            state(self.root, 'joint_evaluation', 'running', f'{recipe}_status.json')
            out = self.evaluate(recipe, run, gpu)
        state(self.root, 'validation_and_joint_evaluation', 'complete', f'{recipe}_status.json')
        return out

    @staticmethod
    def complete_training(run):
        path = Path(run) / 'metrics.json'
        if not path.is_file() or not (Path(run) / 'validation_epochs' / 'epoch_24.pt').is_file():
            return False
        metrics = json.loads(path.read_text())
        history = metrics.get('training_history', [])
        return metrics.get('stage') == 'validate' and [item['epoch'] for item in history] == list(range(1, 25))

    def legacy_evaluation(self, entry):
        recipe, run = entry['recipe'], Path(entry['run_dir'])
        if entry['status'] == 'resource_infeasible':
            if recipe == 'rank32_control':
                raise RuntimeError('the required fixed control is resource-infeasible')
            return None
        started = time.monotonic()
        while not self.complete_training(run):
            state(self.root, 'v19_dependency', 'waiting', f'{recipe}_dependency.json',
                  {'run_dir': str(run), 'needs': 'complete 24 epochs; V19 recovery supervisor remains responsible'})
            if time.monotonic() - started >= self.timeout:
                raise TimeoutError(f'V19 {recipe} has not completed; no silent omission')
            time.sleep(self.interval)
        with self.lease(f'{recipe}_evaluation') as gpu:
            return self.evaluate(recipe, run, gpu)

    def predict(self, checkpoint, kind, gpu):
        out = self.work / 'delivery' / kind
        out.mkdir(parents=True, exist_ok=True)
        self.run(['predict_v20.py', '--checkpoint', str(checkpoint), '--model-dir', self.model,
                  '--test-dir', str(self.data / 'test'), '--output', str(out / 'pred_results.csv'),
                  '--zip-output', str(out / 'pred_results.zip'), '--expected-rows', '37444',
                  '--batch-size', '128', '--workers', '8', '--prefetch-factor', '1', '--device', 'cuda'],
                 out / 'predict.log', gpu)
        export_submission(self.root, kind, checkpoint, self.data / 'test')

    def guarded(self, operation, label, *args):
        try:
            return operation(*args)
        except BaseException as error:
            state(self.root, label, "failed", f"{label}_status.json",
                  {"error": type(error).__name__, "message": str(error)})
            raise

    def execute(self):
        state(self.root, 'preflight_queue', 'running')
        if not self.resources.exists():
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(self.guarded, self.preflight, recipe + "_preflight", recipe) for recipe in RECIPES]
                for job in jobs:
                    job.result()
            freeze_resources(self.root, SimpleNamespace(
                v19_resource=str(self.old_resources), candidate_a=str(self.work / RECIPES[0]),
                candidate_b=str(self.work / RECIPES[1]), model_dir=self.model,
                source_manifest=str(self.project / 'source_manifest.json'), resource_json=str(self.resources)))
        resource = read_resource_plan(self.resources)
        if not Path(os.environ['AIC_SELECTION']).exists():
            state(self.root, 'validation_and_joint_evaluation', 'running')
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = {recipe: pool.submit(self.guarded, self.candidate, recipe, recipe) for recipe in RECIPES}
                new = {recipe: job.result() for recipe, job in jobs.items()}
            state(self.root, 'baseline_wait_and_joint_evaluation', 'running')
            entries = [resource['control'], *resource['dependencies'].values()]
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = {entry['recipe']: pool.submit(self.guarded, self.legacy_evaluation, entry['recipe'] + '_dependency', entry) for entry in entries}
                old = {recipe: job.result() for recipe, job in jobs.items()}
            args = ['select_v20.py', '--control', str(old['rank32_control']), '--resource-json', str(self.resources),
                    '--fallback-v18', resource['historical_refit']['directory'], '--output', os.environ['AIC_SELECTION']]
            for flag, value in [('--v19-resolution384', old['resolution384_rank32']), ('--v19-dropout', old['rank32_dropout']),
                                ('--v20-dora', new['dora_rank32']), ('--v20-loraplus', new['loraplus_rank32'])]:
                if value is not None:
                    args += [flag, str(value)]
            state(self.root, 'selection', 'running')
            self.run(args, self.pipeline / 'selection.log')
        from select_v20 import read_decision
        selected = read_decision(os.environ['AIC_SELECTION'])
        if not selected['refit_required']:
            source = Path(resource['historical_refit']['directory'])
            provenance = json.loads((source / 'provenance.json').read_text())
            out = self.work / 'delivery' / 'fallback_refit'
            out.mkdir(parents=True, exist_ok=True)
            for name in ('pred_results.csv', 'pred_results.zip'):
                shutil.copyfile(source / name, out / name)
            export_submission(self.root, 'fallback_refit', provenance['checkpoint'], self.data / 'test')
        else:
            with self.lease('winner_refit') as gpu:
                self.predict(selected['source_checkpoint'], 'validation', gpu)
                run = self.work / 'refit'
                run.mkdir(parents=True, exist_ok=True)
                version = selected['source_format_version']
                trainer = 'train_v20.py' if version == 20 else 'refit_legacy_v20.py'
                training_resource = self.resources if version == 20 else Path(selected['training_config']['resource_json'])
                state(self.root, 'winner_refit', 'running')
                if not (run / 'model.pt').is_file():
                    self.run([trainer, *self.train_args(selected['recipe'], run, 'refit', training_resource)], run / 'train.log', gpu)
                self.predict(run / 'model.pt', 'refit', gpu)
        state(self.root, 'complete', 'complete')


def main():
    pipeline = Pipeline()
    with (pipeline.pipeline / '.supervisor.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            pipeline.execute()
        except BaseException as error:
            state(pipeline.root, 'pipeline', 'failed', detail={'error': type(error).__name__, 'message': str(error)})
            raise


if __name__ == '__main__':
    main()
