"""Isolated, memory-capped GPU3 LoRA+ engineering probe alongside V19."""
import datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import torch

PROJECT = Path('/home/mcxu/lrl/AIC/.runs/v20_joint_20261006_094716')
WORK = Path('/data/mcxu/AIC/outputs/v20_joint_20261006')
STAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUT = WORK / ('gpu3_shared_probe_' + STAMP)
OUT.mkdir()
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
lock = Path('/data/mcxu/AIC/outputs/v20_gpu_leases/gpu_3.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
free = int(subprocess.check_output(['nvidia-smi', '--id=3', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
if free < 49152:
    raise RuntimeError(f'GPU3 needs at least 48GiB free for this capped probe; actual {free}MiB')
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
torch.cuda.set_per_process_memory_fraction(.4, device=0)
context = {'kind': 'v20_gpu3_shared_engineering_probe', 'recipe': 'loraplus_rank32',
           'physical_gpu': 3, 'pytorch_allocator_cap_fraction': .4, 'pytorch_allocator_cap_gib': 32,
           'minimum_free_mib': 49152, 'free_mib_at_start': free,
           'source_snapshot': str(PROJECT), 'shares_gpu_with_v19_control': True,
           'formal_training_started': False, 'canonical_preflight_outputs_modified': False,
           'started_at': datetime.datetime.now().astimezone().isoformat(), 'status': 'running'}
(OUT / 'context.json').write_text(json.dumps(context, indent=2))
print('shared_probe_directory=' + str(OUT), flush=True)
model = '/home/mcxu/lrl/AIC/clip-ViT-B-32'
inputs = '/data/mcxu/AIC/outputs/v19_dedup_20261005/inputs'
commands = [
    ['scripts/check_v20_official.py', '--recipe', 'loraplus_rank32', '--model-dir', model,
     '--output', str(OUT / 'official_check.json'), '--device', 'cuda'],
    ['benchmark_v20.py', '--recipe', 'loraplus_rank32', '--model-dir', model,
     '--train-dir', '/home/mcxu/lrl/AIC/data/train', '--data-manifest', inputs + '/dataset_manifest_v19_dedup.json',
     '--checkpoint', inputs + '/benchmark_metadata.pt', '--feature-cache', inputs + '/frozen.npy',
     '--output', str(OUT / 'benchmark.json'), '--device', 'cuda', '--batch-size', '256',
     '--gradient-accumulation', '1', '--warmup-steps', '2', '--steps', '8'],
]
try:
    for command in commands:
        sys.argv = command
        runpy.run_path(str(PROJECT / command[0]), run_name='__main__')
        gc.collect()
        torch.cuda.empty_cache()
    report = json.loads((OUT / 'benchmark.json').read_text())
    context['status'] = 'passed' if report['success'] else 'probe_failed'
    context['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
    context['benchmark_success'] = report['success']
    context['configurations'] = report['configurations']
except BaseException as error:
    context.update(status='failed', error_type=type(error).__name__, error=str(error))
    raise
finally:
    context['finished_at'] = datetime.datetime.now().astimezone().isoformat()
    (OUT / 'context.json').write_text(json.dumps(context, indent=2))
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    print(json.dumps(context, indent=2), flush=True)
