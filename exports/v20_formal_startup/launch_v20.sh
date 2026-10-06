#!/usr/bin/env bash
set -euo pipefail
export AIC_PROJECT_DIR=/home/mcxu/lrl/AIC/.runs/v20_formal_shared_20261006_150101
export AIC_PYTHON=/data/mcxu/conda-envs/aic/bin/python
export AIC_OUTPUT_ROOT=/data/mcxu/AIC/outputs
export AIC_V20_ROOT=/data/mcxu/AIC/outputs/v20_joint_20261006
export AIC_V19_RESOURCE_JSON=/data/mcxu/AIC/outputs/v19_dedup_20261005/pipeline/resource_plan.json
export AIC_MODEL_DIR=/home/mcxu/lrl/AIC/clip-ViT-B-32
export AIC_DATA_DIR=/home/mcxu/lrl/AIC/data
export AIC_MANIFEST=/data/mcxu/AIC/outputs/v19_dedup_20261005/inputs/dataset_manifest_v19_dedup.json
export AIC_FEATURE_CACHE=/data/mcxu/AIC/outputs/v19_dedup_20261005/inputs/frozen.npy
export AIC_BENCHMARK_METADATA=/data/mcxu/AIC/outputs/v19_dedup_20261005/inputs/benchmark_metadata.pt
export AIC_PIPELINE_STARTED_AT=1791249793.9967067
export AIC_V20_GPU_POLICY=gpu3_shared_measured_v1
export AIC_V20_SHARED_PROBE=/data/mcxu/AIC/outputs/v20_joint_20261006/gpu3_shared_probe_20261006_143539/context.json
export AIC_GPU_TIMEOUT=172800
export AIC_POLL_INTERVAL=30
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export AIC_PIN_MEMORY=0
cd "$AIC_PROJECT_DIR"
"$AIC_PYTHON" -B - <<'CHECK'
import hashlib,json,os,pathlib
from v19_pipeline_support import read_resource_plan
from v20_core import model_identity
root=pathlib.Path(os.environ['AIC_PROJECT_DIR'])
source=json.loads((root/'source_manifest.json').read_text())
assert source['remote_snapshot']==str(root), 'wrong source snapshot'
for name,digest in source['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, f'frozen source changed: {name}'
assert (root/'v20_final_tests.log').read_text().rstrip().endswith('OK'), 'V20 regression incomplete'
assert 'Ran 58 tests' in (root/'v20_final_tests.log').read_text(), 'wrong V20 regression snapshot'
assert (root/'cpu_regression.log').read_text().rstrip().endswith('OK'), 'V19 compatibility regression incomplete'
assert 'Ran 65 tests' in (root/'cpu_regression.log').read_text(), 'wrong V19 compatibility regression'
plan=read_resource_plan(os.environ['AIC_V19_RESOURCE_JSON'])
assert plan['branch']=='deduplicated_control'
assert plan['base_model_identity']==model_identity(os.environ['AIC_MODEL_DIR'])
for key in ('AIC_MANIFEST','AIC_FEATURE_CACHE','AIC_BENCHMARK_METADATA'):
    assert pathlib.Path(os.environ[key]).is_file(), f'missing input: {key}'
assert pathlib.Path(os.environ['AIC_DATA_DIR'],'test').is_dir()
print('v20_startup_verified source_files='+str(len(source['sha256']))+' cpu_tests=123',flush=True)
CHECK
exec bash "$AIC_PROJECT_DIR/scripts/run_v20_pipeline.sh"
