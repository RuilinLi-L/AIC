#!/usr/bin/env bash
set -euo pipefail
export AIC_PROJECT_DIR=/home/mcxu/lrl/AIC/.runs/v18_20261003_233900
export AIC_DATA_DIR=/home/mcxu/lrl/AIC/data
export AIC_MODEL_DIR=/home/mcxu/lrl/AIC/clip-ViT-B-32
export AIC_PYTHON=/data/mcxu/conda-envs/aic/bin/python
export AIC_OUTPUT_ROOT=/data/mcxu/AIC/outputs
export AIC_PIPELINE_DIR=/data/mcxu/AIC/outputs/v18_pipeline
export AIC_GPU_A=0 AIC_GPU_B=1 AIC_REFIT_GPU=0
export AIC_MIN_FREE_MIB=45056 AIC_GPU_TIMEOUT=172800 AIC_POLL_INTERVAL=30
export AIC_RESOURCE_JSON=/data/mcxu/AIC/outputs/v18_pipeline/resource_plan.json
export AIC_FEATURE_CACHE=/data/mcxu/AIC/outputs/v13_expanded/cache/frozen.npy
export AIC_BENCHMARK_METADATA=/data/mcxu/AIC/outputs/v15_expanded_mlp/model.pt
export AIC_OFFICIAL_CHECK_A=/data/mcxu/AIC/outputs/v18_startup/official_resolution384.json
export AIC_BENCHMARK_A=/data/mcxu/AIC/outputs/v18_startup/benchmark_resolution384.json
export AIC_PIPELINE_STARTED_AT
AIC_PIPELINE_STARTED_AT="$($AIC_PYTHON -c 'import datetime; print(datetime.datetime.fromisoformat("2026-10-03T15:17:40+00:00").timestamp())')"
cd "$AIC_PROJECT_DIR"
"$AIC_PYTHON" - <<'CHECK'
import hashlib,json
from pathlib import Path
record=json.loads(Path('/data/mcxu/AIC/outputs/v18_startup/source_manifest.json').read_text())
assert Path.cwd()==Path(record['remote_snapshot'])
for name,digest in record['sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
print(f"Verified {len(record['sha256'])} frozen source files",flush=True)
CHECK
mkdir -p "$AIC_PIPELINE_DIR"
if screen -ls | grep -q '\.v18_pipeline'; then
  echo 'V18 screen already exists; inspect before resuming.' >&2
  exit 2
fi
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits > /data/mcxu/AIC/outputs/v18_startup/gpu_migration_20261004.csv
date --iso-8601=seconds > /data/mcxu/AIC/outputs/v18_startup/migrated_at_20261004.txt
screen -dmS v18_pipeline bash -c 'exec bash "$AIC_PROJECT_DIR/scripts/run_v18_pipeline.sh" >> "$AIC_PIPELINE_DIR/supervisor.log" 2>&1'
screen -ls
