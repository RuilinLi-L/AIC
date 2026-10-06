#!/usr/bin/env bash
# Execute on A100. This launches only V17 and preserves every existing process.
set -euo pipefail
export AIC_PROJECT_DIR=/home/mcxu/lrl/AIC/.runs/v17_20261002_154924
export AIC_DATA_DIR=/home/mcxu/lrl/AIC/data
export AIC_MODEL_DIR=/home/mcxu/lrl/AIC/clip-ViT-B-32
export AIC_PYTHON=/data/mcxu/conda-envs/aic/bin/python
export AIC_OUTPUT_ROOT=/data/mcxu/AIC/outputs
export AIC_PIPELINE_DIR=/data/mcxu/AIC/outputs/v17_pipeline
export AIC_GPU_A=2 AIC_GPU_B=1 AIC_REFIT_GPU=2
# Both real-image probes used 23.73 GiB allocated; reserve at least 30 GiB free.
export AIC_MIN_FREE_MIB=30720 AIC_GPU_TIMEOUT=172800 AIC_POLL_INTERVAL=30
export AIC_OFFICIAL_CHECK_A=/data/mcxu/AIC/outputs/v17_startup/official_agreement_recovery.json
export AIC_OFFICIAL_CHECK_B=/data/mcxu/AIC/outputs/v17_startup/official_dynamic_prototype.json
export AIC_BENCHMARK_A=/data/mcxu/AIC/outputs/v17_startup/benchmark_agreement_recovery.json
export AIC_BENCHMARK_B=/data/mcxu/AIC/outputs/v17_startup/benchmark_dynamic_prototype.json
export AIC_PIPELINE_STARTED_AT
AIC_PIPELINE_STARTED_AT="$($AIC_PYTHON -c 'import datetime; print(datetime.datetime.fromisoformat("2026-10-02T15:47:00+08:00").timestamp())')"
cd "$AIC_PROJECT_DIR"
"$AIC_PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path
record = json.loads(Path('/data/mcxu/AIC/outputs/v17_startup/source_manifest.json').read_text())
assert Path.cwd() == Path(record['remote_snapshot'])
for name, digest in record['sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
print(f"Verified {len(record['sha256'])} frozen source files", flush=True)
PY
mkdir -p "$AIC_PIPELINE_DIR"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits > /data/mcxu/AIC/outputs/v17_startup/gpu_prelaunch.csv
date --iso-8601=seconds > /data/mcxu/AIC/outputs/v17_startup/launched_at.txt
screen -dmS v17_pipeline bash -c 'exec bash "$AIC_PROJECT_DIR/scripts/run_v17_pipeline.sh" >> "$AIC_PIPELINE_DIR/supervisor.log" 2>&1'
screen -ls
