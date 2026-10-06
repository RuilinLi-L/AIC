#!/usr/bin/env bash
set -euo pipefail
export AIC_PROJECT_DIR=/home/mcxu/lrl/AIC/.runs/v19_dedup_20261005_202607
export AIC_DATA_DIR=/home/mcxu/lrl/AIC/data
export AIC_MODEL_DIR=/home/mcxu/lrl/AIC/clip-ViT-B-32
export AIC_PYTHON=/data/mcxu/conda-envs/aic/bin/python
export AIC_OUTPUT_ROOT=/data/mcxu/AIC/outputs
export AIC_PIPELINE_DIR=/data/mcxu/AIC/outputs/v19_dedup_20261005/pipeline
export AIC_V19_BRANCH=deduplicated_control
export AIC_GPU_A=3 AIC_GPU_B=0 AIC_GPU_CONTROL=2 AIC_REFIT_GPU=3
export AIC_MIN_FREE_MIB=45056 AIC_GPU_TIMEOUT=172800 AIC_POLL_INTERVAL=30
export AIC_BASELINE_V18=/data/mcxu/AIC/outputs/v18_rank32
export AIC_MANIFEST=/data/mcxu/AIC/outputs/v19_dedup_20261005/inputs/dataset_manifest_v19_dedup.json
export AIC_FEATURE_CACHE=/data/mcxu/AIC/outputs/v19_dedup_20261005/inputs/frozen.npy
export AIC_BENCHMARK_METADATA=/data/mcxu/AIC/outputs/v19_dedup_20261005/inputs/benchmark_metadata.pt
export AIC_DEDUP_RECORD=/data/mcxu/AIC/outputs/v19_dedup_20261005/inputs/dedup_derivation.json
export AIC_AUDIT_JSON=/data/mcxu/AIC/outputs/v19_dedup_20261005/audit/audit.json
export AIC_RESOURCE_JSON=/data/mcxu/AIC/outputs/v19_dedup_20261005/pipeline/resource_plan.json
export AIC_SOURCE_MANIFEST="$AIC_PROJECT_DIR/source_manifest.json"
export AIC_CANDIDATE_A=/data/mcxu/AIC/outputs/v19_dedup_20261005/resolution384_rank32
export AIC_CANDIDATE_B=/data/mcxu/AIC/outputs/v19_dedup_20261005/rank32_dropout
export AIC_CONTROL_RUN=/data/mcxu/AIC/outputs/v19_dedup_20261005/rank32_control
export AIC_OFFICIAL_CHECK_A="$AIC_CANDIDATE_A/official_check.json"
export AIC_OFFICIAL_CHECK_B="$AIC_CANDIDATE_B/official_check.json"
export AIC_OFFICIAL_CHECK_CONTROL="$AIC_CONTROL_RUN/official_check.json"
export AIC_BENCHMARK_A="$AIC_CANDIDATE_A/benchmark.json"
export AIC_BENCHMARK_B="$AIC_CANDIDATE_B/benchmark.json"
export AIC_BENCHMARK_CONTROL="$AIC_CONTROL_RUN/benchmark.json"
export AIC_REFIT_DIR=/data/mcxu/AIC/outputs/v19_dedup_20261005/refit
export AIC_SELECTION=/data/mcxu/AIC/outputs/v19_dedup_20261005/selection.json
export AIC_DELIVERY_DIR=/data/mcxu/AIC/outputs/v19_dedup_20261005/delivery
export AIC_BATCH_SIZE=256 AIC_WORKERS=8 AIC_EXPECTED_ROWS=37444
export AIC_PIPELINE_STARTED_AT=1791183600
export PYTHONDONTWRITEBYTECODE=1
cd "$AIC_PROJECT_DIR"
"$AIC_PYTHON" -B - <<'CHECK'
import hashlib,json
from pathlib import Path
record=json.loads(Path('source_manifest.json').read_text())
assert Path.cwd()==Path(record['remote_snapshot'])
for name,digest in record['sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
print(f"Verified {len(record['sha256'])} frozen V19 source files",flush=True)
CHECK
"$AIC_PYTHON" -B v19_pipeline_support.py audit-check --root "$AIC_OUTPUT_ROOT" --audit-json "$AIC_AUDIT_JSON"
"$AIC_PYTHON" -B v19_pipeline_support.py resource-check --root "$AIC_OUTPUT_ROOT" --resource-json "$AIC_RESOURCE_JSON"
mkdir -p "$AIC_PIPELINE_DIR"
if screen -ls | grep -q '[.]v19_dedup_pipeline'; then
  echo 'V19 deduplicated pipeline screen already exists; inspect before resuming.' >&2
  exit 2
fi
nvidia-smi --id=0,1,2,3 --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits > "$AIC_PIPELINE_DIR/gpu_prelaunch.csv"
date --iso-8601=seconds > "$AIC_PIPELINE_DIR/launched_at.txt"
screen -dmS v19_dedup_pipeline bash -c 'exec bash "$AIC_PROJECT_DIR/scripts/run_v19_pipeline.sh" >> "$AIC_PIPELINE_DIR/supervisor.log" 2>&1'
screen -ls
