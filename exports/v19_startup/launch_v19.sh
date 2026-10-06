#!/usr/bin/env bash
# Prepared launcher. Current audit fails; this exits before GPU work until the data protocol is repaired.
set -euo pipefail
export AIC_PROJECT_DIR=/home/mcxu/lrl/AIC/.runs/v19_20261005_180239
export AIC_DATA_DIR=/home/mcxu/lrl/AIC/data
export AIC_MODEL_DIR=/home/mcxu/lrl/AIC/clip-ViT-B-32
export AIC_PYTHON=/data/mcxu/conda-envs/aic/bin/python
export AIC_OUTPUT_ROOT=/data/mcxu/AIC/outputs
export AIC_PIPELINE_DIR=/data/mcxu/AIC/outputs/v19_pipeline
export AIC_GPU_A=0 AIC_GPU_B=3 AIC_REFIT_GPU=0
export AIC_MIN_FREE_MIB=45056 AIC_GPU_TIMEOUT=172800 AIC_POLL_INTERVAL=30
export AIC_BASELINE_V18=/data/mcxu/AIC/outputs/v18_rank32
export AIC_MANIFEST=/data/mcxu/AIC/outputs/v7/dataset_manifest_v7.json
export AIC_FEATURE_CACHE=/data/mcxu/AIC/outputs/v13_expanded/cache/frozen.npy
export AIC_BENCHMARK_METADATA=/data/mcxu/AIC/outputs/v18_rank32/model.pt
export AIC_RESOURCE_JSON=/data/mcxu/AIC/outputs/v19_pipeline/resource_plan.json
export AIC_AUDIT_JSON=/data/mcxu/AIC/outputs/v19_startup/audit/audit.json
export AIC_BATCH_SIZE=256 AIC_WORKERS=8
export PYTHONDONTWRITEBYTECODE=1
export AIC_PIPELINE_STARTED_AT=1791183600.0
cd "$AIC_PROJECT_DIR"
"$AIC_PYTHON" - <<'CHECK'
import hashlib,json
from pathlib import Path
record=json.loads(Path('source_manifest.json').read_text())
if Path.cwd()!=Path(record['remote_snapshot']):
    raise RuntimeError('Incorrect V19 snapshot directory')
for name,digest in record['sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError(f'V19 source changed: {name}')
print(f"Verified {len(record['sha256'])} frozen source files",flush=True)
CHECK
# A failed audit is not permission to switch manifest, baseline, or validation rules.
"$AIC_PYTHON" v19_pipeline_support.py audit-check --root "$AIC_OUTPUT_ROOT" --audit-json "$AIC_AUDIT_JSON"
exec bash scripts/run_v19_pipeline.sh
