#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
: "${AIC_V20_ROOT:?Set an independent V20 output directory}"
: "${AIC_V19_RESOURCE_JSON:?Set the frozen deduplicated V19 resource plan}"
: "${AIC_MODEL_DIR:?Set official CLIP ViT-B/32 weights}"
: "${AIC_DATA_DIR:?Set the official stage data directory}"
: "${AIC_MANIFEST:?Set the deduplicated V19 manifest}"
: "${AIC_FEATURE_CACHE:?Set its frozen feature cache}"
: "${AIC_BENCHMARK_METADATA:?Set its metadata-only checkpoint}"
export AIC_PROJECT_DIR="$PROJECT_DIR" AIC_PYTHON="$PYTHON"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 AIC_PIN_MEMORY=0
exec "$PYTHON" -B -u "$PROJECT_DIR/v20_pipeline.py"
