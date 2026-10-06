#!/usr/bin/env bash
# Execute on A100 after tests and the real benchmark pass.
set -euo pipefail
export AIC_PROJECT_DIR=/home/mcxu/lrl/AIC/.runs/v15_20261001_092146
export AIC_DATA_DIR=/home/mcxu/lrl/AIC/data
export AIC_MODEL_DIR=/home/mcxu/lrl/AIC/clip-ViT-B-32
export AIC_PYTHON=/data/mcxu/conda-envs/aic/bin/python
export AIC_OUTPUT_ROOT=/data/mcxu/AIC/outputs
export AIC_BENCHMARK=/data/mcxu/AIC/outputs/v15_benchmark.json
export AIC_GPU=2 AIC_DELIVERY_GPU=6 AIC_PIN_MEMORY=0
export AIC_WORKERS=8 AIC_EVAL_WORKERS=4 AIC_EVAL_BATCH_SIZE=128
if screen -ls | grep -Eq '[.]v15_pipeline[[:space:]]'; then
  echo 'v15_pipeline screen already exists; inspect it before restarting' >&2
  exit 2
fi
mkdir -p "$AIC_OUTPUT_ROOT/v15_pipeline"
cd "$AIC_PROJECT_DIR"
screen -dmS v15_pipeline bash -c 'exec bash "$AIC_PROJECT_DIR/scripts/run_v15_pipeline.sh" >> "$AIC_OUTPUT_ROOT/v15_pipeline/launcher.log" 2>&1'
