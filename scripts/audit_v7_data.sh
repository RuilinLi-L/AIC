#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
V7_ROOT="${AIC_V7_ROOT:-/data/mcxu/AIC/outputs/v7}"

mkdir -p "${V7_ROOT}"
cd "${PROJECT_DIR}"
"${PYTHON_BIN}" audit_dataset.py \
  --train-dir "${DATA_DIR}/train" \
  --output "${V7_ROOT}/dataset_manifest_v7.json" \
  --seed 2026 \
  --verify-expected-aic-v7
