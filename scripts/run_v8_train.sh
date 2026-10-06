#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
RUN_DIR="${OUTPUT_ROOT}/v8_neighbors"
V7_ROOT="${AIC_V7_ROOT:-${OUTPUT_ROOT}/v7}"
GPU="${AIC_GPU:-0}"
BATCH_SIZE="${AIC_BATCH_SIZE:-64}"
GRAD_ACCUM="${AIC_GRAD_ACCUM:-4}"
WORKERS="${AIC_WORKERS:-4}"

if [[ ! -f "${V7_ROOT}/dataset_manifest_v7.json" ]]; then
  echo "missing V7 dataset manifest" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"
resume_args=()
if [[ "${AIC_RESUME:-1}" == "1" && -f "${RUN_DIR}/resume_latest.pt" ]]; then
  resume_args=(--resume "${RUN_DIR}/resume_latest.pt")
fi
cd "${PROJECT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_v8.py \
  --train-dir "${DATA_DIR}/train" \
  --data-manifest "${V7_ROOT}/dataset_manifest_v7.json" \
  --conflict-policy partial \
  --sampler repeat-factor \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${RUN_DIR}" \
  --feature-cache "${V7_ROOT}/cache/frozen_partial.npy" \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation "${GRAD_ACCUM}" \
  --workers "${WORKERS}" \
  "${resume_args[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
