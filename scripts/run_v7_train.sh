#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_NAME drop|partial shuffle|repeat-factor" >&2
  exit 2
fi

RUN_NAME="$1"
POLICY="$2"
SAMPLER="$3"
if [[ "${POLICY}" != "drop" && "${POLICY}" != "partial" ]]; then
  echo "invalid conflict policy: ${POLICY}" >&2
  exit 2
fi
if [[ "${SAMPLER}" != "shuffle" && "${SAMPLER}" != "repeat-factor" ]]; then
  echo "invalid sampler: ${SAMPLER}" >&2
  exit 2
fi

PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
V7_ROOT="${AIC_V7_ROOT:-${OUTPUT_ROOT}/v7}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
BATCH_SIZE="${AIC_BATCH_SIZE:-64}"
GRAD_ACCUM="${AIC_GRAD_ACCUM:-4}"
WORKERS="${AIC_WORKERS:-4}"
GPU="${AIC_GPU:-0}"

if [[ ! -f "${V7_ROOT}/dataset_manifest_v7.json" ]]; then
  echo "missing ${V7_ROOT}/dataset_manifest_v7.json; run scripts/audit_v7_data.sh first" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}" "${V7_ROOT}/cache"
resume_args=()
if [[ "${AIC_RESUME:-1}" == "1" && -f "${RUN_DIR}/resume_latest.pt" ]]; then
  resume_args=(--resume "${RUN_DIR}/resume_latest.pt")
fi

cd "${PROJECT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_v7.py \
  --train-dir "${DATA_DIR}/train" \
  --data-manifest "${V7_ROOT}/dataset_manifest_v7.json" \
  --conflict-policy "${POLICY}" \
  --sampler "${SAMPLER}" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${RUN_DIR}" \
  --feature-cache "${V7_ROOT}/cache/frozen_${POLICY}.npy" \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --gradient-accumulation "${GRAD_ACCUM}" \
  "${resume_args[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
