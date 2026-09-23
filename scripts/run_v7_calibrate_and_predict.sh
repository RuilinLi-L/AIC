#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

RUN_NAME="$1"
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
V7_ROOT="${AIC_V7_ROOT:-${OUTPUT_ROOT}/v7}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
BATCH_SIZE="${AIC_BATCH_SIZE:-64}"
WORKERS="${AIC_WORKERS:-4}"
GPU="${AIC_GPU:-0}"

cd "${PROJECT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluate_tta_v7.py \
  --validation-checkpoint "${RUN_DIR}/best_model.pt" \
  --final-checkpoint "${RUN_DIR}/model_uncalibrated.pt" \
  --train-dir "${DATA_DIR}/train" \
  --data-manifest "${V7_ROOT}/dataset_manifest_v7.json" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${RUN_DIR}" \
  --output-checkpoint "${RUN_DIR}/model.pt" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --device cuda 2>&1 | tee -a "${RUN_DIR}/calibration.log"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u predict_v7.py \
  --checkpoint "${RUN_DIR}/model.pt" \
  --model-dir "${MODEL_DIR}" \
  --test-dir "${DATA_DIR}/test" \
  --output "${RUN_DIR}/pred_results.csv" \
  --zip-output "${RUN_DIR}/pred_results.zip" \
  --logits-output "${RUN_DIR}/test_logits.npy" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --device cuda \
  --expected-rows 37444 2>&1 | tee -a "${RUN_DIR}/prediction.log"
