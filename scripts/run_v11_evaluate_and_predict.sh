#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
RUN_DIR="${OUTPUT_ROOT}/v11_320"
GPU="${AIC_GPU:?Set AIC_GPU to a physical GPU index after checking nvidia-smi}"
STAGE="${AIC_V11_STAGE:-all}"
BATCH_SIZE="${AIC_BATCH_SIZE:-64}"
WORKERS="${AIC_WORKERS:-4}"

if [[ "${STAGE}" != "all" && "${STAGE}" != "evaluate" && "${STAGE}" != "predict" ]]; then
  echo "AIC_V11_STAGE must be all, evaluate, or predict" >&2
  exit 2
fi

BASELINE_DIR="${OUTPUT_ROOT}/v9_soft_teacher"
REFERENCE_DIR="${OUTPUT_ROOT}/v11_reference_v9_soft_teacher"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if [[ ! -f "${BASELINE_DIR}/metrics.json" ]]; then
  echo "Selected V9 run is incomplete: ${BASELINE_DIR}" >&2
  exit 2
fi
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
  echo "AIC_GPU must be a nonnegative physical GPU index" >&2
  exit 2
fi
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v row="$((GPU + 1))" 'NR == row {gsub(/[^0-9]/, "", $1); print $1}')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < 16384 )); then
  echo "Physical GPU ${GPU} needs at least 16384 MiB free; observed ${free_mib:-unknown}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"

if [[ "${STAGE}" != "predict" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluate_tta_v11.py \
    --run-dir "${BASELINE_DIR}" \
    --train-dir "${DATA_DIR}/train" \
    --data-manifest "${OUTPUT_ROOT}/v7/dataset_manifest_v7.json" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${REFERENCE_DIR}" \
    --batch-size "${BATCH_SIZE}" --workers "${WORKERS}" --device cuda \
    2>&1 | tee -a "${RUN_DIR}/baseline_evaluation.log"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluate_tta_v11.py \
    --run-dir "${RUN_DIR}" \
    --train-dir "${DATA_DIR}/train" \
    --data-manifest "${OUTPUT_ROOT}/v7/dataset_manifest_v7.json" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${RUN_DIR}" \
    --output-checkpoint "${RUN_DIR}/model.pt" \
    --batch-size "${BATCH_SIZE}" --workers "${WORKERS}" --device cuda \
    2>&1 | tee -a "${RUN_DIR}/calibration.log"

  "${PYTHON_BIN}" -u compare_v11.py \
    --baseline "${REFERENCE_DIR}" \
    --candidate "${RUN_DIR}" \
    --output "${RUN_DIR}/comparison.json" \
    2>&1 | tee -a "${RUN_DIR}/comparison.log"
fi

if [[ "${STAGE}" != "evaluate" ]]; then
  if [[ ! -f "${RUN_DIR}/comparison.json" || ! -f "${RUN_DIR}/model.pt" ]]; then
    echo "V11 evaluation must finish before final prediction" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u predict_v11.py \
    --checkpoint "${RUN_DIR}/model.pt" \
    --model-dir "${MODEL_DIR}" \
    --test-dir "${DATA_DIR}/test" \
    --output "${RUN_DIR}/pred_results.csv" \
    --zip-output "${RUN_DIR}/pred_results.zip" \
    --batch-size "${BATCH_SIZE}" --workers "${WORKERS}" \
    --device cuda --expected-rows 37444 2>&1 | tee -a "${RUN_DIR}/prediction.log"
fi
