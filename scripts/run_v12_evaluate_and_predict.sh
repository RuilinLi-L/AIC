#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
RUN_DIR="${OUTPUT_ROOT}/v12_lora_fast"
GPU="${AIC_GPU:?Set AIC_GPU to a physical GPU index after checking nvidia-smi}"
STAGE="${AIC_V12_STAGE:-all}"
BATCH_SIZE="${AIC_EVAL_BATCH_SIZE:-256}"
WORKERS="${AIC_WORKERS:-16}"
PREFETCH="${AIC_PREFETCH_FACTOR:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then echo "GPU must be a nonnegative physical index" >&2; exit 2; fi
if [[ "${STAGE}" != "all" && "${STAGE}" != "evaluate" && "${STAGE}" != "predict" ]]; then
  echo "AIC_V12_STAGE must be all, evaluate or predict" >&2; exit 2
fi
free_mib="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < 8192 )); then
  echo "GPU ${GPU} needs at least 8192 MiB free before evaluation; observed ${free_mib:-unknown}" >&2; exit 2
fi
cd "${PROJECT_DIR}"
if [[ "${STAGE}" != "predict" ]]; then
  if [[ ! -f "${RUN_DIR}/metrics.json" ]]; then echo "V12 training is incomplete" >&2; exit 2; fi
  common=(--train-dir "${DATA_DIR}/train" --data-manifest "${OUTPUT_ROOT}/v7/dataset_manifest_v7.json"
    --model-dir "${MODEL_DIR}" --batch-size "${BATCH_SIZE}" --workers "${WORKERS}" --prefetch-factor "${PREFETCH}" --device cuda)
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluate_tta_v12.py --run-dir "${RUN_DIR}" \
    --output-dir "${RUN_DIR}" --output-checkpoint "${RUN_DIR}/model.pt" "${common[@]}" 2>&1 | tee -a "${RUN_DIR}/calibration.log"
  for baseline in v9_soft_teacher v11_320; do
    if [[ ! -f "${OUTPUT_ROOT}/${baseline}/metrics.json" ]]; then
      echo "Comparison pending: ${baseline} has not finished training" | tee -a "${RUN_DIR}/comparison.log"
      continue
    fi
    reference="${OUTPUT_ROOT}/v12_reference_${baseline}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u evaluate_tta_v12.py --run-dir "${OUTPUT_ROOT}/${baseline}" \
      --output-dir "${reference}" "${common[@]}" 2>&1 | tee -a "${RUN_DIR}/baseline_${baseline}.log"
    "${PYTHON_BIN}" -u compare_v12.py --baseline "${reference}" --candidate "${RUN_DIR}" \
      --output "${RUN_DIR}/comparison_${baseline}.json" 2>&1 | tee -a "${RUN_DIR}/comparison.log"
  done
fi
if [[ "${STAGE}" != "evaluate" ]]; then
  if [[ ! -f "${RUN_DIR}/strict_eval.json" || ! -f "${RUN_DIR}/model.pt" ]]; then
    echo "V12 calibration must finish before prediction" >&2; exit 2
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u predict_v12.py --checkpoint "${RUN_DIR}/model.pt" \
    --model-dir "${MODEL_DIR}" --test-dir "${DATA_DIR}/test" --output "${RUN_DIR}/pred_results.csv" \
    --zip-output "${RUN_DIR}/pred_results.zip" --batch-size "${BATCH_SIZE}" --workers "${WORKERS}" \
    --prefetch-factor "${PREFETCH}" --device cuda --expected-rows 37444 2>&1 | tee -a "${RUN_DIR}/prediction.log"
fi
