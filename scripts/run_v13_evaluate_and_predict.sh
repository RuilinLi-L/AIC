#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
STAGE="${AIC_V13_STAGE:-evaluate}"
SELECTION="${AIC_SELECTION:-${OUTPUT_ROOT}/v13_selection.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "${PROJECT_DIR}"
case "${STAGE}" in evaluate|select|predict) ;; *) echo "Stage must be evaluate, select or predict" >&2; exit 2 ;; esac
if [[ "${STAGE}" != select ]]; then
  GPU="${AIC_GPU:?Set AIC_GPU to your chosen physical GPU index}"
  if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then echo "Invalid GPU index" >&2; exit 2; fi
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi
if [[ "${STAGE}" == evaluate ]]; then
  for recipe in strength dynamic expanded; do
    run="${OUTPUT_ROOT}/v13_${recipe}"
    "${PYTHON_BIN}" -u evaluate_tta_v13.py --run-dir "${run}" --output-dir "${run}" \
      --train-dir "${DATA_DIR}/train" --model-dir "${MODEL_DIR}" \
      --data-manifest "${AIC_MANIFEST:-${OUTPUT_ROOT}/v7/dataset_manifest_v7.json}" \
      --device cuda --batch-size "${AIC_EVAL_BATCH_SIZE:-256}" --workers "${AIC_WORKERS:-16}" \
      --prefetch-factor "${AIC_PREFETCH_FACTOR:-2}"
  done
fi
if [[ "${STAGE}" == evaluate || "${STAGE}" == select ]]; then
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" select_v13.py --baseline "${OUTPUT_ROOT}/v12_lora_fast" \
    --candidates "${OUTPUT_ROOT}/v13_strength" "${OUTPUT_ROOT}/v13_dynamic" "${OUTPUT_ROOT}/v13_expanded" \
    --model-dir "${MODEL_DIR}" --output "${SELECTION}"
  echo "Selection saved. Next: AIC_GPU=N AIC_V13_STAGE=refit bash scripts/run_v13_train.sh"
else
  run="${OUTPUT_ROOT}/v13_refit"
  "${PYTHON_BIN}" -u predict_v13.py --checkpoint "${run}/model.pt" --model-dir "${MODEL_DIR}" \
    --test-dir "${DATA_DIR}/test" --output "${run}/pred_results.csv" --zip-output "${run}/pred_results.zip" \
    --expected-rows 37444 --device cuda --batch-size "${AIC_EVAL_BATCH_SIZE:-256}" \
    --workers "${AIC_WORKERS:-16}" --prefetch-factor "${AIC_PREFETCH_FACTOR:-2}"
fi
