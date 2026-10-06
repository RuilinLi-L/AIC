#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
STAGE="${AIC_V14_STAGE:-evaluate}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export AIC_PIN_MEMORY="${AIC_PIN_MEMORY:-0}"
cd "$PROJECT_DIR"
if [[ "$STAGE" != select ]]; then
  GPU="${AIC_GPU:?Set AIC_GPU to a physical GPU index}"
  [[ "$GPU" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index" >&2; exit 2; }
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
case "$STAGE" in
  baseline|evaluate)
    run="${OUTPUT_ROOT}/v14_expanded_robust"; out="$run"
    if [[ "$STAGE" == baseline ]]; then
      run="${OUTPUT_ROOT}/v13_expanded"; out="${OUTPUT_ROOT}/v14_baseline_expanded"
    fi
    mkdir -p "$out"
    "$PYTHON_BIN" -u evaluate_tta_v14.py --run-dir "$run" --output-dir "$out" \
      --train-dir "${DATA_DIR}/train" --model-dir "$MODEL_DIR" \
      --data-manifest "${AIC_MANIFEST:-${OUTPUT_ROOT}/v7/dataset_manifest_v7.json}" \
      --device cuda --batch-size "${AIC_EVAL_BATCH_SIZE:-128}" --workers "${AIC_EVAL_WORKERS:-4}" \
      --prefetch-factor 1 2>&1 | tee -a "${out}/evaluate.log"
    ;;
  select)
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" select_v14.py \
      --baseline "${OUTPUT_ROOT}/v14_baseline_expanded" --candidate "${OUTPUT_ROOT}/v14_expanded_robust" \
      --model-dir "$MODEL_DIR" --output "${AIC_SELECTION:-${OUTPUT_ROOT}/v14_selection.json}"
    ;;
  predict)
    run="${OUTPUT_ROOT}/v14_refit"
    "$PYTHON_BIN" -u predict_v14.py --checkpoint "${run}/model.pt" --model-dir "$MODEL_DIR" \
      --test-dir "${DATA_DIR}/test" --output "${run}/pred_results.csv" --zip-output "${run}/pred_results.zip" \
      --expected-rows 37444 --device cuda --batch-size "${AIC_EVAL_BATCH_SIZE:-128}" \
      --workers "${AIC_EVAL_WORKERS:-4}" --prefetch-factor 1 2>&1 | tee -a "${run}/predict.log"
    ;;
  *) echo "V14 stage must be baseline, evaluate, select or predict" >&2; exit 2;;
esac
