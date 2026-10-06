#!/usr/bin/env bash
set -eEuo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
STAGE="${AIC_V15_STAGE:-evaluate}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AIC_PIN_MEMORY=0
cd "$PROJECT_DIR"
support() { "$PYTHON_BIN" v15_pipeline_support.py "$@" --root "$OUTPUT_ROOT"; }
if [[ "$STAGE" != select ]]; then
  GPU="${AIC_GPU:?Set AIC_GPU to a physical GPU index}"
  [[ "$GPU" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index" >&2; exit 2; }
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
deliver() {
  local kind="$1" checkpoint out
  checkpoint="$(support checkpoint --kind "$kind")"
  out="${OUTPUT_ROOT}/v15_delivery/${kind}"
  mkdir -p "$out"
  # Separate output lock makes a manual delivery launch safe beside the pipeline.
  (
    flock -n 7 || { echo "Delivery $kind already active" >&2; exit 2; }
    if [[ ! -f "${out}/provenance.json" ]]; then
      if [[ "$kind" == v14_refit ]]; then
        support export --kind "$kind" --checkpoint "$checkpoint" --test-dir "${DATA_DIR}/test" --collect
      else
        "$PYTHON_BIN" -u predict_v15.py --checkpoint "$checkpoint" --model-dir "$MODEL_DIR" \
          --test-dir "${DATA_DIR}/test" --output "${out}/pred_results.csv" --zip-output "${out}/pred_results.zip" \
          --expected-rows 37444 --device cuda --batch-size "${AIC_EVAL_BATCH_SIZE:-128}" \
          --workers "${AIC_EVAL_WORKERS:-4}" --prefetch-factor 1 2>&1 | tee -a "${out}/predict.log"
        support export --kind "$kind" --checkpoint "$checkpoint" --test-dir "${DATA_DIR}/test"
      fi
    else
      support export --kind "$kind" --checkpoint "$checkpoint" --test-dir "${DATA_DIR}/test"
    fi
  ) 7>"${out}/.lock"
}
case "$STAGE" in
  evaluate)
    run="${OUTPUT_ROOT}/v15_expanded_mlp"
    mkdir -p "$run"
    exec 8>"${run}/.evaluate.lock"
    flock -n 8 || { echo "V15 evaluation already active" >&2; exit 2; }
    "$PYTHON_BIN" -u evaluate_tta_v15.py --run-dir "$run" --output-dir "$run" \
      --train-dir "${DATA_DIR}/train" --model-dir "$MODEL_DIR" \
      --data-manifest "${AIC_MANIFEST:-${OUTPUT_ROOT}/v7/dataset_manifest_v7.json}" \
      --device cuda --batch-size "${AIC_EVAL_BATCH_SIZE:-128}" --workers "${AIC_EVAL_WORKERS:-4}" \
      --prefetch-factor 1 2>&1 | tee -a "${run}/evaluate.log"
    ;;
  select)
    support wait-selection --timeout "${AIC_DEPENDENCY_TIMEOUT:-172800}"
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" select_v15.py \
      --baseline-selection "${OUTPUT_ROOT}/v14_selection.json" --candidate "${OUTPUT_ROOT}/v15_expanded_mlp" \
      --model-dir "$MODEL_DIR" --output "${OUTPUT_ROOT}/v15_selection.json"
    baseline_evaluation="$(support baseline-evaluation)"
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" compare_v15.py --baseline "$baseline_evaluation" \
      --candidate "${OUTPUT_ROOT}/v15_expanded_mlp" --output "${OUTPUT_ROOT}/v15_pipeline/paired_comparison.json" \
      --bootstrap 2000
    ;;
  v14-pair)
    delivery_stage=waiting_v14_selection
    delivery_state() { support state --stage "$delivery_stage" --status "$1" --state-file v14_delivery_status.json; }
    trap 'delivery_state failed' ERR
    trap 'delivery_state interrupted; exit 130' INT TERM
    delivery_state running
    support wait-selection --timeout "${AIC_DEPENDENCY_TIMEOUT:-172800}"
    delivery_stage=v14_validation; delivery_state running
    deliver v14_validation
    delivery_stage=waiting_v14_prediction; delivery_state running
    support wait-prediction --timeout "${AIC_DEPENDENCY_TIMEOUT:-172800}"
    delivery_stage=v14_refit; delivery_state running
    deliver v14_refit
    delivery_stage=complete; delivery_state complete
    touch "${OUTPUT_ROOT}/v15_pipeline/v14_pair.done"
    ;;
  predict-validation) deliver v15_validation ;;
  predict) deliver v15_refit ;;
  *) echo "V15 stage must be evaluate, select, v14-pair, predict-validation or predict" >&2; exit 2;;
esac
