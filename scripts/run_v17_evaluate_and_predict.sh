#!/usr/bin/env bash
set -eEuo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
PIPELINE_DIR="${AIC_PIPELINE_DIR:-${OUTPUT_ROOT}/v17_pipeline}"
DELIVERY_DIR="${AIC_DELIVERY_DIR:-${OUTPUT_ROOT}/v17_delivery}"
STAGE="${AIC_V17_STAGE:-evaluate}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AIC_PIN_MEMORY=0
mkdir -p "$PIPELINE_DIR"
cd "$PROJECT_DIR"
support() { "$PYTHON_BIN" v17_pipeline_support.py "$@" --root "$OUTPUT_ROOT"; }
acquire_gpu() {
  local gpu="${AIC_GPU:?Set an explicit physical GPU index}"
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Invalid GPU index' >&2; return 2; }
  exec 5>"${PIPELINE_DIR}/gpu_${gpu}.lock"
  flock -w "${AIC_GPU_TIMEOUT:-172800}" 5
  support wait-gpu --gpu "$gpu" --minimum "${AIC_EVAL_MIN_FREE_MIB:-${AIC_MIN_FREE_MIB:-32768}}" \
    --timeout "${AIC_GPU_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}" \
    --state-file "${AIC_V17_RECIPE:-selected}_${STAGE}_gpu.json"
  export CUDA_VISIBLE_DEVICES="$gpu"
}
deliver() {
  local kind="$1" checkpoint out reusable source
  checkpoint="$(support checkpoint --kind "$kind")"
  out="${DELIVERY_DIR}/${kind}"
  mkdir -p "$out"
  (
    flock -n 7 || { echo "V17 delivery $kind already active" >&2; exit 2; }
    if [[ -f "${out}/provenance.json" ]]; then
      support export --kind "$kind" --checkpoint "$checkpoint" --test-dir "${DATA_DIR}/test" --expected-rows "${AIC_EXPECTED_ROWS:-37444}"
      exit 0
    fi
    reusable=null
    if [[ "$kind" == validation ]]; then reusable="$(support reusable --kind validation)"; fi
    if [[ "$reusable" != null ]]; then
      source="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1])["directory"])' "$reusable")"
      support export --kind "$kind" --checkpoint "$checkpoint" --source "$source" \
        --test-dir "${DATA_DIR}/test" --expected-rows "${AIC_EXPECTED_ROWS:-37444}"
    else
      acquire_gpu
      "$PYTHON_BIN" -u predict_v17.py --checkpoint "$checkpoint" --model-dir "$MODEL_DIR" \
        --test-dir "${DATA_DIR}/test" --output "${out}/pred_results.csv" --zip-output "${out}/pred_results.zip" \
        --expected-rows "${AIC_EXPECTED_ROWS:-37444}" --device cuda --batch-size "${AIC_EVAL_BATCH_SIZE:-128}" \
        --workers "${AIC_EVAL_WORKERS:-4}" --prefetch-factor 1 2>&1 | tee -a "${out}/predict.log"
      support export --kind "$kind" --checkpoint "$checkpoint" --test-dir "${DATA_DIR}/test" --expected-rows "${AIC_EXPECTED_ROWS:-37444}"
    fi
  ) 7>"${out}/.lock"
}
case "$STAGE" in
  evaluate)
    RECIPE="${AIC_V17_RECIPE:?Set V17 recipe}"
    case "$RECIPE" in agreement_recovery|dynamic_prototype) ;; *) echo 'Invalid recipe' >&2; exit 2;; esac
    run="${AIC_RUN_DIR:-${OUTPUT_ROOT}/v17_${RECIPE}}"
    mkdir -p "$run"
    exec 8>"${run}/.evaluate.lock"
    flock -n 8 || { echo 'V17 evaluation already active' >&2; exit 2; }
    acquire_gpu
    "$PYTHON_BIN" -u evaluate_tta_v17.py --run-dir "$run" --output-dir "$run" \
      --train-dir "${DATA_DIR}/train" --model-dir "$MODEL_DIR" \
      --data-manifest "${AIC_MANIFEST:-${OUTPUT_ROOT}/v7/dataset_manifest_v7.json}" \
      --device cuda --batch-size "${AIC_EVAL_BATCH_SIZE:-128}" --workers "${AIC_EVAL_WORKERS:-4}" \
      --prefetch-factor 1 2>&1 | tee -a "${run}/evaluate.log"
    ;;
  select)
    baseline="${AIC_BASELINE_V15:-${OUTPUT_ROOT}/v15_expanded_mlp}"
    support wait-baselines --baseline-v15 "$baseline" \
      --timeout "${AIC_DEPENDENCY_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}"
    arguments=(--baseline "$baseline"
      --candidate-a "${AIC_CANDIDATE_A:-${OUTPUT_ROOT}/v17_agreement_recovery}"
      --candidate-b "${AIC_CANDIDATE_B:-${OUTPUT_ROOT}/v17_dynamic_prototype}")
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" compare_v17.py "${arguments[@]}" \
      --output "${PIPELINE_DIR}/paired_comparison.json" --bootstrap "${AIC_BOOTSTRAP:-2000}"
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" select_v17.py "${arguments[@]}" \
      --output "${AIC_SELECTION:-${OUTPUT_ROOT}/v17_selection.json}"
    support bind
    ;;
  predict-validation) deliver validation ;;
  predict) deliver refit ;;
  fallback-refit)
    mkdir -p "${DELIVERY_DIR}/fallback_refit"
    exec 7>"${DELIVERY_DIR}/fallback_refit/.lock"
    flock -n 7 || { echo 'V17 fallback delivery already active' >&2; exit 2; }
    reusable="$(support reusable --kind refit)"
    if [[ "$reusable" != null ]]; then
      checkpoint="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1])["checkpoint"])' "$reusable")"
      source="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1])["directory"])' "$reusable")"
      support export --kind fallback_refit --checkpoint "$checkpoint" --source "$source" \
        --test-dir "${DATA_DIR}/test" --expected-rows "${AIC_EXPECTED_ROWS:-37444}"
      support state --stage fallback_refit --status reused --state-file fallback_refit_status.json
    else
      support state --stage fallback_refit --status unavailable --state-file fallback_refit_status.json \
        --detail 'No matching existing baseline refit package; no retraining or indefinite wait.'
    fi
    ;;
  *) echo 'Invalid V17 evaluation stage' >&2; exit 2;;
esac
