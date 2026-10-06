#!/usr/bin/env bash
# Launch this supervisor inside screen. Children and all later stages are autonomous.
set -eEuo pipefail
while (($#)); do
  case "$1" in
    --gpu-a) export AIC_GPU_A="$2"; shift 2;;
    --gpu-b) export AIC_GPU_B="$2"; shift 2;;
    --output-root) export AIC_OUTPUT_ROOT="$2"; shift 2;;
    --project-dir) export AIC_PROJECT_DIR="$2"; shift 2;;
    --python) export AIC_PYTHON="$2"; shift 2;;
    --data-dir) export AIC_DATA_DIR="$2"; shift 2;;
    --model-dir) export AIC_MODEL_DIR="$2"; shift 2;;
    --min-free-mib) export AIC_MIN_FREE_MIB="$2"; shift 2;;
    --baseline-v15) export AIC_BASELINE_V15="$2"; shift 2;;
    --candidate-a) export AIC_CANDIDATE_A="$2"; shift 2;;
    --candidate-b) export AIC_CANDIDATE_B="$2"; shift 2;;
    --selection) export AIC_SELECTION="$2"; shift 2;;
    *) echo "Unknown V17 pipeline argument: $1" >&2; exit 2;;
  esac
done
export AIC_PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
export AIC_PYTHON="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
export AIC_OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
export AIC_PIPELINE_DIR="${AIC_PIPELINE_DIR:-${AIC_OUTPUT_ROOT}/v17_pipeline}"
export AIC_SELECTION="${AIC_SELECTION:-${AIC_OUTPUT_ROOT}/v17_selection.json}"
export AIC_CANDIDATE_A="${AIC_CANDIDATE_A:-${AIC_OUTPUT_ROOT}/v17_agreement_recovery}"
export AIC_CANDIDATE_B="${AIC_CANDIDATE_B:-${AIC_OUTPUT_ROOT}/v17_dynamic_prototype}"
GPU_A="${AIC_GPU_A:-${AIC_GPU:?Set AIC_GPU_A (or AIC_GPU) explicitly}}"
GPU_B="${AIC_GPU_B:-$GPU_A}"
[[ "$GPU_A" =~ ^[0-9]+$ && "$GPU_B" =~ ^[0-9]+$ ]] || { echo 'Explicit numeric GPU indices required' >&2; exit 2; }
mkdir -p "$AIC_PIPELINE_DIR"
cd "$AIC_PROJECT_DIR"
exec 9>"${AIC_PIPELINE_DIR}/.lock"
flock -n 9 || { echo 'V17 pipeline already active' >&2; exit 2; }
support() { "$AIC_PYTHON" v17_pipeline_support.py "$@" --root "$AIC_OUTPUT_ROOT"; }
stage=starting
state() { support state --stage "$stage" --status "$1"; }
trap 'state failed; echo "V17 pipeline failed during $stage; inspect logs and child status" >&2' ERR
trap 'state interrupted; exit 130' INT TERM
trap 'rc=$?; if ((rc != 0 && rc != 130)); then state failed; fi' EXIT
export AIC_PIPELINE_STARTED_AT="$(support init)"
state running
stage=baseline_binding; state running
support wait-baselines --baseline-v15 "${AIC_BASELINE_V15:-${AIC_OUTPUT_ROOT}/v15_expanded_mlp}" \
  --timeout "${AIC_DEPENDENCY_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}"
candidate() (
  local recipe="$1" gpu="$2" run="$3" candidate_stage=starting
  export AIC_V17_RECIPE="$recipe" AIC_GPU="$gpu" AIC_RUN_DIR="$run"
  mkdir -p "$run"
  exec 6>"${run}/.candidate.lock"
  flock -n 6 || { echo "$recipe candidate already active" >&2; exit 2; }
  candidate_state() { support state --stage "$candidate_stage" --status "$1" --state-file "${recipe}_status.json"; }
  trap 'candidate_state failed; echo "V17 $recipe failed during $candidate_stage" >&2' ERR
  trap 'candidate_state interrupted; exit 130' INT TERM
  trap 'rc=$?; if ((rc != 0 && rc != 130)); then candidate_state failed; fi' EXIT
  candidate_stage=benchmark; candidate_state running
  if [[ "$recipe" == agreement_recovery ]]; then
    export AIC_BENCHMARK_REPORT="${AIC_BENCHMARK_A:-${run}/benchmark.json}"
    export AIC_OFFICIAL_CHECK="${AIC_OFFICIAL_CHECK_A:-${run}/official_check.json}"
  else
    export AIC_BENCHMARK_REPORT="${AIC_BENCHMARK_B:-${run}/benchmark.json}"
    export AIC_OFFICIAL_CHECK="${AIC_OFFICIAL_CHECK_B:-${run}/official_check.json}"
  fi
  candidate_stage=official_check; candidate_state running
  if [[ ! -f "$AIC_OFFICIAL_CHECK" ]]; then
    (
      exec 5>"${AIC_PIPELINE_DIR}/gpu_${gpu}.lock"
      flock -w "${AIC_GPU_TIMEOUT:-172800}" 5
      support wait-gpu --gpu "$gpu" --minimum "${AIC_MIN_FREE_MIB:-32768}" \
        --timeout "${AIC_GPU_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}" \
        --state-file "${recipe}_official_check_gpu.json"
      CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$AIC_PYTHON" -u scripts/check_v17_official.py --recipe "$recipe" \
        --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}" \
        --output "$AIC_OFFICIAL_CHECK" --device cuda 2>&1 | tee -a "${run}/official_check.log"
    )
  fi
  support official-check --report "$AIC_OFFICIAL_CHECK" --recipe "$recipe" \
    --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}"
  candidate_stage=benchmark; candidate_state running
  benchmark_needed="$(support benchmark-needed --benchmark "$AIC_BENCHMARK_REPORT" --recipe "$recipe")"
  if [[ "$benchmark_needed" == 1 ]]; then
    if [[ -f "$AIC_BENCHMARK_REPORT" ]]; then
      cp "$AIC_BENCHMARK_REPORT" "${AIC_BENCHMARK_REPORT}.failed.$(date +%s).json"
    fi
    (
      exec 5>"${AIC_PIPELINE_DIR}/gpu_${gpu}.lock"
      flock -w "${AIC_GPU_TIMEOUT:-172800}" 5
      support wait-gpu --gpu "$gpu" --minimum "${AIC_MIN_FREE_MIB:-32768}" \
        --timeout "${AIC_GPU_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}" \
        --state-file "${recipe}_benchmark_gpu.json"
      CUDA_VISIBLE_DEVICES="$gpu" AIC_PIN_MEMORY=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$AIC_PYTHON" -u benchmark_v17.py --recipe "$recipe" \
        --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}" \
        --train-dir "${AIC_DATA_DIR:-${AIC_PROJECT_DIR}/data}/train" \
        --data-manifest "${AIC_MANIFEST:-${AIC_OUTPUT_ROOT}/v7/dataset_manifest_v7.json}" \
        --checkpoint "${AIC_BENCHMARK_METADATA:-${AIC_OUTPUT_ROOT}/v13_expanded/best_model.pt}" \
        --feature-cache "${AIC_FEATURE_CACHE:-${AIC_OUTPUT_ROOT}/v13_expanded/cache/frozen.npy}" \
        --output "$AIC_BENCHMARK_REPORT" --device cuda --steps 8 --warmup-steps 2 \
        2>&1 | tee -a "${run}/benchmark.log"
    )
  fi
  settings="$(support benchmark-settings --benchmark "$AIC_BENCHMARK_REPORT" --recipe "$recipe")"
  read -r AIC_WORKERS AIC_BATCH_SIZE <<< "$settings"
  export AIC_WORKERS AIC_BATCH_SIZE
  candidate_stage=validation; candidate_state running
  if [[ ! -f "${run}/validation.done" ]]; then
    AIC_V17_STAGE=validate bash scripts/run_v17_train.sh
    [[ -f "${run}/best_model.pt" ]] || { echo 'Training returned without best_model.pt' >&2; exit 2; }
    touch "${run}/validation.done"
  fi
  candidate_stage=evaluation; candidate_state running
  if [[ ! -f "${run}/evaluation.done" ]]; then
    AIC_V17_STAGE=evaluate bash scripts/run_v17_evaluate_and_predict.sh
    [[ -f "${run}/strict_eval.json" && -f "${run}/model.pt" ]]
    touch "${run}/evaluation.done"
  fi
  candidate_stage=complete; candidate_state complete
)
stage=candidate_validation_and_evaluation; state running
if [[ "$GPU_A" != "$GPU_B" ]]; then
  # Children release the supervisor lock. Never stop any existing server process.
  (exec 9>&-; candidate agreement_recovery "$GPU_A" "$AIC_CANDIDATE_A") >"${AIC_PIPELINE_DIR}/agreement_recovery.log" 2>&1 &
  pid_a=$!
  (exec 9>&-; candidate dynamic_prototype "$GPU_B" "$AIC_CANDIDATE_B") >"${AIC_PIPELINE_DIR}/dynamic_prototype.log" 2>&1 &
  pid_b=$!
  failure=0
  # Wait commands run with errexit enabled in the children; parent handles either failure explicitly.
  wait "$pid_a" || failure=1
  wait "$pid_b" || failure=1
  if ((failure)); then state failed; echo 'At least one V17 candidate failed; resume after diagnosis' >&2; exit 1; fi
else
  candidate agreement_recovery "$GPU_A" "$AIC_CANDIDATE_A" 2>&1 | tee -a "${AIC_PIPELINE_DIR}/agreement_recovery.log"
  candidate dynamic_prototype "$GPU_B" "$AIC_CANDIDATE_B" 2>&1 | tee -a "${AIC_PIPELINE_DIR}/dynamic_prototype.log"
fi
stage=selection; state running
if [[ ! -f "${AIC_PIPELINE_DIR}/selection.done" ]]; then
  if [[ -e "${AIC_REFIT_DIR:-${AIC_OUTPUT_ROOT}/v17_refit}/resume_latest.pt" ]]; then
    # Never replace a contract already consumed by a partially completed refit.
    support bind
  else
    AIC_V17_STAGE=select bash scripts/run_v17_evaluate_and_predict.sh
  fi
  touch "${AIC_PIPELINE_DIR}/selection.done"
fi
support bind
accepted="$(support accepted)"
export AIC_GPU="${AIC_REFIT_GPU:-$GPU_A}"
stage=validation_prediction; state running
# The delivery command revalidates completed packages without repeating inference.
AIC_V17_STAGE=predict-validation bash scripts/run_v17_evaluate_and_predict.sh
touch "${AIC_PIPELINE_DIR}/validation_prediction.done"
if [[ "$accepted" == 1 ]]; then
  export AIC_V17_RECIPE="$(support recipe)"
  if [[ "$AIC_V17_RECIPE" == agreement_recovery ]]; then benchmark_run="$AIC_CANDIDATE_A"; else benchmark_run="$AIC_CANDIDATE_B"; fi
  if [[ "$AIC_V17_RECIPE" == agreement_recovery ]]; then
    export AIC_BENCHMARK_REPORT="${AIC_BENCHMARK_A:-${benchmark_run}/benchmark.json}"
    export AIC_OFFICIAL_CHECK="${AIC_OFFICIAL_CHECK_A:-${benchmark_run}/official_check.json}"
  else
    export AIC_BENCHMARK_REPORT="${AIC_BENCHMARK_B:-${benchmark_run}/benchmark.json}"
    export AIC_OFFICIAL_CHECK="${AIC_OFFICIAL_CHECK_B:-${benchmark_run}/official_check.json}"
  fi
  settings="$(support benchmark-settings --benchmark "$AIC_BENCHMARK_REPORT" --recipe "$AIC_V17_RECIPE")"
  read -r AIC_WORKERS AIC_BATCH_SIZE <<< "$settings"
  export AIC_WORKERS AIC_BATCH_SIZE
  stage=full_data_refit; state running
  if [[ ! -f "${AIC_PIPELINE_DIR}/refit.done" ]]; then
    AIC_V17_STAGE=refit bash scripts/run_v17_train.sh
    touch "${AIC_PIPELINE_DIR}/refit.done"
  fi
  stage=refit_prediction; state running
  AIC_V17_STAGE=predict bash scripts/run_v17_evaluate_and_predict.sh
  touch "${AIC_PIPELINE_DIR}/prediction.done"
else
  [[ "$accepted" == 0 ]] || { echo 'Invalid V17 acceptance value' >&2; exit 2; }
  stage=fallback_delivery; state running
  AIC_V17_STAGE=fallback-refit bash scripts/run_v17_evaluate_and_predict.sh
  touch "${AIC_PIPELINE_DIR}/fallback.done"
fi
stage=complete; state complete
echo "v17_pipeline_complete refit_required=$accepted delivery=${AIC_DELIVERY_DIR:-${AIC_OUTPUT_ROOT}/v17_delivery}"
