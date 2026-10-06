#!/usr/bin/env bash
# Run inside screen. Freeze the engineering branch before either scientific run.
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
    --resource-json) export AIC_RESOURCE_JSON="$2"; shift 2;;
    *) echo "Unknown V18 pipeline argument: $1" >&2; exit 2;;
  esac
done
export AIC_PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
export AIC_PYTHON="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
export AIC_OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
export AIC_PIPELINE_DIR="${AIC_PIPELINE_DIR:-${AIC_OUTPUT_ROOT}/v18_pipeline}"
export AIC_SELECTION="${AIC_SELECTION:-${AIC_OUTPUT_ROOT}/v18_selection.json}"
export AIC_RESOURCE_JSON="${AIC_RESOURCE_JSON:-${AIC_PIPELINE_DIR}/resource_plan.json}"
export AIC_BASELINE_V15="${AIC_BASELINE_V15:-${AIC_OUTPUT_ROOT}/v15_expanded_mlp}"
GPU_A="${AIC_GPU_A:-${AIC_GPU:?Set AIC_GPU_A (or AIC_GPU) explicitly}}"
GPU_B="${AIC_GPU_B:-$GPU_A}"
[[ "$GPU_A" =~ ^[0-9]+$ && "$GPU_B" =~ ^[0-9]+$ ]] || { echo 'Explicit numeric GPU indices required' >&2; exit 2; }
mkdir -p "$AIC_PIPELINE_DIR"
cd "$AIC_PROJECT_DIR"
exec 9>"${AIC_PIPELINE_DIR}/.lock"
flock -n 9 || { echo 'V18 pipeline already active' >&2; exit 2; }
support() { "$AIC_PYTHON" v18_pipeline_support.py "$@" --root "$AIC_OUTPUT_ROOT"; }
stage=starting
state() { support state --stage "$stage" --status "$1"; }
trap 'state failed; echo "V18 pipeline failed during $stage; no scientific fallback is implied" >&2' ERR
trap 'state interrupted; exit 130' INT TERM
trap 'rc=$?; if ((rc != 0 && rc != 130)); then state failed; fi' EXIT
export AIC_PIPELINE_STARTED_AT="$(support init)"
state running
stage=baseline_binding; state running
support wait-baselines --baseline-v15 "$AIC_BASELINE_V15" \
  --timeout "${AIC_DEPENDENCY_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}"
preflight() (
  local recipe="$1" gpu="$2" run="$3" batch="$4" report="$5" official="$6"
  export AIC_V18_RECIPE="$recipe" AIC_GPU="$gpu" AIC_RUN_DIR="$run"
  mkdir -p "$run"
  exec 6>"${run}/.candidate.lock"
  flock -n 6 || { echo "$recipe preflight already active" >&2; exit 2; }
  acquire_gpu() {
    exec 5>"${AIC_PIPELINE_DIR}/gpu_${gpu}.lock"
    flock -w "${AIC_GPU_TIMEOUT:-172800}" 5
    support wait-gpu --gpu "$gpu" --minimum "${AIC_MIN_FREE_MIB:-32768}" \
      --timeout "${AIC_GPU_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}" \
      --state-file "${recipe}_preflight_gpu.json"
  }
  if [[ ! -f "$official" ]]; then
    (
      acquire_gpu
      CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$AIC_PYTHON" -u scripts/check_v18_official.py --recipe "$recipe" \
        --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}" \
        --output "$official" --device cuda 2>&1 | tee -a "${run}/official_check.log"
    )
  fi
  support official-check --report "$official" --recipe "$recipe" \
    --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}"
  needed="$(support benchmark-needed --benchmark "$report" --recipe "$recipe")"
  if [[ "$needed" == 1 ]]; then
    if [[ -f "$report" ]]; then cp "$report" "${report}.failed.$(date +%s).json"; fi
    (
      acquire_gpu
      CUDA_VISIBLE_DEVICES="$gpu" AIC_PIN_MEMORY=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$AIC_PYTHON" -u benchmark_v18.py --recipe "$recipe" \
        --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}" \
        --train-dir "${AIC_DATA_DIR:-${AIC_PROJECT_DIR}/data}/train" \
        --data-manifest "${AIC_MANIFEST:-${AIC_OUTPUT_ROOT}/v7/dataset_manifest_v7.json}" \
        --checkpoint "${AIC_BENCHMARK_METADATA:-${AIC_OUTPUT_ROOT}/v15_expanded_mlp/best_model.pt}" \
        --feature-cache "${AIC_FEATURE_CACHE:-${AIC_OUTPUT_ROOT}/v13_expanded/cache/frozen.npy}" \
        --output "$report" --device cuda --steps 8 --warmup-steps 2 \
        --batch-size "$batch" --gradient-accumulation "$((256 / batch))" \
        2>&1 | tee -a "${run}/benchmark.log"
    )
  fi
  support benchmark-settings --benchmark "$report" --recipe "$recipe"
)
stage=resource_preflight; state running
if [[ ! -f "$AIC_RESOURCE_JSON" ]]; then
  export AIC_CANDIDATE_A="${AIC_CANDIDATE_A:-${AIC_OUTPUT_ROOT}/v18_resolution384}"
  benchmark_a="${AIC_BENCHMARK_A:-${AIC_CANDIDATE_A}/benchmark.json}"
  official_a="${AIC_OFFICIAL_CHECK_A:-${AIC_CANDIDATE_A}/official_check.json}"
  preflight resolution384 "$GPU_A" "$AIC_CANDIDATE_A" 256 "$benchmark_a" "$official_a"
  second_recipe="$(support second-recipe --benchmark "$benchmark_a")"
  [[ "$second_recipe" == rank32 || "$second_recipe" == matched_control ]]
  export AIC_CANDIDATE_B="${AIC_CANDIDATE_B:-${AIC_OUTPUT_ROOT}/v18_${second_recipe}}"
  benchmark_b="${AIC_BENCHMARK_B:-${AIC_CANDIDATE_B}/benchmark.json}"
  official_b="${AIC_OFFICIAL_CHECK_B:-${AIC_CANDIDATE_B}/official_check.json}"
  batch=256; if [[ "$second_recipe" == matched_control ]]; then batch=128; fi
  preflight "$second_recipe" "$GPU_B" "$AIC_CANDIDATE_B" "$batch" "$benchmark_b" "$official_b"
  support freeze-resources --resource-json "$AIC_RESOURCE_JSON" \
    --benchmark-a "$benchmark_a" --benchmark-b "$benchmark_b" --official-a "$official_a" --official-b "$official_b" \
    --candidate-a "$AIC_CANDIDATE_A" --candidate-b "$AIC_CANDIDATE_B" --baseline-v15 "$AIC_BASELINE_V15" \
    --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}"
fi
support resource-check --resource-json "$AIC_RESOURCE_JSON"
frozen_a="$(support resource-field --candidate candidate_a --field run_dir)"
frozen_b="$(support resource-field --candidate candidate_b --field run_dir)"
[[ "${AIC_CANDIDATE_A:-$frozen_a}" == "$frozen_a" && "${AIC_CANDIDATE_B:-$frozen_b}" == "$frozen_b" ]] || {
  echo 'Candidate directories differ from frozen V18 resources' >&2; exit 2;
}
export AIC_CANDIDATE_A="$frozen_a" AIC_CANDIDATE_B="$frozen_b"
second_recipe="$(support resource-field --candidate candidate_b --field recipe)"
candidate() (
  local recipe="$1" gpu="$2" run="$3" candidate_stage=starting
  export AIC_V18_RECIPE="$recipe" AIC_GPU="$gpu" AIC_RUN_DIR="$run"
  exec 6>"${run}/.candidate.lock"
  flock -n 6 || { echo "$recipe candidate already active" >&2; exit 2; }
  candidate_state() { support state --stage "$candidate_stage" --status "$1" --state-file "${recipe}_status.json"; }
  trap 'candidate_state failed; echo "V18 $recipe failed during $candidate_stage" >&2' ERR
  trap 'candidate_state interrupted; exit 130' INT TERM
  trap 'rc=$?; if ((rc != 0 && rc != 130)); then candidate_state failed; fi' EXIT
  settings="$(support resource-settings --recipe "$recipe")"
  read -r AIC_WORKERS AIC_BATCH_SIZE <<< "$settings"
  export AIC_WORKERS AIC_BATCH_SIZE
  export AIC_BENCHMARK_REPORT="$(support resource-field --recipe "$recipe" --field benchmark)"
  export AIC_OFFICIAL_CHECK="$(support resource-field --recipe "$recipe" --field official_check)"
  candidate_stage=validation; candidate_state running
  if [[ ! -f "${run}/validation.done" ]]; then
    AIC_V18_STAGE=validate bash scripts/run_v18_train.sh
    [[ -f "${run}/best_model.pt" ]] || { echo 'Training returned without best_model.pt' >&2; exit 2; }
    touch "${run}/validation.done"
  fi
  candidate_stage=evaluation; candidate_state running
  if [[ ! -f "${run}/evaluation.done" ]]; then
    AIC_V18_STAGE=evaluate bash scripts/run_v18_evaluate_and_predict.sh
    [[ -f "${run}/strict_eval.json" && -f "${run}/model.pt" ]]
    touch "${run}/evaluation.done"
  fi
  candidate_stage=complete; candidate_state complete
)
stage=candidate_validation_and_evaluation; state running
if [[ "$GPU_A" != "$GPU_B" ]]; then
  (exec 9>&-; candidate resolution384 "$GPU_A" "$AIC_CANDIDATE_A") >"${AIC_PIPELINE_DIR}/resolution384.log" 2>&1 &
  pid_a=$!
  (exec 9>&-; candidate "$second_recipe" "$GPU_B" "$AIC_CANDIDATE_B") >"${AIC_PIPELINE_DIR}/${second_recipe}.log" 2>&1 &
  pid_b=$!
  failure=0
  wait "$pid_a" || failure=1
  wait "$pid_b" || failure=1
  if ((failure)); then state failed; echo 'At least one V18 candidate failed; resume after diagnosis' >&2; exit 1; fi
else
  candidate resolution384 "$GPU_A" "$AIC_CANDIDATE_A" 2>&1 | tee -a "${AIC_PIPELINE_DIR}/resolution384.log"
  candidate "$second_recipe" "$GPU_B" "$AIC_CANDIDATE_B" 2>&1 | tee -a "${AIC_PIPELINE_DIR}/${second_recipe}.log"
fi
stage=selection; state running
if [[ ! -f "${AIC_PIPELINE_DIR}/selection.done" ]]; then
  if [[ -e "${AIC_REFIT_DIR:-${AIC_OUTPUT_ROOT}/v18_refit}/resume_latest.pt" ]]; then
    support bind
  else
    AIC_V18_STAGE=select bash scripts/run_v18_evaluate_and_predict.sh
  fi
  touch "${AIC_PIPELINE_DIR}/selection.done"
fi
support bind
accepted="$(support accepted)"
export AIC_GPU="${AIC_REFIT_GPU:-$GPU_A}"
stage=validation_prediction; state running
AIC_V18_STAGE=predict-validation bash scripts/run_v18_evaluate_and_predict.sh
touch "${AIC_PIPELINE_DIR}/validation_prediction.done"
if [[ "$accepted" == 1 ]]; then
  export AIC_V18_RECIPE="$(support recipe)"
  case "$AIC_V18_RECIPE" in resolution384|rank32) ;; *) echo 'Matched control cannot win or refit' >&2; exit 2;; esac
  settings="$(support resource-settings --recipe "$AIC_V18_RECIPE")"
  read -r AIC_WORKERS AIC_BATCH_SIZE <<< "$settings"
  export AIC_WORKERS AIC_BATCH_SIZE
  export AIC_BENCHMARK_REPORT="$(support resource-field --recipe "$AIC_V18_RECIPE" --field benchmark)"
  export AIC_OFFICIAL_CHECK="$(support resource-field --recipe "$AIC_V18_RECIPE" --field official_check)"
  stage=full_data_refit; state running
  if [[ ! -f "${AIC_PIPELINE_DIR}/refit.done" ]]; then
    AIC_V18_STAGE=refit bash scripts/run_v18_train.sh
    [[ -f "${AIC_REFIT_DIR:-${AIC_OUTPUT_ROOT}/v18_refit}/model.pt" ]]
    touch "${AIC_PIPELINE_DIR}/refit.done"
  fi
  stage=refit_prediction; state running
  AIC_V18_STAGE=predict bash scripts/run_v18_evaluate_and_predict.sh
  touch "${AIC_PIPELINE_DIR}/prediction.done"
else
  [[ "$accepted" == 0 ]] || { echo 'Invalid V18 acceptance value' >&2; exit 2; }
  stage=fallback_delivery; state running
  AIC_V18_STAGE=fallback-refit bash scripts/run_v18_evaluate_and_predict.sh
  touch "${AIC_PIPELINE_DIR}/fallback.done"
fi
stage=complete; state complete
echo "v18_pipeline_complete refit_required=$accepted resource_branch=$(support resource-branch) delivery=${AIC_DELIVERY_DIR:-${AIC_OUTPUT_ROOT}/v18_delivery}"
