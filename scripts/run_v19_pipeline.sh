#!/usr/bin/env bash
# Run inside screen. Freeze the engineering branch before either scientific run.
set -eEuo pipefail
while (($#)); do
  case "$1" in
    --gpu-a) export AIC_GPU_A="$2"; shift 2;;
    --gpu-b) export AIC_GPU_B="$2"; shift 2;;
    --gpu-control) export AIC_GPU_CONTROL="$2"; shift 2;;
    --output-root) export AIC_OUTPUT_ROOT="$2"; shift 2;;
    --project-dir) export AIC_PROJECT_DIR="$2"; shift 2;;
    --python) export AIC_PYTHON="$2"; shift 2;;
    --data-dir) export AIC_DATA_DIR="$2"; shift 2;;
    --model-dir) export AIC_MODEL_DIR="$2"; shift 2;;
    --min-free-mib) export AIC_MIN_FREE_MIB="$2"; shift 2;;
    --baseline-v18) export AIC_BASELINE_V18="$2"; shift 2;;
    --candidate-a) export AIC_CANDIDATE_A="$2"; shift 2;;
    --candidate-b) export AIC_CANDIDATE_B="$2"; shift 2;;
    --selection) export AIC_SELECTION="$2"; shift 2;;
    --audit-json) export AIC_AUDIT_JSON="$2"; shift 2;;
    --resource-json) export AIC_RESOURCE_JSON="$2"; shift 2;;
    *) echo "Unknown V19 pipeline argument: $1" >&2; exit 2;;
  esac
done
export AIC_PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
export AIC_PYTHON="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
export AIC_OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
export AIC_PIPELINE_DIR="${AIC_PIPELINE_DIR:-${AIC_OUTPUT_ROOT}/v19_pipeline}"
export AIC_SELECTION="${AIC_SELECTION:-${AIC_OUTPUT_ROOT}/v19_selection.json}"
export AIC_RESOURCE_JSON="${AIC_RESOURCE_JSON:-${AIC_PIPELINE_DIR}/resource_plan.json}"
export AIC_BASELINE_V18="${AIC_BASELINE_V18:-${AIC_OUTPUT_ROOT}/v18_rank32}"
GPU_A="${AIC_GPU_A:-${AIC_GPU:?Set AIC_GPU_A (or AIC_GPU) explicitly}}"
GPU_B="${AIC_GPU_B:-$GPU_A}"
AIC_V19_BRANCH="${AIC_V19_BRANCH:-standard}"
GPU_CONTROL="${AIC_GPU_CONTROL:-$GPU_A}"
[[ "$GPU_A" =~ ^[0-9]+$ && "$GPU_B" =~ ^[0-9]+$ ]] || { echo 'Explicit numeric GPU indices required' >&2; exit 2; }
if [[ "$AIC_V19_BRANCH" == deduplicated_control ]]; then
  [[ "$GPU_A" =~ ^[0-3]$ && "$GPU_B" =~ ^[0-3]$ && "$GPU_CONTROL" =~ ^[0-3]$
     && "${AIC_REFIT_GPU:-$GPU_A}" =~ ^[0-3]$ ]] || {
    echo 'Deduplicated V19 is restricted to physical GPUs 0,1,2,3' >&2; exit 2;
  }
else
  [[ "$AIC_V19_BRANCH" == standard ]] || { echo 'Unknown V19 data branch' >&2; exit 2; }
fi
mkdir -p "$AIC_PIPELINE_DIR"
cd "$AIC_PROJECT_DIR"
exec 9>"${AIC_PIPELINE_DIR}/.lock"
flock -n 9 || { echo 'V19 pipeline already active' >&2; exit 2; }
support() { "$AIC_PYTHON" v19_pipeline_support.py "$@" --root "$AIC_OUTPUT_ROOT"; }
stage=starting
state() { support state --stage "$stage" --status "$1"; }
trap 'state failed; echo "V19 pipeline failed during $stage; no scientific fallback is implied" >&2' ERR
trap 'state interrupted; exit 130' INT TERM
trap 'rc=$?; if ((rc != 0 && rc != 130)); then state failed; fi' EXIT
export AIC_PIPELINE_STARTED_AT="$(support init)"
state running
stage=data_audit; state running
export AIC_AUDIT_JSON="${AIC_AUDIT_JSON:-${AIC_OUTPUT_ROOT}/v19_startup/audit/audit.json}"
mkdir -p "$(dirname "$AIC_AUDIT_JSON")"
(
  exec 7>"$(dirname "$AIC_AUDIT_JSON")/.audit.lock"
  flock -n 7 || { echo 'V19 audit already running' >&2; exit 2; }
  if [[ ! -f "$AIC_AUDIT_JSON" ]]; then
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      "$AIC_PYTHON" -u audit_generalization_v19.py \
      --data-manifest "${AIC_MANIFEST:-${AIC_OUTPUT_ROOT}/v7/dataset_manifest_v7.json}" \
      --train-dir "${AIC_DATA_DIR:-${AIC_PROJECT_DIR}/data}/train" \
      --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}" \
      --feature-cache "${AIC_FEATURE_CACHE:-${AIC_OUTPUT_ROOT}/v13_expanded/cache/frozen.npy}" \
      --output-dir "$(dirname "$AIC_AUDIT_JSON")" --workers 8 \
      2>&1 | tee -a "${AIC_PIPELINE_DIR}/audit.log"
  fi
  support audit-check --audit-json "$AIC_AUDIT_JSON"
)
stage=baseline_binding; state running
support wait-baselines --baseline-v18 "$AIC_BASELINE_V18" \
  --timeout "${AIC_DEPENDENCY_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}"
preflight() (
  local recipe="$1" gpu="$2" run="$3" batch="$4" report="$5" official="$6"
  export AIC_V19_RECIPE="$recipe" AIC_GPU="$gpu" AIC_RUN_DIR="$run"
  mkdir -p "$run"
  exec 6>"${run}/.candidate.lock"
  flock -n 6 || { echo "$recipe preflight already active" >&2; exit 2; }
  acquire_gpu() {
    exec 5>"${AIC_PIPELINE_DIR}/gpu_${gpu}.lock"
    flock -w "${AIC_GPU_TIMEOUT:-172800}" 5
    support wait-gpu --gpu "$gpu" --minimum "${AIC_MIN_FREE_MIB:-45056}" \
      --timeout "${AIC_GPU_TIMEOUT:-172800}" --interval "${AIC_POLL_INTERVAL:-30}" \
      --state-file "${recipe}_preflight_gpu.json"
  }
  if [[ ! -f "$official" ]]; then
    (
      acquire_gpu
      CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$AIC_PYTHON" -u scripts/check_v19_official.py --recipe "$recipe" \
        --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}" \
        --output "$official" --device cuda 2>&1 | tee -a "${run}/official_check.log"
    )
  fi
  support official-check --report "$official" --recipe "$recipe" \
    --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}"
  needed="$(support benchmark-needed --benchmark "$report" --recipe "$recipe")"
  if [[ "$needed" == 1 ]]; then
    (
      acquire_gpu
      CUDA_VISIBLE_DEVICES="$gpu" AIC_PIN_MEMORY=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$AIC_PYTHON" -u benchmark_v19.py --recipe "$recipe" \
        --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}" \
        --train-dir "${AIC_DATA_DIR:-${AIC_PROJECT_DIR}/data}/train" \
        --data-manifest "${AIC_MANIFEST:-${AIC_OUTPUT_ROOT}/v7/dataset_manifest_v7.json}" \
        --checkpoint "${AIC_BENCHMARK_METADATA:-${AIC_OUTPUT_ROOT}/v18_rank32/best_model.pt}" \
        --feature-cache "${AIC_FEATURE_CACHE:-${AIC_OUTPUT_ROOT}/v13_expanded/cache/frozen.npy}" \
        --output "$report" --device cuda --steps 8 --warmup-steps 2 \
        --batch-size "$batch" --gradient-accumulation "$((256 / batch))" \
        2>&1 | tee -a "${run}/benchmark.log"
    )
  fi
  support benchmark-status --benchmark "$report" --recipe "$recipe"
)
stage=resource_preflight; state running
if [[ ! -f "$AIC_RESOURCE_JSON" ]]; then
  export AIC_CANDIDATE_A="${AIC_CANDIDATE_A:-${AIC_OUTPUT_ROOT}/v19_resolution384_rank32}"
  export AIC_CANDIDATE_B="${AIC_CANDIDATE_B:-${AIC_OUTPUT_ROOT}/v19_rank32_dropout}"
  benchmark_a="${AIC_BENCHMARK_A:-${AIC_CANDIDATE_A}/benchmark.json}"
  official_a="${AIC_OFFICIAL_CHECK_A:-${AIC_CANDIDATE_A}/official_check.json}"
  benchmark_b="${AIC_BENCHMARK_B:-${AIC_CANDIDATE_B}/benchmark.json}"
  official_b="${AIC_OFFICIAL_CHECK_B:-${AIC_CANDIDATE_B}/official_check.json}"
  if [[ "$AIC_V19_BRANCH" == deduplicated_control ]]; then
    export AIC_CONTROL_RUN="${AIC_CONTROL_RUN:-${AIC_OUTPUT_ROOT}/v19_rank32_control}"
    benchmark_control="${AIC_BENCHMARK_CONTROL:-${AIC_CONTROL_RUN}/benchmark.json}"
    official_control="${AIC_OFFICIAL_CHECK_CONTROL:-${AIC_CONTROL_RUN}/official_check.json}"
    # Preflight is short and serialized on a ready card. Formal runs use their assigned cards.
    preflight resolution384_rank32 "$GPU_A" "$AIC_CANDIDATE_A" 256 "$benchmark_a" "$official_a"
    preflight rank32_dropout "$GPU_A" "$AIC_CANDIDATE_B" 256 "$benchmark_b" "$official_b"
    preflight rank32_control "$GPU_A" "$AIC_CONTROL_RUN" 256 "$benchmark_control" "$official_control"
  elif [[ "$GPU_A" != "$GPU_B" ]]; then
    (exec 9>&-; preflight resolution384_rank32 "$GPU_A" "$AIC_CANDIDATE_A" 256 "$benchmark_a" "$official_a") >"${AIC_PIPELINE_DIR}/preflight_a.log" 2>&1 & pa=$!
    (exec 9>&-; preflight rank32_dropout "$GPU_B" "$AIC_CANDIDATE_B" 256 "$benchmark_b" "$official_b") >"${AIC_PIPELINE_DIR}/preflight_b.log" 2>&1 & pb=$!
    failed=0; wait "$pa" || failed=1; wait "$pb" || failed=1
    ((failed == 0)) || { echo 'V19 preflight failed; no automatic retry' >&2; exit 1; }
  else
    preflight resolution384_rank32 "$GPU_A" "$AIC_CANDIDATE_A" 256 "$benchmark_a" "$official_a"
    preflight rank32_dropout "$GPU_B" "$AIC_CANDIDATE_B" 256 "$benchmark_b" "$official_b"
  fi
  freeze_args=(--resource-json "$AIC_RESOURCE_JSON" --audit-json "$AIC_AUDIT_JSON" \
    --benchmark-a "$benchmark_a" --benchmark-b "$benchmark_b" --official-a "$official_a" --official-b "$official_b" \
    --candidate-a "$AIC_CANDIDATE_A" --candidate-b "$AIC_CANDIDATE_B" --baseline-v18 "$AIC_BASELINE_V18" \
    --model-dir "${AIC_MODEL_DIR:-${AIC_PROJECT_DIR}/clip-ViT-B-32}")
  if [[ "$AIC_V19_BRANCH" == deduplicated_control ]]; then
    freeze_args+=(--branch deduplicated_control --benchmark-control "$benchmark_control" \
      --official-control "$official_control" --control-run "$AIC_CONTROL_RUN" \
      --dedup-record "${AIC_DEDUP_RECORD:?Set AIC_DEDUP_RECORD for deduplicated training}" \
      --source-manifest "${AIC_SOURCE_MANIFEST:?Set AIC_SOURCE_MANIFEST for deduplicated training}")
  fi
  support freeze-resources "${freeze_args[@]}"

fi
support resource-check --resource-json "$AIC_RESOURCE_JSON"
frozen_a="$(support resource-field --candidate candidate_a --field run_dir)"
frozen_b="$(support resource-field --candidate candidate_b --field run_dir)"
[[ "${AIC_CANDIDATE_A:-$frozen_a}" == "$frozen_a" && "${AIC_CANDIDATE_B:-$frozen_b}" == "$frozen_b" ]] || {
  echo 'Candidate directories differ from frozen V19 resources' >&2; exit 2;
}
export AIC_CANDIDATE_A="$frozen_a" AIC_CANDIDATE_B="$frozen_b"
second_recipe="$(support resource-field --candidate candidate_b --field recipe)"
if [[ "$AIC_V19_BRANCH" == deduplicated_control ]]; then
  export AIC_CONTROL_RUN="$(support resource-field --candidate control --field run_dir)"
  [[ "${AIC_BASELINE_V18}" != "$AIC_CONTROL_RUN" ]] || { echo 'Historical and clean control directories collide' >&2; exit 2; }
  export AIC_BASELINE_V18="$AIC_CONTROL_RUN"
fi
candidate() (
  local recipe="$1" gpu="$2" run="$3" candidate_stage=starting
  if [[ "$AIC_V19_BRANCH" == deduplicated_control ]]; then
    # Keep one GPU lease through training and evaluation. Waiting runs may use any
    # of the user's permitted cards as soon as it has enough free memory.
    local lease_fd free_mib ready=0
    local started_wait="$(date +%s)"
    while ((ready == 0)); do
      for gpu in 0 1 2 3; do
        exec {lease_fd}>"${AIC_PIPELINE_DIR}/gpu_${gpu}.lock"
        if flock -n "$lease_fd"; then
          free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
          if [[ "$free_mib" =~ ^[0-9]+$ ]] && ((free_mib >= ${AIC_MIN_FREE_MIB:-45056})); then
            export AIC_GPU_LEASE_FD="$lease_fd"
            ready=1
            break
          fi
          flock -u "$lease_fd"
        fi
        exec {lease_fd}>&-
      done
      if ((ready == 0)); then
        if (( $(date +%s) - started_wait >= ${AIC_GPU_TIMEOUT:-172800} )); then
          echo "No GPU 0/1/2/3 reached ${AIC_MIN_FREE_MIB:-45056} MiB before queue timeout" >&2
          exit 1
        fi
        sleep "${AIC_POLL_INTERVAL:-30}"
      fi
    done
    echo "v19_gpu_pool recipe=$recipe physical_gpu=$gpu free_mib=$free_mib" >&2
  fi
  export AIC_V19_RECIPE="$recipe" AIC_GPU="$gpu" AIC_RUN_DIR="$run"
  exec 6>"${run}/.candidate.lock"
  flock -n 6 || { echo "$recipe candidate already active" >&2; exit 2; }
  candidate_state() { support state --stage "$candidate_stage" --status "$1" --state-file "${recipe}_status.json"; }
  trap 'candidate_state failed; echo "V19 $recipe failed during $candidate_stage" >&2' ERR
  trap 'candidate_state interrupted; exit 130' INT TERM
  trap 'rc=$?; if ((rc != 0 && rc != 130)); then candidate_state failed; fi' EXIT
  if [[ "$(support resource-field --recipe "$recipe" --field status)" == resource_infeasible ]]; then
    candidate_stage=resource_infeasible; candidate_state skipped
    exit 0
  fi
  settings="$(support resource-settings --recipe "$recipe")"
  read -r AIC_WORKERS AIC_BATCH_SIZE <<< "$settings"
  export AIC_WORKERS AIC_BATCH_SIZE
  export AIC_BENCHMARK_REPORT="$(support resource-field --recipe "$recipe" --field benchmark)"
  export AIC_OFFICIAL_CHECK="$(support resource-field --recipe "$recipe" --field official_check)"
  candidate_stage=validation; candidate_state running
  if [[ ! -f "${run}/validation.done" ]]; then
    AIC_V19_STAGE=validate bash scripts/run_v19_train.sh
    [[ -f "${run}/best_model.pt" ]] || { echo 'Training returned without best_model.pt' >&2; exit 2; }
    touch "${run}/validation.done"
  fi
  candidate_stage=evaluation; candidate_state running
  if [[ ! -f "${run}/evaluation.done" ]]; then
    AIC_V19_STAGE=evaluate bash scripts/run_v19_evaluate_and_predict.sh
    [[ -f "${run}/strict_eval.json" && -f "${run}/model.pt" ]]
    touch "${run}/evaluation.done"
  fi
  candidate_stage=complete; candidate_state complete
)
stage=candidate_validation_and_evaluation; state running
if [[ "$AIC_V19_BRANCH" == deduplicated_control ]]; then
  (exec 9>&-; candidate resolution384_rank32 "$GPU_A" "$AIC_CANDIDATE_A") >"${AIC_PIPELINE_DIR}/resolution384_rank32.log" 2>&1 & pid_a=$!
  (exec 9>&-; candidate "$second_recipe" "$GPU_B" "$AIC_CANDIDATE_B") >"${AIC_PIPELINE_DIR}/${second_recipe}.log" 2>&1 & pid_b=$!
  (exec 9>&-; candidate rank32_control "$GPU_CONTROL" "$AIC_CONTROL_RUN") >"${AIC_PIPELINE_DIR}/rank32_control.log" 2>&1 & pid_control=$!
  failure=0
  wait "$pid_a" || failure=1; wait "$pid_b" || failure=1; wait "$pid_control" || failure=1
  ((failure == 0)) || { state failed; echo 'At least one V19 clean run failed; resume after diagnosis' >&2; exit 1; }
elif [[ "$GPU_A" != "$GPU_B" ]]; then
  (exec 9>&-; candidate resolution384_rank32 "$GPU_A" "$AIC_CANDIDATE_A") >"${AIC_PIPELINE_DIR}/resolution384_rank32.log" 2>&1 &
  pid_a=$!
  (exec 9>&-; candidate "$second_recipe" "$GPU_B" "$AIC_CANDIDATE_B") >"${AIC_PIPELINE_DIR}/${second_recipe}.log" 2>&1 &
  pid_b=$!
  failure=0
  wait "$pid_a" || failure=1
  wait "$pid_b" || failure=1
  if ((failure)); then state failed; echo 'At least one V19 candidate failed; resume after diagnosis' >&2; exit 1; fi
else
  candidate resolution384_rank32 "$GPU_A" "$AIC_CANDIDATE_A" 2>&1 | tee -a "${AIC_PIPELINE_DIR}/resolution384_rank32.log"
  candidate "$second_recipe" "$GPU_B" "$AIC_CANDIDATE_B" 2>&1 | tee -a "${AIC_PIPELINE_DIR}/${second_recipe}.log"
fi
stage=selection; state running
if [[ ! -f "${AIC_PIPELINE_DIR}/selection.done" ]]; then
  if [[ -e "${AIC_REFIT_DIR:-${AIC_OUTPUT_ROOT}/v19_refit}/resume_latest.pt" ]]; then
    support bind
  else
    AIC_V19_STAGE=select bash scripts/run_v19_evaluate_and_predict.sh
  fi
  touch "${AIC_PIPELINE_DIR}/selection.done"
fi
support bind
accepted="$(support accepted)"
export AIC_GPU="${AIC_REFIT_GPU:-$GPU_A}"
stage=validation_prediction; state running
AIC_V19_STAGE=predict-validation bash scripts/run_v19_evaluate_and_predict.sh
touch "${AIC_PIPELINE_DIR}/validation_prediction.done"
if [[ "$accepted" == 1 ]]; then
  export AIC_V19_RECIPE="$(support recipe)"
  case "$AIC_V19_RECIPE" in resolution384_rank32|rank32_dropout) ;; *) echo 'Only a runnable V19 winner may refit' >&2; exit 2;; esac
  settings="$(support resource-settings --recipe "$AIC_V19_RECIPE")"
  read -r AIC_WORKERS AIC_BATCH_SIZE <<< "$settings"
  export AIC_WORKERS AIC_BATCH_SIZE
  export AIC_BENCHMARK_REPORT="$(support resource-field --recipe "$AIC_V19_RECIPE" --field benchmark)"
  export AIC_OFFICIAL_CHECK="$(support resource-field --recipe "$AIC_V19_RECIPE" --field official_check)"
  stage=full_data_refit; state running
  if [[ ! -f "${AIC_PIPELINE_DIR}/refit.done" ]]; then
    AIC_V19_STAGE=refit bash scripts/run_v19_train.sh
    [[ -f "${AIC_REFIT_DIR:-${AIC_OUTPUT_ROOT}/v19_refit}/model.pt" ]]
    touch "${AIC_PIPELINE_DIR}/refit.done"
  fi
  stage=refit_prediction; state running
  AIC_V19_STAGE=predict bash scripts/run_v19_evaluate_and_predict.sh
  touch "${AIC_PIPELINE_DIR}/prediction.done"
else
  [[ "$accepted" == 0 ]] || { echo 'Invalid V19 acceptance value' >&2; exit 2; }
  stage=fallback_delivery; state running
  AIC_V19_STAGE=fallback-refit bash scripts/run_v19_evaluate_and_predict.sh
  touch "${AIC_PIPELINE_DIR}/fallback.done"
fi
stage=complete; state complete
echo "v19_pipeline_complete refit_required=$accepted resource_branch=$(support resource-branch) delivery=${AIC_DELIVERY_DIR:-${AIC_OUTPUT_ROOT}/v19_delivery}"
