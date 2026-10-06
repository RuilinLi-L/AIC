#!/usr/bin/env bash
# Run inside screen; V14 remains a read-only dependency throughout this pipeline.
set -eEuo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
export AIC_GPU="${AIC_GPU:?Set a V15 GPU independent of the active V14 run}"
[[ "$AIC_GPU" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index" >&2; exit 2; }
DELIVERY_GPU="${AIC_DELIVERY_GPU:-}"
if [[ -n "$DELIVERY_GPU" && ( ! "$DELIVERY_GPU" =~ ^[0-9]+$ || "$DELIVERY_GPU" == "$AIC_GPU" ) ]]; then
  echo "AIC_DELIVERY_GPU must differ from the V15 training GPU" >&2; exit 2
fi
PIPELINE_DIR="${OUTPUT_ROOT}/v15_pipeline"
mkdir -p "$PIPELINE_DIR"
exec 9>"${PIPELINE_DIR}/.lock"
flock -n 9 || { echo "V15 pipeline already active" >&2; exit 2; }
cd "$PROJECT_DIR"
support() { "$PYTHON_BIN" v15_pipeline_support.py "$@" --root "$OUTPUT_ROOT"; }
stage=starting
state() { support state --stage "$stage" --status "$1"; }
delivery_pid=""
cleanup() {
  if [[ -n "$delivery_pid" ]] && kill -0 "$delivery_pid" 2>/dev/null; then
    # The delivery child owns its process group; never signal the V14 pipeline.
    "$PYTHON_BIN" - "$delivery_pid" <<'PY'
import os, signal, sys
try: os.killpg(int(sys.argv[1]), signal.SIGTERM)
except ProcessLookupError: pass
PY
    wait "$delivery_pid" || true
  fi
}
trap cleanup EXIT
trap 'state failed; echo "V15 pipeline failed during $stage; inspect stage logs" >&2' ERR
trap 'state interrupted; exit 130' INT TERM
if [[ -n "$DELIVERY_GPU" && ! -f "${PIPELINE_DIR}/v14_pair.done" ]]; then
  # Drop the parent's lock descriptor, then create a separately cancellable group.
  (exec 9>&-; exec setsid env AIC_GPU="$DELIVERY_GPU" AIC_V15_STAGE=v14-pair \
    bash scripts/run_v15_evaluate_and_predict.sh) >"${PIPELINE_DIR}/v14_delivery.log" 2>&1 &
  delivery_pid=$!
fi
BENCHMARK="${AIC_BENCHMARK:-${OUTPUT_ROOT}/v15_benchmark.json}"
stage=benchmark; state running
if [[ ! -f "$BENCHMARK" ]]; then
  CUDA_VISIBLE_DEVICES="$AIC_GPU" AIC_PIN_MEMORY=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PYTHON_BIN" -u benchmark_v15.py --model-dir "$MODEL_DIR" --train-dir "${DATA_DIR}/train" \
    --data-manifest "${AIC_MANIFEST:-${OUTPUT_ROOT}/v7/dataset_manifest_v7.json}" \
    --checkpoint "${AIC_BENCHMARK_METADATA:-${OUTPUT_ROOT}/v13_expanded/best_model.pt}" \
    --feature-cache "${AIC_FEATURE_CACHE:-${OUTPUT_ROOT}/v13_expanded/cache/frozen.npy}" \
    --output "$BENCHMARK" --device cuda --steps 8 --warmup-steps 2 2>&1 | tee -a "${PIPELINE_DIR}/benchmark.log"
fi
settings="$("$PYTHON_BIN" - "$BENCHMARK" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d['format_version'] == 15 and d['recipe'] == 'expanded_mlp' and d['success'] is True
assert d['initialization'] == 'official_base_only' and d['trained_checkpoint_weights_loaded'] is False
workers, batch = int(d['chosen_workers']), int(d['batch_size'])
assert workers == 8 and batch in (128, 256) and int(d['gradient_accumulation']) * batch == 256
assert d['pin_memory'] is False and int(d['prefetch_factor']) == 1
assert batch == 256 or any(c['batch_size'] == 256 and c['status'] == 'out_of_memory' for c in d['configurations'])
print(workers, batch)
PY
)"
read -r AIC_WORKERS AIC_BATCH_SIZE <<< "$settings"
export AIC_WORKERS AIC_BATCH_SIZE AIC_PIN_MEMORY=0 AIC_PREFETCH_FACTOR=1
echo "v15_budget target_hours=48 schedule_epochs=24 refit=selected_epoch benchmark_excludes_audits_validation_and_waits=1"
stage=validation; state running
if [[ ! -f "${PIPELINE_DIR}/validation.done" ]]; then
  AIC_V15_STAGE=validate bash scripts/run_v15_train.sh
  touch "${PIPELINE_DIR}/validation.done"
fi
stage=candidate_evaluation; state running
if [[ ! -f "${PIPELINE_DIR}/evaluation.done" ]]; then
  AIC_V15_STAGE=evaluate bash scripts/run_v15_evaluate_and_predict.sh
  touch "${PIPELINE_DIR}/evaluation.done"
fi
stage=selection; state running
if [[ ! -f "${PIPELINE_DIR}/selection.done" && ! -e "${OUTPUT_ROOT}/v15_refit/resume_latest.pt" ]]; then
  AIC_V15_STAGE=select bash scripts/run_v15_evaluate_and_predict.sh
  touch "${PIPELINE_DIR}/selection.done"
fi
accepted="$(support accepted)"
if [[ "$accepted" == 1 ]]; then
  stage=validation_prediction; state running
  if [[ ! -f "${PIPELINE_DIR}/validation_prediction.done" ]]; then
    AIC_V15_STAGE=predict-validation bash scripts/run_v15_evaluate_and_predict.sh
    touch "${PIPELINE_DIR}/validation_prediction.done"
  fi
  stage=full_data_refit; state running
  if [[ ! -f "${PIPELINE_DIR}/refit.done" ]]; then
    AIC_V15_STAGE=refit bash scripts/run_v15_train.sh
    touch "${PIPELINE_DIR}/refit.done"
  fi
  stage=prediction; state running
  if [[ ! -f "${PIPELINE_DIR}/prediction.done" ]]; then
    AIC_V15_STAGE=predict bash scripts/run_v15_evaluate_and_predict.sh
    touch "${PIPELINE_DIR}/prediction.done"
  fi
else
  [[ "$accepted" == 0 ]] || { echo "Invalid candidate acceptance" >&2; exit 2; }
  echo "v15_candidate_rejected fallback=v14 no_v15_refit=1"
fi
stage=v14_delivery; state running
if [[ -n "$delivery_pid" ]]; then
  wait "$delivery_pid"
  delivery_pid=""
elif [[ ! -f "${PIPELINE_DIR}/v14_pair.done" ]]; then
  AIC_V15_STAGE=v14-pair bash scripts/run_v15_evaluate_and_predict.sh
fi
stage=complete; state complete
echo "v15_pipeline_complete candidate_accepted=$accepted delivery=${OUTPUT_ROOT}/v15_delivery"
