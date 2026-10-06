#!/usr/bin/env bash
# Run in screen; after validation, continue unattended to a validated ZIP.
set -euo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
export AIC_GPU="${AIC_GPU:?Set the V14 training GPU}"
BASELINE_GPU="${AIC_BASELINE_GPU:?Set a distinct baseline evaluation GPU}"
[[ "$AIC_GPU" != "$BASELINE_GPU" ]] || { echo "Use distinct GPUs for the two initial jobs" >&2; exit 2; }
PIPELINE_DIR="${OUTPUT_ROOT}/v14_pipeline"
mkdir -p "$PIPELINE_DIR"
exec 9>"${PIPELINE_DIR}/.lock"
flock -n 9 || { echo "V14 pipeline already active" >&2; exit 2; }
cd "$PROJECT_DIR"
BENCHMARK="${AIC_BENCHMARK:-${OUTPUT_ROOT}/v14_benchmark.json}"
read -r AIC_WORKERS AIC_BATCH_SIZE < <("$PYTHON_BIN" - "$BENCHMARK" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
workers, batch = int(d['chosen_workers']), int(d['batch_size'])
assert workers in (4, 8) and batch in (128, 256)
assert int(d['gradient_accumulation']) * batch == 256
assert d['pin_memory'] is False and int(d['prefetch_factor']) == 1
print(workers, batch)
PY
)
export AIC_WORKERS AIC_BATCH_SIZE AIC_PIN_MEMORY=0 AIC_PREFETCH_FACTOR=1
stage=starting
state() {
  "$PYTHON_BIN" - "${PIPELINE_DIR}/status.json" "$stage" "$1" <<'PY'
import datetime, json, os, sys
path, stage, status = sys.argv[1:]
tmp = path + '.tmp'
with open(tmp, 'w') as f:
    json.dump({'stage': stage, 'status': status, 'updated_at': datetime.datetime.now().astimezone().isoformat()}, f, indent=2)
os.replace(tmp, path)
PY
}
trap 'state failed; echo "V14 pipeline failed during $stage; inspect stage logs" >&2' ERR
stage=baseline_and_validation; state running
# Baseline writes only its new evaluation directory; original V13 files are inputs.
baseline_pid=""
if [[ ! -f "${PIPELINE_DIR}/baseline.done" ]]; then
  (AIC_GPU="$BASELINE_GPU" AIC_V14_STAGE=baseline bash scripts/run_v14_evaluate_and_predict.sh && touch "${PIPELINE_DIR}/baseline.done") > "${PIPELINE_DIR}/baseline.log" 2>&1 &
  baseline_pid=$!
fi
if [[ ! -f "${PIPELINE_DIR}/validation.done" ]]; then
  AIC_V14_STAGE=validate bash scripts/run_v14_train.sh
  touch "${PIPELINE_DIR}/validation.done"
fi
stage=waiting_for_baseline; state running
if [[ -n "$baseline_pid" ]]; then wait "$baseline_pid"; fi
stage=candidate_evaluation; state running
if [[ ! -f "${PIPELINE_DIR}/evaluation.done" ]]; then
  AIC_V14_STAGE=evaluate bash scripts/run_v14_evaluate_and_predict.sh
  touch "${PIPELINE_DIR}/evaluation.done"
fi
stage=selection; state running
# Preserve the selection that an existing refit resume checkpoint is bound to.
if [[ ! -f "${PIPELINE_DIR}/selection.done" && ! -e "${OUTPUT_ROOT}/v14_refit/resume_latest.pt" ]]; then
  AIC_V14_STAGE=select bash scripts/run_v14_evaluate_and_predict.sh
  touch "${PIPELINE_DIR}/selection.done"
fi
stage=full_data_refit; state running
if [[ ! -f "${PIPELINE_DIR}/refit.done" ]]; then
  AIC_V14_STAGE=refit bash scripts/run_v14_train.sh
  touch "${PIPELINE_DIR}/refit.done"
fi
stage=prediction; state running
if [[ ! -f "${PIPELINE_DIR}/prediction.done" ]]; then
  AIC_V14_STAGE=predict bash scripts/run_v14_evaluate_and_predict.sh
  touch "${PIPELINE_DIR}/prediction.done"
fi
stage=complete; state complete
echo "v14_pipeline_complete zip=${OUTPUT_ROOT}/v14_refit/pred_results.zip"
