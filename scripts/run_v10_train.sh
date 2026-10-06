#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
RUN_DIR="${OUTPUT_ROOT}/v10_robust_views"
GPU="${AIC_GPU:?Set AIC_GPU to a physical GPU index after checking nvidia-smi}"
BATCH_SIZE="${AIC_BATCH_SIZE:-64}"
GRAD_ACCUM="${AIC_GRAD_ACCUM:-4}"
WORKERS="${AIC_WORKERS:-4}"

selection_args=()
if [[ -n "${AIC_V10_BASE:-}" ]]; then
  case "${AIC_V10_BASE}" in
    v9_coverage|v9_soft_teacher) selected_run="${AIC_V10_BASE}" ;;
    *) echo "AIC_V10_BASE must be v9_coverage or v9_soft_teacher" >&2; exit 2 ;;
  esac
  selection_args=(--v9-base "${selected_run}")
else
  selection="${OUTPUT_ROOT}/v9/selection.json"
  if [[ ! -f "${selection}" ]]; then
    echo "V9 selection is not finished: missing ${selection}" >&2
    exit 2
  fi
  selected_run="$("${PYTHON_BIN}" - "${selection}" <<'PY'
import json
import sys
run = json.load(open(sys.argv[1], encoding="utf-8")).get("selected_run")
if run not in {"v9_coverage", "v9_soft_teacher"}:
    raise SystemExit(f"invalid V9 selected_run: {run!r}")
print(run)
PY
  )"
  selection_args=(--selection-json "${selection}")
fi
metrics="${OUTPUT_ROOT}/${selected_run}/metrics.json"
if [[ ! -f "${metrics}" ]]; then
  echo "V9 base run lacks metrics.json: ${selected_run}" >&2
  exit 2
fi
"${PYTHON_BIN}" - "${metrics}" <<'PY'
import json
import sys
metrics = json.load(open(sys.argv[1], encoding="utf-8"))
if len(metrics.get("validation_history", [])) != 12:
    raise SystemExit("V9 base run has not completed all 12 validation epochs")
PY
neighbor_cache="${OUTPUT_ROOT}/v9_coverage/neighbor_evidence.npz"
if [[ ! -f "${neighbor_cache}" ]]; then
  echo "Missing reusable V9 neighbor cache: ${neighbor_cache}" >&2
  exit 2
fi
if [[ ! "${GPU}" =~ ^[0-9]+$ ]]; then
  echo "AIC_GPU must be a nonnegative physical GPU index" >&2
  exit 2
fi
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v row="$((GPU + 1))" 'NR == row {gsub(/[^0-9]/, "", $1); print $1}')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
  echo "Could not read free memory for physical GPU ${GPU}" >&2
  exit 2
fi
if (( free_mib < 12288 )); then
  echo "Physical GPU ${GPU} has ${free_mib} MiB free; V10 requires at least 12288 MiB before launch" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"
resume_args=()
if [[ "${AIC_RESUME:-1}" == "0" && -f "${RUN_DIR}/resume_latest.pt" ]]; then
  echo "Refusing fresh V10 training in a directory with resume_latest.pt; archive the old run first" >&2
  exit 2
fi
if [[ "${AIC_RESUME:-1}" == "1" && -f "${RUN_DIR}/resume_latest.pt" ]]; then
  resume_args=(--resume "${RUN_DIR}/resume_latest.pt")
fi
cd "${PROJECT_DIR}"
echo "v10_base=${selected_run} physical_gpu=${GPU} free_mib=${free_mib}" | tee -a "${RUN_DIR}/train.log"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_v10.py \
  --train-dir "${DATA_DIR}/train" \
  --data-manifest "${OUTPUT_ROOT}/v7/dataset_manifest_v7.json" \
  --conflict-policy partial \
  --sampler repeat-factor \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${RUN_DIR}" \
  --feature-cache "${OUTPUT_ROOT}/v7/cache/frozen_partial.npy" \
  --neighbor-cache "${neighbor_cache}" \
  "${selection_args[@]}" \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation "${GRAD_ACCUM}" \
  --workers "${WORKERS}" \
  "${resume_args[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
