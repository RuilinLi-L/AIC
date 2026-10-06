#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
RUN_DIR="${OUTPUT_ROOT}/v11_320"
GPU="${AIC_GPU:?Set AIC_GPU to a physical GPU index after checking nvidia-smi}"
BATCH_SIZE="${AIC_BATCH_SIZE:-64}"
GRAD_ACCUM="${AIC_GRAD_ACCUM:-4}"
WORKERS="${AIC_WORKERS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ ! "${GPU}" =~ ^[0-9]+$ || ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ || ! "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU, batch size and gradient accumulation must be valid integers" >&2
  exit 2
fi
if (( BATCH_SIZE * GRAD_ACCUM != 256 )); then
  echo "Keep effective batch 256: use 64/4 or 32/8" >&2
  exit 2
fi
feature_cache="${OUTPUT_ROOT}/v7/cache/frozen_partial.npy"
neighbor_cache="${OUTPUT_ROOT}/v9_coverage/neighbor_evidence.npz"
for required in "${feature_cache}" "${feature_cache}.json" "${neighbor_cache}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing reusable cache: ${required}" >&2
    exit 2
  fi
done
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v row="$((GPU + 1))" 'NR == row {gsub(/[^0-9]/, "", $1); print $1}')"
minimum_mib=24576
if (( BATCH_SIZE <= 32 )); then minimum_mib=16384; fi
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < minimum_mib )); then
  echo "Physical GPU ${GPU} needs ${minimum_mib} MiB free before launch; observed ${free_mib:-unknown}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"
resume_args=()
if [[ "${AIC_RESUME:-1}" == "0" && -f "${RUN_DIR}/resume_latest.pt" ]]; then
  echo "Refusing to overwrite an existing V11 run; archive it before fresh training" >&2
  exit 2
fi
if [[ "${AIC_RESUME:-1}" == "1" && -f "${RUN_DIR}/resume_latest.pt" ]]; then
  resume_args=(--resume "${RUN_DIR}/resume_latest.pt")
fi
cd "${PROJECT_DIR}"
echo "v11_base=v9_soft_teacher image_size=320 physical_gpu=${GPU} free_mib=${free_mib}" | tee -a "${RUN_DIR}/train.log"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_v11.py \
  --train-dir "${DATA_DIR}/train" \
  --data-manifest "${OUTPUT_ROOT}/v7/dataset_manifest_v7.json" \
  --conflict-policy partial --sampler repeat-factor \
  --model-dir "${MODEL_DIR}" --output-dir "${RUN_DIR}" \
  --feature-cache "${feature_cache}" --neighbor-cache "${neighbor_cache}" \
  --device cuda --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation "${GRAD_ACCUM}" --workers "${WORKERS}" \
  "${resume_args[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
