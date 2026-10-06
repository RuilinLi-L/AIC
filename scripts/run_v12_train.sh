#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${AIC_PROJECT_DIR:-/home/mcxu/lrl/AIC}"
PYTHON_BIN="${AIC_PYTHON:-/data/mcxu/conda-envs/aic/bin/python}"
DATA_DIR="${AIC_DATA_DIR:-${PROJECT_DIR}/data}"
MODEL_DIR="${AIC_MODEL_DIR:-${PROJECT_DIR}/clip-ViT-B-32}"
OUTPUT_ROOT="${AIC_OUTPUT_ROOT:-/data/mcxu/AIC/outputs}"
RUN_DIR="${OUTPUT_ROOT}/v12_lora_fast"
GPU="${AIC_GPU:?Set AIC_GPU to a physical GPU index after checking nvidia-smi}"
STAGE="${AIC_V12_STAGE:-train}"
BATCH_SIZE="${AIC_BATCH_SIZE:-256}"
WORKERS="${AIC_WORKERS:-16}"
PREFETCH="${AIC_PREFETCH_FACTOR:-2}"
EVAL_BATCH="${AIC_EVAL_BATCH_SIZE:-256}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if [[ ! "${GPU}" =~ ^[0-9]+$ || ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU and batch size must be valid integers" >&2; exit 2
fi
if (( 256 % BATCH_SIZE != 0 )); then
  echo "Batch size must divide effective batch 256" >&2; exit 2
fi
GRAD_ACCUM="${AIC_GRAD_ACCUM:-$((256 / BATCH_SIZE))}"
if [[ ! "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]] || (( BATCH_SIZE * GRAD_ACCUM != 256 )); then
  echo "Keep effective batch 256: 256/1, 128/2 or 64/4" >&2; exit 2
fi
if [[ "${STAGE}" != "train" && "${STAGE}" != "benchmark" ]]; then
  echo "AIC_V12_STAGE must be train or benchmark" >&2; exit 2
fi
for required in "${OUTPUT_ROOT}/v7/cache/frozen_partial.npy" "${OUTPUT_ROOT}/v7/cache/frozen_partial.npy.json" "${OUTPUT_ROOT}/v9_coverage/neighbor_evidence.npz"; do
  if [[ ! -f "${required}" ]]; then echo "Missing reusable cache: ${required}" >&2; exit 2; fi
done
free_mib="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
minimum_mib="${AIC_MIN_FREE_MIB:-24576}"
if [[ ! "${minimum_mib}" =~ ^[1-9][0-9]*$ || ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < minimum_mib )); then
  echo "GPU ${GPU}: free=${free_mib:-unknown} MiB, launch threshold=${minimum_mib} MiB (not a peak-memory guarantee)" >&2; exit 2
fi
args=(--train-dir "${DATA_DIR}/train" --data-manifest "${OUTPUT_ROOT}/v7/dataset_manifest_v7.json"
  --conflict-policy partial --sampler repeat-factor --model-dir "${MODEL_DIR}" --output-dir "${RUN_DIR}"
  --feature-cache "${OUTPUT_ROOT}/v7/cache/frozen_partial.npy" --neighbor-cache "${OUTPUT_ROOT}/v9_coverage/neighbor_evidence.npz"
  --device cuda --batch-size "${BATCH_SIZE}" --gradient-accumulation "${GRAD_ACCUM}"
  --workers "${WORKERS}" --prefetch-factor "${PREFETCH}" --eval-batch-size "${EVAL_BATCH}")
cd "${PROJECT_DIR}"
if [[ "${STAGE}" == "benchmark" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_v12.py "${args[@]}" --benchmark-only \
    --benchmark-steps "${AIC_BENCHMARK_STEPS:-10}" --benchmark-warmup-steps "${AIC_BENCHMARK_WARMUP_STEPS:-3}" \
    --benchmark-output "${AIC_BENCHMARK_OUTPUT:-${OUTPUT_ROOT}/v12_benchmark.json}"
else
  mkdir -p "${RUN_DIR}"
  if [[ -f "${RUN_DIR}/resume_latest.pt" ]]; then
    if [[ "${AIC_RESUME:-1}" != "1" ]]; then echo "Existing V12 run requires resume or a different output root" >&2; exit 2; fi
    args+=(--resume "${RUN_DIR}/resume_latest.pt")
  fi
  echo "v12_launch physical_gpu=${GPU} free_mib=${free_mib} batch=${BATCH_SIZE} accumulation=${GRAD_ACCUM} workers=${WORKERS}" | tee -a "${RUN_DIR}/train.log"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_v12.py "${args[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
fi
