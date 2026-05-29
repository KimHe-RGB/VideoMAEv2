#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${RUN_DIR:?RUN_DIR is required}"

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"

FINETUNE_DATA_PATH="${FINETUNE_DATA_PATH:?FINETUNE_DATA_PATH is required}"
FINETUNE_DATA_ROOT="${FINETUNE_DATA_ROOT:-}"
EVAL_CKPT="${EVAL_CKPT:-${RUN_DIR}/finetune/exported/finetune_best.pth}"

EVAL_MODEL="${EVAL_MODEL:-vit_base_patch16_224}"
EVAL_DATA_SET="${EVAL_DATA_SET:-Kinetics-400}"
EVAL_NB_CLASSES="${EVAL_NB_CLASSES:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_INPUT_SIZE="${EVAL_INPUT_SIZE:-224}"
EVAL_SHORT_SIDE_SIZE="${EVAL_SHORT_SIDE_SIZE:-224}"
EVAL_NUM_FRAMES="${EVAL_NUM_FRAMES:-16}"
EVAL_SAMPLING_RATE="${EVAL_SAMPLING_RATE:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
EVAL_TEST_NUM_SEGMENT="${EVAL_TEST_NUM_SEGMENT:-5}"
EVAL_TEST_NUM_CROP="${EVAL_TEST_NUM_CROP:-3}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

mkdir -p "${RUN_DIR}/evaluation/output"
mkdir -p "${RUN_DIR}/metadata"

require_dir "${FINETUNE_DATA_PATH}"
require_file "${EVAL_CKPT}"

cmd_log="${RUN_DIR}/metadata/commands.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}"
  "${PROJECT_ROOT}/run_class_finetuning.py"
  --model "${EVAL_MODEL}"
  --data_set "${EVAL_DATA_SET}"
  --nb_classes "${EVAL_NB_CLASSES}"
  --data_path "${FINETUNE_DATA_PATH}"
  --data_root "${FINETUNE_DATA_ROOT}"
  --finetune "${EVAL_CKPT}"
  --batch_size "${EVAL_BATCH_SIZE}"
  --input_size "${EVAL_INPUT_SIZE}"
  --short_side_size "${EVAL_SHORT_SIDE_SIZE}"
  --num_frames "${EVAL_NUM_FRAMES}"
  --sampling_rate "${EVAL_SAMPLING_RATE}"
  --num_workers "${EVAL_NUM_WORKERS}"
  --test_num_segment "${EVAL_TEST_NUM_SEGMENT}"
  --test_num_crop "${EVAL_TEST_NUM_CROP}"
  --device "${EVAL_DEVICE}"
  --dist_eval
  --eval
  --output_dir "${RUN_DIR}/evaluation/output"
  --log_dir "${RUN_DIR}/evaluation/output"
)

if [[ -n "${EVAL_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EVAL_EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

log_cmd "${cmd_log}" "${cmd[@]}"
"${cmd[@]}"

summary_file="${RUN_DIR}/evaluation/output/summary.txt"
{
  echo "checkpoint=${EVAL_CKPT}"
  echo "log_file=${RUN_DIR}/evaluation/output/log.txt"
  echo "metrics:"
  if [[ -f "${RUN_DIR}/evaluation/output/log.txt" ]]; then
    grep -E 'Final top-1|Final Top-5' "${RUN_DIR}/evaluation/output/log.txt" || true
  else
    echo "log.txt missing"
  fi
} > "${summary_file}"

echo "[INFO] Evaluation done: ${summary_file}"
