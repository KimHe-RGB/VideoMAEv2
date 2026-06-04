#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${RUN_DIR:?RUN_DIR is required}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"

FINETUNE_DATA_PATH="${FINETUNE_DATA_PATH:?FINETUNE_DATA_PATH is required}"
FINETUNE_DATA_ROOT="${FINETUNE_DATA_ROOT:-}"
FINETUNE_INIT_CKPT="${FINETUNE_INIT_CKPT:-${RUN_DIR}/post_pretrain/exported/post_pretrain_latest.pth}"

FIN_MODEL="${FIN_MODEL:-vit_base_patch16_224}"
FIN_DATA_SET="${FIN_DATA_SET:-Kinetics-400}"
FIN_NB_CLASSES="${FIN_NB_CLASSES:-2}"
FIN_EPOCHS="${FIN_EPOCHS:-35}"
FIN_BATCH_SIZE="${FIN_BATCH_SIZE:-8}"
FIN_INPUT_SIZE="${FIN_INPUT_SIZE:-224}"
FIN_SHORT_SIDE_SIZE="${FIN_SHORT_SIDE_SIZE:-224}"
FIN_SAVE_CKPT_FREQ="${FIN_SAVE_CKPT_FREQ:-5}"
FIN_NUM_FRAMES="${FIN_NUM_FRAMES:-16}"
FIN_SAMPLING_RATE="${FIN_SAMPLING_RATE:-4}"
FIN_NUM_SAMPLE="${FIN_NUM_SAMPLE:-2}"
FIN_NUM_WORKERS="${FIN_NUM_WORKERS:-4}"
FIN_OPT="${FIN_OPT:-adamw}"
FIN_LR="${FIN_LR:-1e-3}"
FIN_DROP_PATH="${FIN_DROP_PATH:-0.3}"
FIN_CLIP_GRAD="${FIN_CLIP_GRAD:-5.0}"
FIN_LAYER_DECAY="${FIN_LAYER_DECAY:-0.9}"
FIN_WEIGHT_DECAY="${FIN_WEIGHT_DECAY:-0.1}"
FIN_WARMUP_EPOCHS="${FIN_WARMUP_EPOCHS:-5}"
FIN_TEST_NUM_SEGMENT="${FIN_TEST_NUM_SEGMENT:-5}"
FIN_TEST_NUM_CROP="${FIN_TEST_NUM_CROP:-3}"
FIN_DEVICE="${FIN_DEVICE:-cuda}"

mkdir -p "${RUN_DIR}/finetune/logs"
mkdir -p "${RUN_DIR}/finetune/exported"
mkdir -p "${RUN_DIR}/data/finetune_split_snapshot"
mkdir -p "${RUN_DIR}/metadata"

require_dir "${FINETUNE_DATA_PATH}"
copy_finetune_splits_snapshot "${FINETUNE_DATA_PATH}" "${RUN_DIR}/data/finetune_split_snapshot"
require_file "${FINETUNE_INIT_CKPT}"

cmd_log="${RUN_DIR}/metadata/commands.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}"
  "${PROJECT_ROOT}/run_class_finetuning.py"
  --model "${FIN_MODEL}"
  --data_set "${FIN_DATA_SET}"
  --nb_classes "${FIN_NB_CLASSES}"
  --data_path "${FINETUNE_DATA_PATH}"
  --data_root "${FINETUNE_DATA_ROOT}"
  --finetune "${FINETUNE_INIT_CKPT}"
  --log_dir "${RUN_DIR}/finetune/logs"
  --output_dir "${RUN_DIR}/finetune/logs"
  --epochs "${FIN_EPOCHS}"
  --batch_size "${FIN_BATCH_SIZE}"
  --input_size "${FIN_INPUT_SIZE}"
  --short_side_size "${FIN_SHORT_SIDE_SIZE}"
  --save_ckpt_freq "${FIN_SAVE_CKPT_FREQ}"
  --num_frames "${FIN_NUM_FRAMES}"
  --sampling_rate "${FIN_SAMPLING_RATE}"
  --num_sample "${FIN_NUM_SAMPLE}"
  --num_workers "${FIN_NUM_WORKERS}"
  --opt "${FIN_OPT}"
  --lr "${FIN_LR}"
  --drop_path "${FIN_DROP_PATH}"
  --clip_grad "${FIN_CLIP_GRAD}"
  --layer_decay "${FIN_LAYER_DECAY}"
  --weight_decay "${FIN_WEIGHT_DECAY}"
  --warmup_epochs "${FIN_WARMUP_EPOCHS}"
  --test_num_segment "${FIN_TEST_NUM_SEGMENT}"
  --test_num_crop "${FIN_TEST_NUM_CROP}"
  --device "${FIN_DEVICE}"
  --dist_eval
)

if [[ -n "${FIN_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${FIN_EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

log_cmd "${cmd_log}" "${cmd[@]}"
"${cmd[@]}"

best_ckpt="${RUN_DIR}/finetune/logs/checkpoint-best.pth"
if [[ -f "${best_ckpt}" ]]; then
  cp -f "${best_ckpt}" "${RUN_DIR}/finetune/exported/finetune_best.pth"
else
  latest_ckpt="$(latest_checkpoint_in_dir "${RUN_DIR}/finetune/logs")"
  if [[ -z "${latest_ckpt}" ]]; then
    echo "[ERROR] No finetune checkpoint found in ${RUN_DIR}/finetune/logs" >&2
    exit 1
  fi
  cp -f "${latest_ckpt}" "${RUN_DIR}/finetune/exported/finetune_best.pth"
fi

echo "[INFO] Finetune done: ${RUN_DIR}/finetune/exported/finetune_best.pth"
