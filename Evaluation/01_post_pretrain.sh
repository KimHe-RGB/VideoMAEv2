#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${RUN_DIR:?RUN_DIR is required}"

UNLABELED_CLIPS_DIR="${UNLABELED_CLIPS_DIR:-${PROJECT_ROOT}/Data/dataset_post_pretrain}"
PYTHON_BIN="${PYTHON_BIN:-python}"

POST_MODEL="${POST_MODEL:-pretrain_videomae_base_patch16_224}"
POST_MASK_TYPE="${POST_MASK_TYPE:-tube}"
POST_MASK_RATIO="${POST_MASK_RATIO:-0.9}"
POST_DECODER_MASK_TYPE="${POST_DECODER_MASK_TYPE:-run_cell}"
POST_DECODER_MASK_RATIO="${POST_DECODER_MASK_RATIO:-0.5}"
POST_DECODER_DEPTH="${POST_DECODER_DEPTH:-4}"
POST_EPOCHS="${POST_EPOCHS:-50}"
POST_BATCH_SIZE="${POST_BATCH_SIZE:-4}"
POST_NUM_FRAMES="${POST_NUM_FRAMES:-16}"
POST_SAMPLING_RATE="${POST_SAMPLING_RATE:-4}"
POST_NUM_SAMPLE="${POST_NUM_SAMPLE:-2}"
POST_NUM_WORKERS="${POST_NUM_WORKERS:-4}"
POST_INPUT_SIZE="${POST_INPUT_SIZE:-224}"
POST_LR="${POST_LR:-1e-4}"
POST_OPT="${POST_OPT:-adamw}"
POST_OPT_BETAS_A="${POST_OPT_BETAS_A:-0.9}"
POST_OPT_BETAS_B="${POST_OPT_BETAS_B:-0.95}"
POST_WARMUP_EPOCHS="${POST_WARMUP_EPOCHS:-5}"
POST_SAVE_CKPT_FREQ="${POST_SAVE_CKPT_FREQ:-5}"
POST_DEVICE="${POST_DEVICE:-cuda}"
POST_INIT_CKPT="${POST_INIT_CKPT:-}"

mkdir -p "${RUN_DIR}/data"
mkdir -p "${RUN_DIR}/post_pretrain/logs"
mkdir -p "${RUN_DIR}/post_pretrain/exported"
mkdir -p "${RUN_DIR}/metadata"

manifest_path="${RUN_DIR}/data/post_pretrain_manifest.csv"
safe_dir="${RUN_DIR}/data/post_pretrain_videos_nospace"
cmd_log="${RUN_DIR}/metadata/commands.log"

count="$(build_unlabeled_manifest_with_safe_links "${UNLABELED_CLIPS_DIR}" "${safe_dir}" "${manifest_path}")"
echo "[INFO] Post-pretrain clips indexed: ${count}"

cmd=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/run_mae_pretraining.py"
  --data_path "${manifest_path}"
  --data_root ""
  --model "${POST_MODEL}"
  --mask_type "${POST_MASK_TYPE}"
  --mask_ratio "${POST_MASK_RATIO}"
  --decoder_mask_type "${POST_DECODER_MASK_TYPE}"
  --decoder_mask_ratio "${POST_DECODER_MASK_RATIO}"
  --decoder_depth "${POST_DECODER_DEPTH}"
  --batch_size "${POST_BATCH_SIZE}"
  --num_frames "${POST_NUM_FRAMES}"
  --sampling_rate "${POST_SAMPLING_RATE}"
  --num_sample "${POST_NUM_SAMPLE}"
  --num_workers "${POST_NUM_WORKERS}"
  --input_size "${POST_INPUT_SIZE}"
  --opt "${POST_OPT}"
  --lr "${POST_LR}"
  --opt_betas "${POST_OPT_BETAS_A}" "${POST_OPT_BETAS_B}"
  --warmup_epochs "${POST_WARMUP_EPOCHS}"
  --save_ckpt_freq "${POST_SAVE_CKPT_FREQ}"
  --epochs "${POST_EPOCHS}"
  --device "${POST_DEVICE}"
  --output_dir "${RUN_DIR}/post_pretrain/logs"
  --log_dir "${RUN_DIR}/post_pretrain/logs"
)

if [[ -n "${POST_INIT_CKPT}" ]]; then
  if [[ "${POST_INIT_CKPT}" == *.safetensors ]]; then
    echo "[ERROR] POST_INIT_CKPT points to .safetensors, but run_mae_pretraining.py expects torch checkpoint format." >&2
    exit 1
  fi
  require_file "${POST_INIT_CKPT}"
  cmd+=(--finetune "${POST_INIT_CKPT}")
fi

if [[ -n "${POST_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${POST_EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

log_cmd "${cmd_log}" "${cmd[@]}"
"${cmd[@]}"

latest_ckpt="$(latest_checkpoint_in_dir "${RUN_DIR}/post_pretrain/logs")"
if [[ -z "${latest_ckpt}" ]]; then
  echo "[ERROR] No post-pretrain checkpoint found in ${RUN_DIR}/post_pretrain/logs" >&2
  exit 1
fi

cp -f "${latest_ckpt}" "${RUN_DIR}/post_pretrain/exported/post_pretrain_latest.pth"
echo "${latest_ckpt}" > "${RUN_DIR}/post_pretrain/exported/latest_checkpoint_source.txt"

echo "[INFO] Post-pretraining done: ${RUN_DIR}/post_pretrain/exported/post_pretrain_latest.pth"
