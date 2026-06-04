#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${RUN_DIR:?RUN_DIR is required}"

PYTHON_BIN="${PYTHON_BIN:-python}"

PREP_INPUT_CSV="${PREP_INPUT_CSV:-${PROJECT_ROOT}/../预训练数据集_0603_实际路径_found.csv}"
PREP_OUTPUT_DIR="${PREP_OUTPUT_DIR:-${PROJECT_ROOT}/Data/dataset_post_pretrain}"
PREP_METADATA_DIR="${PREP_METADATA_DIR:-${PROJECT_ROOT}/Data/dataset_post_pretrain_metadata}"
PREP_MANIFEST_PATH="${PREP_MANIFEST_PATH:-${PROJECT_ROOT}/Data/dataset_post_pretrain_manifest.csv}"
PREP_MODEL_ID="${PREP_MODEL_ID:-iic/cv_tinynas_head-detection_damoyolo}"
PREP_OUTPUT_SIZE="${PREP_OUTPUT_SIZE:-224}"
PREP_CROP_SCALE="${PREP_CROP_SCALE:-1.15}"
PREP_SCORE_THRESHOLD="${PREP_SCORE_THRESHOLD:-0.35}"
PREP_SMOOTH_ALPHA="${PREP_SMOOTH_ALPHA:-0.80}"
PREP_MAX_VIDEOS="${PREP_MAX_VIDEOS:-0}"
PREP_MAX_FRAMES="${PREP_MAX_FRAMES:-0}"
PREP_DEVICE="${PREP_DEVICE:-cuda}"
PREP_PATH_COLUMN="${PREP_PATH_COLUMN:-实际路径}"
PREP_ID_COLUMN="${PREP_ID_COLUMN:-编号}"

mkdir -p "${RUN_DIR}/metadata"
mkdir -p "${PREP_OUTPUT_DIR}"
mkdir -p "${PREP_METADATA_DIR}"

require_file "${PREP_INPUT_CSV}"

cmd_log="${RUN_DIR}/metadata/commands.log"

cmd=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/Evaluation/preprocess_face_centered_videos.py"
  --input_csv "${PREP_INPUT_CSV}"
  --videos_dir "${PREP_OUTPUT_DIR}"
  --metadata_dir "${PREP_METADATA_DIR}"
  --manifest_path "${PREP_MANIFEST_PATH}"
  --model_id "${PREP_MODEL_ID}"
  --output_size "${PREP_OUTPUT_SIZE}"
  --crop_scale "${PREP_CROP_SCALE}"
  --score_threshold "${PREP_SCORE_THRESHOLD}"
  --smooth_alpha "${PREP_SMOOTH_ALPHA}"
  --max_videos "${PREP_MAX_VIDEOS}"
  --max_frames "${PREP_MAX_FRAMES}"
  --device "${PREP_DEVICE}"
  --path_column "${PREP_PATH_COLUMN}"
  --id_column "${PREP_ID_COLUMN}"
)

if [[ "${PREP_SKIP_EXISTING:-1}" == "1" ]]; then
  cmd+=(--skip_existing)
fi

if [[ "${PREP_DRAW_BOXES:-0}" == "1" ]]; then
  cmd+=(--draw_boxes)
fi

if [[ -n "${PREP_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${PREP_EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

log_cmd "${cmd_log}" "${cmd[@]}"
"${cmd[@]}"

require_file "${PREP_MANIFEST_PATH}"

echo "[INFO] Prepared unlabeled clips dir: ${PREP_OUTPUT_DIR}"
echo "[INFO] Prepared manifest: ${PREP_MANIFEST_PATH}"
