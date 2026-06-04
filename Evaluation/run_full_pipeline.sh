#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_common.sh"

PROJECT_ROOT="$(abs_path "${SCRIPT_DIR}/..")"
CONFIG_PATH="${1:-${SCRIPT_DIR}/pipeline_config.env}"

if [[ -f "${CONFIG_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"
else
  echo "[WARN] Config file not found, using defaults: ${CONFIG_PATH}"
fi

RUNS_ROOT="${RUNS_ROOT:-${SCRIPT_DIR}/test_pipeline_1}"
RUN_NAME="${RUN_NAME:-run_$(now_ts)}"
RUN_DIR="$(abs_path "${RUNS_ROOT}/${RUN_NAME}")"

mkdir -p "${RUN_DIR}"
mkdir -p "${RUN_DIR}/metadata"

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] RUN_DIR=${RUN_DIR}"

{
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "project_root=${PROJECT_ROOT}"
} > "${RUN_DIR}/metadata/run_info.txt"

(
  cd "${PROJECT_ROOT}"
  git rev-parse HEAD > "${RUN_DIR}/metadata/git_head.txt" 2>/dev/null || true
  git status --short > "${RUN_DIR}/metadata/git_status.txt" 2>/dev/null || true
)

{
  echo "# Resolved pipeline env"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "RUNS_ROOT=${RUNS_ROOT}"
  echo "RUN_NAME=${RUN_NAME}"
  echo "RUN_DIR=${RUN_DIR}"
  env | sort | grep -E '^(PYTHON_BIN|TORCHRUN_BIN|NPROC_PER_NODE|MASTER_PORT|PREP_|UNLABELED_CLIPS_DIR|POST_|FINETUNE_|FIN_|EVAL_)=' || true
} > "${RUN_DIR}/metadata/config.resolved.env"

export PROJECT_ROOT
export RUN_DIR

STAGES="${STAGES:-prepare_clips,post_pretrain}"

run_stage() {
  local name="$1"
  echo "[INFO] ===== stage: ${name} ====="
  case "$name" in
    prepare_clips)
      bash "${SCRIPT_DIR}/prepare_clips.sh"
      ;;
    post_pretrain)
      bash "${SCRIPT_DIR}/01_post_pretrain.sh"
      ;;
    finetune)
      bash "${SCRIPT_DIR}/02_finetune.sh"
      ;;
    evaluate)
      bash "${SCRIPT_DIR}/03_evaluate.sh"
      ;;
    *)
      echo "[ERROR] Unknown stage: ${name}" >&2
      exit 1
      ;;
  esac
}

IFS=',' read -r -a stage_list <<< "${STAGES}"
for s in "${stage_list[@]}"; do
  run_stage "${s}"
done

echo "[INFO] Pipeline complete."
echo "[INFO] Run folder: ${RUN_DIR}"
