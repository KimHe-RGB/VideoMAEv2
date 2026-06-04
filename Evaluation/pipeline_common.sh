#!/usr/bin/env bash
set -euo pipefail

now_ts() {
  date +%Y%m%d_%H%M%S
}

abs_path() {
  local py_bin="${PYTHON_BIN:-python3}"
  if ! command -v "$py_bin" >/dev/null 2>&1; then
    py_bin="python"
  fi
  "$py_bin" - "$1" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] Required file not found: $p" >&2
    exit 1
  fi
}

require_dir() {
  local p="$1"
  if [[ ! -d "$p" ]]; then
    echo "[ERROR] Required directory not found: $p" >&2
    exit 1
  fi
}

log_cmd() {
  local log_file="$1"
  shift
  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
    printf ' '
    printf '%q ' "$@"
    echo
  } >> "$log_file"
}

latest_checkpoint_in_dir() {
  local out_dir="$1"
  find "$out_dir" -maxdepth 1 -type f -name 'checkpoint-*.pth' -print \
    | awk -F'checkpoint-|\\.pth' '{print $2 "\t" $0}' \
    | awk '$1 ~ /^[0-9]+$/ {print $0}' \
    | sort -n \
    | tail -n 1 \
    | cut -f2-
}

copy_finetune_splits_snapshot() {
  local split_root="$1"
  local dest_dir="$2"
  mkdir -p "$dest_dir"

  require_file "$split_root/train.csv"
  require_file "$split_root/val.csv"
  require_file "$split_root/test.csv"

  cp -f "$split_root/train.csv" "$dest_dir/train.csv"
  cp -f "$split_root/val.csv" "$dest_dir/val.csv"
  cp -f "$split_root/test.csv" "$dest_dir/test.csv"

  {
    echo "# sha256"
    shasum -a 256 "$dest_dir/train.csv"
    shasum -a 256 "$dest_dir/val.csv"
    shasum -a 256 "$dest_dir/test.csv"
  } > "$dest_dir/split_checksums.txt"
}

build_unlabeled_manifest_with_safe_links() {
  local src_dir="$1"
  local safe_dir="$2"
  local manifest_path="$3"

  require_dir "$src_dir"
  rm -rf "$safe_dir"
  mkdir -p "$safe_dir"
  mkdir -p "$(dirname "$manifest_path")"
  : > "$manifest_path"

  local count=0
  while IFS= read -r -d '' src; do
    local base ext stem safe_stem candidate dedup
    base="$(basename "$src")"
    ext="${base##*.}"
    stem="${base%.*}"
    safe_stem="$(echo "$stem" | tr '[:space:]' '_' | tr -cd '[:alnum:]_.-')"
    [[ -n "$safe_stem" ]] || safe_stem="clip_${count}"

    candidate="$safe_dir/${safe_stem}.${ext}"
    dedup=1
    while [[ -e "$candidate" ]]; do
      candidate="$safe_dir/${safe_stem}_${dedup}.${ext}"
      dedup=$((dedup + 1))
    done

    ln -s "$src" "$candidate"
    echo "$candidate 0 -1" >> "$manifest_path"
    count=$((count + 1))
  done < <(find "$src_dir" -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \) -print0)

  if [[ "$count" -eq 0 ]]; then
    echo "[ERROR] No video clips found in $src_dir" >&2
    exit 1
  fi

  echo "$count"
}
