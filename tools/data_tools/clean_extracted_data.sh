#!/usr/bin/env bash
set -euo pipefail

DST_ROOT="/root/autodl-tmp/data"
NS_DST="${DST_ROOT}/nuscenes"
CANBUS_DST="${DST_ROOT}/can_bus"

usage() {
  cat <<EOF
Usage:
  $0            # delete only extracted dirs: nuscenes/ and can_bus/
  $0 --all      # delete EVERYTHING under ${DST_ROOT}
  $0 --dry-run  # show what would be deleted
  $0 -h|--help  # show this help

Examples:
  $0
  $0 --dry-run
  $0 --all
EOF
}

DRY_RUN=0
DELETE_ALL=0

# ✅ 修复点：用 "$@"，无参数时不会进入循环
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --all)     DELETE_ALL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: '$arg'"; usage; exit 1 ;;
  esac
done

echo "[INFO] DST_ROOT = ${DST_ROOT}"
if [[ ! -d "${DST_ROOT}" ]]; then
  echo "[INFO] ${DST_ROOT} does not exist. Nothing to delete."
  exit 0
fi

# 额外保护：防止 DST_ROOT 为空或意外为 /
if [[ -z "${DST_ROOT}" || "${DST_ROOT}" == "/" ]]; then
  echo "[ERROR] Refuse to delete because DST_ROOT is empty or '/'."
  exit 1
fi

do_rm_rf() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "[INFO] Not found, skip: $p"
    return 0
  fi

  # 只允许删除 /root/autodl-tmp/data 下的路径
  case "$p" in
    "${DST_ROOT}"|${DST_ROOT}/*) ;;
    *) echo "[ERROR] Refuse to delete outside ${DST_ROOT}: $p"; exit 1 ;;
  esac

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY-RUN] rm -rf '$p'"
  else
    echo "[INFO] rm -rf '$p'"
    rm -rf "$p"
  fi
}

if [[ "${DELETE_ALL}" -eq 1 ]]; then
  echo "[WARN] --all specified: will delete EVERYTHING under ${DST_ROOT} (but keep the root dir itself)"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY-RUN] Would delete:"
    find "${DST_ROOT}" -mindepth 1 -maxdepth 1 -print
  else
    # ✅ 更彻底：连隐藏文件一起删（通过逐项删除子项实现）
    find "${DST_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
else
  echo "[INFO] Will delete extracted dirs only:"
  echo "  - ${NS_DST}"
  echo "  - ${CANBUS_DST}"
  do_rm_rf "${NS_DST}"
  do_rm_rf "${CANBUS_DST}"
fi

echo "[DONE] Cleanup finished."
