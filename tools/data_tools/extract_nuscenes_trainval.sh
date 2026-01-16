#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/root/autodl-pub/nuScenes"
DST_ROOT="/root/autodl-tmp/data"
NS_DST="${DST_ROOT}/nuscenes"

echo "[INFO] SRC_ROOT = ${SRC_ROOT}"
echo "[INFO] DST_ROOT = ${DST_ROOT}"
echo "[INFO] NS_DST    = ${NS_DST}"

# -------- helpers --------
need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1"; exit 1; }
}

extract_tgz() {
  local tgz="$1"
  echo "[INFO] Extract: ${tgz}"
  tar -xzf "${tgz}" -C "${NS_DST}"
}

# -------- precheck --------
need_cmd tar
need_cmd unzip
need_cmd find

mkdir -p "${DST_ROOT}"
mkdir -p "${NS_DST}"

# -------- 1) extract can_bus.zip --------
echo "[INFO] Searching can_bus.zip under ${SRC_ROOT} ..."
CANBUS_ZIP="$(find "${SRC_ROOT}" -maxdepth 3 -type f -name "can_bus.zip" | head -n 1 || true)"
if [[ -z "${CANBUS_ZIP}" ]]; then
  echo "[WARN] can_bus.zip not found under ${SRC_ROOT} (maxdepth=3). Skip."
else
  echo "[INFO] Found can_bus.zip: ${CANBUS_ZIP}"
  # 通常 zip 内部自带 can_bus/ 目录，因此直接解到 DST_ROOT 即可
  unzip -oq "${CANBUS_ZIP}" -d "${DST_ROOT}"
fi

# -------- 2) extract all trainval tgz --------
# 你截图中路径是: ${SRC_ROOT}/Fulldatasetv1.0/Trainval/*.tgz
TRAINVAL_DIR="${SRC_ROOT}/Fulldatasetv1.0/Trainval"
if [[ ! -d "${TRAINVAL_DIR}" ]]; then
  echo "[WARN] ${TRAINVAL_DIR} not found. Try to locate Trainval dir under ${SRC_ROOT} ..."
  TRAINVAL_DIR="$(find "${SRC_ROOT}" -maxdepth 4 -type d -name "Trainval" | head -n 1 || true)"
fi

if [[ -z "${TRAINVAL_DIR}" || ! -d "${TRAINVAL_DIR}" ]]; then
  echo "[ERROR] Trainval directory not found under ${SRC_ROOT}."
  exit 1
fi

echo "[INFO] Using TRAINVAL_DIR = ${TRAINVAL_DIR}"

# 先解 meta，再解 blobs（顺序更稳妥）
META_TGZ="$(find "${TRAINVAL_DIR}" -maxdepth 1 -type f -name "*meta*.tgz" | sort | head -n 1 || true)"
if [[ -n "${META_TGZ}" ]]; then
  extract_tgz "${META_TGZ}"
else
  echo "[WARN] meta tgz not found (e.g., v1.0-trainval_meta.tgz)."
fi

# blobs 按文件名排序解压
mapfile -t BLOBS_TGZ < <(find "${TRAINVAL_DIR}" -maxdepth 1 -type f -name "*blobs*.tgz" | sort)
if [[ "${#BLOBS_TGZ[@]}" -eq 0 ]]; then
  echo "[WARN] blobs tgz not found under ${TRAINVAL_DIR}."
else
  for f in "${BLOBS_TGZ[@]}"; do
    extract_tgz "${f}"
  done
fi

# -------- 3) post-fix: handle accidental extra nesting --------
# 若某些包解出来多了一层 nuScenes/，则拍平
if [[ -d "${NS_DST}/nuScenes" ]]; then
  echo "[WARN] Found nested directory: ${NS_DST}/nuScenes. Flattening ..."
  shopt -s dotglob
  mv "${NS_DST}/nuScenes/"* "${NS_DST}/"
  rmdir "${NS_DST}/nuScenes" || true
  shopt -u dotglob
fi

# -------- 4) sanity check --------
echo "[INFO] Sanity check directories:"
for d in samples sweeps v1.0-trainval; do
  if [[ -d "${NS_DST}/${d}" ]]; then
    echo "  [OK] ${NS_DST}/${d}"
  else
    echo "  [MISS] ${NS_DST}/${d}"
  fi
done

if [[ -d "${DST_ROOT}/can_bus" ]]; then
  echo "  [OK] ${DST_ROOT}/can_bus"
else
  echo "  [MISS] ${DST_ROOT}/can_bus"
fi

echo "[DONE] Extraction finished."
