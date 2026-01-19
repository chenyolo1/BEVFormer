#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/root/autodl-pub/nuScenes"
DST_ROOT="/root/autodl-tmp/data_mini"
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

# -------- 2) extract all mini tgz --------
MINI_DIR="${SRC_ROOT}/Fulldatasetv1.0/Mimi"
if [[ ! -d "${MINI_DIR}" ]]; then
  echo "[WARN] ${MINI_DIR} not found. Try to locate Mini dir under ${SRC_ROOT} ..."
  MINI_DIR="$(find "${SRC_ROOT}" -maxdepth 4 -type d -name "Mini" | head -n 1 || true)"
fi

if [[ -z "${MINI_DIR}" || ! -d "${MINI_DIR}" ]]; then
  echo "[ERROR] Mini directory not found under ${SRC_ROOT}."
  exit 1
fi

echo "[INFO] Using MINI_DIR = ${MINI_DIR}"


MINI_TGZ="$(find "${SRC_ROOT}" -maxdepth 3 -type f -name "v1.0-mini.tgz" | head -n 1 || true)"
extract_tgz "${MINI_TGZ}"

# -------- 4) sanity check --------
echo "[INFO] Sanity check directories:"
for d in samples sweeps v1.0-mini; do
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


# 建立数据软链接
# 1) 进入 BEVFormer 根目录
cd /root/BEVFormer

# 2) 如果已经有 data 目录/链接，先移走（避免覆盖失败）
#    注意：rm -rf data 会删除“链接本身”，但如果你写成 rm -rf data/ 可能会跟随链接删真实数据，别加尾部斜杠
rm -rf data

# 3) 建立软链接：BEVFormer/data -> /root/autodl-tmp/data
ln -s /root/autodl-tmp/data_mini data

# 4) 检查
ls -l data
ls data/nuscenes | head
ls data/can_bus   | head


# 建立ckpts软链接
cd /root/BEVFormer
rm -rf ckpts
ln -s /root/autodl-tmp/ckpts ckpts
ls -l ckpts