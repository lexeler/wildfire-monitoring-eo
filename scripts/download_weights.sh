#!/usr/bin/env bash
# Скачивает чекпойнты сетей модуля BS (≈210 МБ) из GitHub Releases в weights/
# и сверяет контрольные суммы. Остальные веса (LightGBM, GBDT, конфигурации)
# лежат в репозитории.
#   bash scripts/download_weights.sh            # релиз по умолчанию
#   RELEASE=v1.0 bash scripts/download_weights.sh
set -euo pipefail

REPO="${REPO:-lexeler/wildfire-monitoring-eo}"
RELEASE="${RELEASE:-v1.0}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/weights"
BASE="https://github.com/${REPO}/releases/download/${RELEASE}"

CHECKSUMS="95696c15b6162710fa35550f0b63853ad123474969495606f63d2df80a11d813  bs_ctx2_s2.pt
97785cc82e591d91ebf194b718b008149ee1a668f829f4ee82ed3368163f7191  bs_rgy16_s42.pt
54750e1a6387e3804f4a72d10fc23aeae1fab4d9f190e2b32f39a33637bab847  bs_rgy16_s7.pt
9dae68d8652a7d73a2a41c7c437f04e10c8a54de099e182468360142291a619b  bs_rgy32.pt
d8e7600ac07af8868ad4e8ee6172a30711190d8dbbacdea0365cbb678bf4ebbf  bs_unetpp_ctx_s2.pt"

sha256() {
  if command -v shasum >/dev/null; then shasum -a 256 "$1"; else sha256sum "$1"; fi | cut -d' ' -f1
}

mkdir -p "$DEST"
cd "$DEST"
while read -r sum name; do
  if [ -f "$name" ] && [ "$(sha256 "$name")" = "$sum" ]; then
    echo "ok    $name"
    continue
  fi
  echo "fetch $name"
  curl -fL --retry 3 -o "$name.part" "$BASE/$name"
  if [ "$(sha256 "$name.part")" != "$sum" ]; then
    echo "checksum mismatch: $name" >&2
    rm -f "$name.part"
    exit 1
  fi
  mv "$name.part" "$name"
  echo "ok    $name"
done <<< "$CHECKSUMS"
