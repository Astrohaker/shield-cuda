#!/bin/bash
# Packs the whole folder (toolchains + prebuilt binaries) into one archive for offline PCs.
# Usage: scripts/pack.sh [output.tar.gz]   (default: ../cuda-usb-offline.tar.gz)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$(dirname "$ROOT")/cuda-usb-offline.tar.gz}"
make -C "$ROOT" -s
name="$(basename "$ROOT")"
tar -C "$(dirname "$ROOT")" -czf "$OUT" --exclude="$name/.git" --exclude="$name/android-app/.gradle" \
    --exclude="$name/android-app/app/build" --exclude="$name/android-app/app/.cxx" \
    --exclude="$name/android-app/local.properties" --exclude='__pycache__' "$name"
echo "$OUT ($(du -h "$OUT" | cut -f1))"
echo "On the other PC:  tar xzf $(basename "$OUT") && cuda-usb/scripts/install.sh"
