#!/bin/bash
# Finds the SHIELD Tablet over adb, (re)starts cudausb-server on it and forwards the port.
# After this, programs linked against build/lib/libcuda.so talk to the tablet GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/adb-path.sh"
PORT="${CUDA_USB_PORT:-7777}"
SERVER="$ROOT/build/android/cudausb-server"
REMOTE=/data/local/tmp/cudausb-server
LOG=/data/local/tmp/cudausb-server.log

[ -x "$ADB" ] || { echo "adb not found: $ADB"; exit 1; }
[ -f "$SERVER" ] || { echo "server not built: run make in $ROOT"; exit 1; }

serial="${ANDROID_SERIAL:-}"
if [ -z "$serial" ]; then
    for s in $("$ADB" devices | awk 'NR > 1 && $2 == "device" { print $1 }'); do
        if [ "$("$ADB" -s "$s" shell getprop ro.board.platform | tr -d '\r')" = tegra ] &&
           "$ADB" -s "$s" shell ls /vendor/lib/libcuda.so >/dev/null 2>&1; then
            serial=$s
            break
        fi
    done
fi
[ -n "$serial" ] || { echo "no Tegra device with libcuda.so found over adb (USB debugging on?)"; exit 1; }
adb() { "$ADB" -s "$serial" "$@"; }
echo "device: $(adb shell getprop ro.product.model | tr -d '\r') ($serial)"

adb push "$SERVER" "$REMOTE" >/dev/null 2>&1 || { echo "cannot push the server to the tablet"; exit 1; }
adb shell chmod 755 "$REMOTE"
adb shell "pkill -f $REMOTE" >/dev/null 2>&1 || true
adb shell "$REMOTE -d $PORT > $LOG 2>&1"
adb forward "tcp:$PORT" "tcp:$PORT" >/dev/null

for _ in $(seq 1 20); do
    if adb shell cat "$LOG" | grep -q listening; then
        echo "cudausb-server is running, forwarded to 127.0.0.1:$PORT"
        echo
        echo "Use it with:  source $ROOT/scripts/env.sh"
        exit 0
    fi
    sleep 0.2
done
echo "server did not start, log:"
adb shell cat "$LOG"
exit 1
