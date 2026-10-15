#!/bin/bash
# Stops cudausb-server on the tablet and removes the port forward.
source "$(dirname "$0")/adb-path.sh"
PORT="${CUDA_USB_PORT:-7777}"
"$ADB" shell "pkill -f /data/local/tmp/cudausb-server" >/dev/null 2>&1
"$ADB" forward --remove "tcp:$PORT" >/dev/null 2>&1
echo "cudausb-server stopped"
