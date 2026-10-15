# Sourced by the other scripts: picks adb. An installed adb wins so two different adb
# servers do not fight over the device; the bundled one is the offline fallback.
CUDA_USB_ROOT="${CUDA_USB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ADB="${ADB:-$(command -v adb || true)}"
[ -n "$ADB" ] || ADB="$CUDA_USB_ROOT/toolchain/platform-tools/adb"
