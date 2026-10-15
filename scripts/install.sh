#!/bin/bash
# Offline setup of cuda-usb on this PC: checks the bundled toolchains, builds (or keeps the
# prebuilt binaries if there is no compiler) and optionally sets up udev and ~/.bashrc.
# Usage: scripts/install.sh [--udev] [--bashrc]
#   --udev    install udev/51-cuda-usb-shield.rules with sudo (adb access without root)
#   --bashrc  add "source .../scripts/env.sh" to ~/.bashrc
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TC="$ROOT/toolchain"
CLANG="$TC/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"
want_udev=0 want_bashrc=0
for arg in "$@"; do
    case "$arg" in
        --udev) want_udev=1 ;;
        --bashrc) want_bashrc=1 ;;
        *) echo "unknown option: $arg"; exit 2 ;;
    esac
done
ok() { echo "  [ok]   $*"; }
warn() { echo "  [warn] $*"; }
fail() { echo "  [fail] $*"; exit 1; }

echo "Toolchains (bundled in $TC):"
[ "$(uname -m)" = x86_64 ] || fail "x86_64 Linux is required (found $(uname -m))"
"$TC/cuda/bin/nvcc" --version >/dev/null 2>&1 && ok "CUDA 7.5 nvcc" || fail "nvcc does not run"
"$CLANG" --version >/dev/null 2>&1 && ok "NDK 21 clang" || fail "NDK clang does not run (needs glibc 2.17+, libz)"
"$TC/platform-tools/adb" version >/dev/null 2>&1 && ok "adb (bundled)" || warn "bundled adb does not run"
command -v adb >/dev/null && ok "adb (installed: $(command -v adb), used first)"

echo "Kernel compiler:"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
"$ROOT/scripts/nvcc-k1" -cubin -o "$tmp/t.cubin" "$ROOT/examples/vecadd.cu" >"$tmp/nvcc.log" 2>&1 &&
    ok "nvcc-k1 builds sm_32 cubins" || { cat "$tmp/nvcc.log"; fail "nvcc-k1 failed"; }

echo "Build:"
if command -v make >/dev/null && command -v cc >/dev/null; then
    make -C "$ROOT" -s >"$tmp/make.log" 2>&1 && ok "built server, libcuda.so shim and examples" ||
        { cat "$tmp/make.log"; fail "make failed"; }
else
    warn "no make/cc: using prebuilt binaries from build/ (install gcc to build your own programs)"
    [ -f "$ROOT/build/lib/libcuda.so.1" ] && [ -f "$ROOT/build/android/cudausb-server" ] ||
        fail "prebuilt binaries are missing"
    warn "prebuilt examples need glibc 2.34+; the libcuda.so shim and the server work anywhere"
fi

echo "USB access:"
rules=/etc/udev/rules.d/51-cuda-usb-shield.rules
if [ -f "$rules" ] || ls /etc/udev/rules.d /lib/udev/rules.d 2>/dev/null | grep -qi android; then
    ok "udev rules for adb present"
elif [ $want_udev = 1 ]; then
    sudo install -m 644 "$ROOT/udev/51-cuda-usb-shield.rules" "$rules" &&
        sudo udevadm control --reload-rules && sudo udevadm trigger &&
        ok "installed $rules (replug the tablet)" || warn "could not install udev rule"
else
    warn "no udev rule for adb; if adb shows 'no permissions', rerun with --udev"
fi

if [ $want_bashrc = 1 ]; then
    line="source \"$ROOT/scripts/env.sh\""
    grep -qxF "$line" ~/.bashrc 2>/dev/null || echo "$line" >>~/.bashrc
    ok "added env.sh to ~/.bashrc"
fi

echo
echo "Done. Connect the tablet (USB debugging on), then:"
echo "  $ROOT/scripts/connect.sh"
echo "  source $ROOT/scripts/env.sh && devicequery"
