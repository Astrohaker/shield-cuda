# source this: puts the cuda-usb libcuda.so first on the library path and nvcc-k1 on PATH.
CUDA_USB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export CUDA_USB_ROOT
export LD_LIBRARY_PATH="$CUDA_USB_ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CUDA_USB_ROOT/scripts:$CUDA_USB_ROOT/build/examples:$PATH:$CUDA_USB_ROOT/toolchain/platform-tools"
# For your own programs: gcc $CUDA_USB_CFLAGS main.c -o main $CUDA_USB_LDFLAGS
export CUDA_USB_CFLAGS="-I$CUDA_USB_ROOT/toolchain/cuda/include"
export CUDA_USB_LDFLAGS="-L$CUDA_USB_ROOT/build/lib -lcuda"
export PYTHONPATH="$CUDA_USB_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"  # import k1torch
export CUDA_USB_ADDR="${CUDA_USB_ADDR:-127.0.0.1:${CUDA_USB_PORT:-7777}}"
