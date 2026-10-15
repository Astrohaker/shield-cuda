# cuda-usb

Use the GPU of an NVIDIA SHIELD Tablet (Tegra K1, Kepler sm_32, CUDA 7.5 driver) from a Linux PC
over USB. Host programs written against the CUDA **Driver API** link to a `libcuda.so` shim that
forwards each call through `adb forward` to `cudausb-server`, which runs the call on the tablet's
own `libcuda.so`.

```
PC: your program ──> build/lib/libcuda.so (shim) ──TCP──> adb forward ──USB──>
tablet: cudausb-server (adb shell) ──> /vendor/lib/libcuda.so ──> Tegra K1 GPU
```

## Repository contents

| Path | What |
|---|---|
| `server/`, `client/`, `common/` | cuda-usb: tablet server and the PC `libcuda.so` shim |
| `python/k1torch/` | PyTorch-like framework for training CNNs on the tablet GPU |
| `examples/`, `python/examples/` | C and Python examples |
| `android-app/` | NSTcuda: Android app that runs a CUDA kernel on the tablet itself (managed memory) |
| `server-app/` | CUDA USB Server: the same server as an Android app with START/STOP and live GPU monitoring |
| `toolchain/` | **not in git** - CUDA 7.5 compiler, NDK subset, adb; comes with the release archive |

After `git clone`, get `toolchain/` from the `cuda-usb-offline.tar.gz` release asset (it contains
the whole folder, ready to use offline):

```bash
gh release download --repo Astrohaker/cuda-usb --pattern cuda-usb-offline.tar.gz
tar xzf cuda-usb-offline.tar.gz cuda-usb/toolchain --strip-components=1 -C .   # inside the clone
```

`android-app/` builds with Android Studio or `./gradlew bundleRelease`; it compiles its kernel with
`../toolchain/cuda` (else `~/cuda-7.5`). Release signing reads `NSTCUDA_*` / `DESCENT2_*` from
`~/.gradle/gradle.properties` - no keys or passwords are stored in the repository.

This is a prototype: it is **not** a real GPU for the PC. `nvidia-smi`, the NVIDIA kernel driver,
OpenGL/Vulkan, games, PyTorch/TensorFlow (need sm_35+, CUDA 10+, cuDNN) will not see it.

## Requirements

Everything needed is bundled in `toolchain/`, so an offline PC works:

| Folder | What | Size |
|---|---|---|
| `toolchain/cuda` | CUDA 7.5 compiler only (nvcc, cicc, ptxas, headers) + `hostcc/` wrapper | 133 MB |
| `toolchain/ndk` | Android NDK 21 clang, GNU ld, armv7 API 23 sysroot | 147 MB |
| `toolchain/platform-tools` | adb (used only if no adb is installed) | 12 MB |

The PC needs x86_64 Linux with glibc 2.17+ (any distro since ~2014). `gcc` and `make` are needed
to build host programs; without them the prebuilt `build/` binaries are used (the prebuilt
examples need glibc 2.34+). On the tablet: USB debugging enabled; nothing is installed except
`/data/local/tmp/cudausb-server`.

Installed toolchains can be used instead: `make CUDA_HOME=... NDK=...`, `CUDA_HOME=... nvcc-k1`,
`ADB=... scripts/connect.sh`.

## Offline installation

On a PC with this folder:

```bash
scripts/pack.sh                       # -> ../cuda-usb-offline.tar.gz (~95 MB)
```

Copy the archive (USB stick, etc.) to the other PC:

```bash
tar xzf cuda-usb-offline.tar.gz
cuda-usb/scripts/install.sh           # checks toolchains, builds, self-test
cuda-usb/scripts/install.sh --udev    # also lets adb reach the tablet without root (sudo)
cuda-usb/scripts/install.sh --bashrc  # also adds env.sh to ~/.bashrc
```

The folder is relocatable: all paths are relative to it.

## GUI server on the tablet (server-app)

Instead of the console `cudausb-server`, the server can run inside the **CUDA USB Server** app:
START/STOP buttons, segmented gauges and 60-second charts of GPU load and clock (gk20a sysfs),
USB traffic, CUDA calls and kernel launches per second, CPU and memory, plus server counters and
a log. It compiles `server/server.c` with `-DCUDAUSB_LIBRARY` and keeps running in a foreground
service with a wake lock when the screen is off.

```bash
cd server-app && ./gradlew installDebug   # build + install (uses ../toolchain/cuda for cuda.h)
scripts/disconnect.sh                     # the console server and the app share port 7777
```

Press START in the app, then `scripts/connect.sh` on the PC: it sees the app serving and only
sets up `adb forward`. Temperatures other than the battery's are closed to apps by SELinux.

## Usage

```bash
cd cuda-usb
scripts/connect.sh          # find the tablet, push + start the server, forward port 7777
source scripts/env.sh       # LD_LIBRARY_PATH -> shim, PATH -> nvcc-k1 and examples
devicequery
cd build/examples && vecadd 1048576 && launchbench 1000
scripts/disconnect.sh       # stop the server
```

`connect.sh` needs to run once per USB connection.

## Writing your own program

Kernels are compiled separately as 32-bit sm_32 cubins; host code is ordinary C/C++ using
`cuda.h` (Driver API) and is linked against the shim:

```bash
nvcc-k1 -cubin -o mykernel.cubin mykernel.cu     # extern "C" __global__ ... avoids name mangling
gcc $CUDA_USB_CFLAGS main.c -o main $CUDA_USB_LDFLAGS   # variables come from env.sh
./main
```

See `examples/vecadd.c` + `examples/vecadd.cu`. The same program runs unchanged on a real NVIDIA
GPU if its kernels are built for that GPU instead.

## Python: k1torch (training CNNs)

`python/k1torch` is a small PyTorch-like framework on top of the shim: pure Python (ctypes),
no NumPy, with its own sm_32 kernels (`kernels.cu`).

```python
import k1torch as kt
from k1torch import nn, optim, functional as F
from k1torch.data import DataLoader

model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Flatten(), nn.Linear(8 * 10 * 10, 5))
opt = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
for x, y in DataLoader(x_gpu, y_gpu, batch_size=64, shuffle=True):
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
```

- Tensors: float32/int32 on the GPU, `kt.tensor(list | array | bytes)`, `zeros`, `randn`, views,
  slicing along dim 0, `.tolist()`, `.item()`, `no_grad()`.
- Autograd for `conv2d` (any stride/padding), `linear`, `relu`, `max_pool2d`, `flatten`/`view`,
  `cross_entropy`; layers `Conv2d`, `Linear`, `ReLU`, `MaxPool2d`, `Flatten`, `Sequential`;
  `SGD` with momentum and weight decay; `Module.save/load`.
- `DataLoader` shuffles and batches on the GPU: upload the dataset once, then each step only
  sends kernel launches. Allocations are cached, so steady-state steps never wait for USB.
- Read metrics rarely: `.item()` waits for all queued work (see `total_loss += loss.detach()`
  in the example).

```bash
source scripts/env.sh                                # also sets PYTHONPATH
python3 python/tests/test_grad.py                    # gradient check of every op
python3 python/examples/train_synthetic.py --epochs 5
```

The example (5 classes of noisy 20x20 shapes, 6400 training images, 27k-parameter CNN) reaches
~99% test accuracy in 5 epochs at ~2 s and ~3000 images/s per epoch.

Not PyTorch: no general tensor arithmetic or broadcasting, no automatic differentiation of
arbitrary Python code, no batch norm / dropout / Adam yet, float32 only.

## Asynchronous calls

Like a real GPU, calls that only enqueue work return immediately: `cuLaunchKernel`,
`cuMemsetD8/D32`, `cuMemcpyHtoD/DtoD`, `cuEventRecord`, `cuMemFree`, `cuEventDestroy`. They are
batched in a 64 KB buffer and sent without waiting for a reply. Everything else waits for the
tablet (DtoH copies, synchronize, allocations, module loading, queries) and also flushes the queue.

An error in a queued call is returned by the **next synchronizing call** (`cuCtxSynchronize`,
`cuEventSynchronize`, `cuMemcpyDtoH`, ...), as with real CUDA. `CUDA_USB_SYNC=1` makes every call
wait, so errors point at the failing call (slower, for debugging).

`examples/launchbench` (tiny kernel, 4096 elements):

| | per launch |
|---|---|
| `CUDA_USB_SYNC=1`, 1000 launches | 871 us |
| async, 1000 launches | 108 us |
| async, 10000 launches | 78 us |
| native on the tablet, 10000 launches | 45 us |

So on the PC: keep data on the GPU, queue many launches, synchronize rarely.

## Supported API

`cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount/Get/GetName/GetAttribute/TotalMem/ComputeCapability`,
`cuCtxCreate/Destroy/Synchronize`, `cuMemGetInfo`, `cuMemAlloc/Free`, `cuMemcpyHtoD/DtoH/DtoD`,
`cuMemsetD8/D32`, `cuModuleLoad/LoadData/Unload/GetFunction`, `cuLaunchKernel` (kernelParams only),
`cuEventCreate/Record/Synchronize/ElapsedTime/Destroy`, `cuGetErrorName/String`.

## Limitations

- **Driver API only.** The runtime API (`cudaMalloc`, `kernel<<<>>>`, libcudart), cuBLAS, cuDNN,
  cuFFT are not available.
- **Cubins only, 32-bit, sm_32.** PTX and fatbins are rejected; build kernels with `nvcc-k1`.
  The shim reads parameter sizes from the cubin and narrows 64-bit host pointers to 32-bit.
- **USB is slow:** ~7 MB/s each way, the same as `adb push` on this tablet (adbd on Android 6
  over USB 2.0). Keep data on the GPU and launch many kernels per transfer.
- **Weak GPU:** 192 CUDA cores, ~0.36 TFLOPS fp32, ~130 MB of free memory at a time.
- **One client at a time,** default stream only (calls are still asynchronous, see above),
  no managed memory (the CPU side is on the PC).
- No authentication: the server listens on the tablet's 127.0.0.1 only and is reached through adb.

Debug output: `CUDA_USB_DEBUG=1`. Server log: `adb shell cat /data/local/tmp/cudausb-server.log`.

## Layout

```
common/protocol.h      wire format
server/server.c        tablet side (Android armv7, NDK)
client/libcuda.c       PC side libcuda.so shim
examples/              devicequery, vecadd, launchbench (+ kernels)
python/k1torch/        PyTorch-like framework (kernels.cu -> kernels.cubin)
python/examples/       train_synthetic.py
python/tests/          test_grad.py
scripts/connect.sh     detect tablet, start server, adb forward
scripts/disconnect.sh  stop server
scripts/env.sh         environment for using the shim
scripts/nvcc-k1        kernel compiler wrapper (CUDA 7.5 nvcc-android)
scripts/install.sh     offline setup on a new PC
scripts/pack.sh        make cuda-usb-offline.tar.gz
scripts/adb-path.sh    picks installed adb, else the bundled one
udev/                  udev rule for adb access to NVIDIA devices
toolchain/             bundled CUDA 7.5 compiler, NDK 21 subset, adb
build/                 prebuilt server, shim and examples
```

Bundled third-party software keeps its own license: CUDA (`toolchain/cuda/EULA.txt`), Android NDK
(`toolchain/ndk/NOTICE`), platform-tools (`toolchain/platform-tools/NOTICE.txt`).
