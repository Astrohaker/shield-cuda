"""CUDA Driver API through ctypes: context, caching allocator, kernel launches.

Loads libcuda.so - the cuda-usb shim (tablet over USB) or a real driver - and the kernels
from kernels.cubin next to this file.
"""
import ctypes
import os
import struct

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUBIN = os.path.join(_HERE, "kernels.cubin")


def _load_library():
    candidates = []
    root = os.environ.get("CUDA_USB_ROOT")
    if root:
        candidates.append(os.path.join(root, "build", "lib", "libcuda.so.1"))
    candidates.append(os.path.join(_HERE, "..", "..", "build", "lib", "libcuda.so.1"))
    candidates.append("libcuda.so.1")
    errors = []
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            errors.append(f"{path}: {e}")
    raise OSError("libcuda.so not found:\n  " + "\n  ".join(errors))


class CudaError(RuntimeError):
    pass


class Device:
    """Owns the context, the kernel module and the memory cache. One per process."""

    def __init__(self):
        self.lib = _load_library()
        self._check(self.lib.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        name = ctypes.create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(name, 256, dev), "cuDeviceGetName")
        self.name = name.value.decode()
        self.ctx = ctypes.c_void_p()
        self._check(self.lib.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev), "cuCtxCreate")

        with open(_CUBIN, "rb") as f:
            self._image = f.read()
        self.module = ctypes.c_void_p()
        self._check(self.lib.cuModuleLoadData(ctypes.byref(self.module), self._image),
                    "cuModuleLoadData")
        self._functions = {}
        self._cache = {}  # rounded size -> [device pointers]
        self.allocated = 0

    def _check(self, result, what):
        if result != 0:
            name = ctypes.c_char_p()
            self.lib.cuGetErrorName(result, ctypes.byref(name))
            raise CudaError(f"{what} failed: {name.value.decode() if name.value else result}")

    # -- memory -------------------------------------------------------------------------------
    # Allocation is a synchronous USB round trip, so freed blocks are cached and reused
    # (like PyTorch's caching allocator). Sizes are rounded to keep the cache hit rate high.

    @staticmethod
    def _round(nbytes):
        if nbytes <= 4096:
            return max(256, (nbytes + 255) & ~255)
        return (nbytes + 65535) & ~65535

    def malloc(self, nbytes):
        size = self._round(nbytes)
        free = self._cache.get(size)
        if free:
            return free.pop(), size
        ptr = ctypes.c_uint64()
        result = self.lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(size))
        if result == 2:  # CUDA_ERROR_OUT_OF_MEMORY: drop the cache and retry once
            self.empty_cache()
            result = self.lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(size))
        self._check(result, f"cuMemAlloc({size})")
        self.allocated += size
        return ptr.value, size

    def free(self, ptr, size):
        self._cache.setdefault(size, []).append(ptr)

    def empty_cache(self):
        for size, ptrs in self._cache.items():
            for p in ptrs:
                self.lib.cuMemFree_v2(ctypes.c_uint64(p))
                self.allocated -= size
        self._cache.clear()

    def upload(self, ptr, data: bytes):
        self._check(self.lib.cuMemcpyHtoD_v2(ctypes.c_uint64(ptr), data, ctypes.c_size_t(len(data))),
                    "cuMemcpyHtoD")

    def download(self, ptr, nbytes) -> bytes:
        buf = ctypes.create_string_buffer(nbytes)
        self._check(self.lib.cuMemcpyDtoH_v2(buf, ctypes.c_uint64(ptr), ctypes.c_size_t(nbytes)),
                    "cuMemcpyDtoH")
        return buf.raw

    def memset32(self, ptr, value_bits, count):
        self._check(self.lib.cuMemsetD32_v2(ctypes.c_uint64(ptr), ctypes.c_uint(value_bits),
                                            ctypes.c_size_t(count)), "cuMemsetD32")

    def synchronize(self):
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    # -- kernels ------------------------------------------------------------------------------

    def function(self, name):
        f = self._functions.get(name)
        if f is None:
            f = ctypes.c_void_p()
            self._check(self.lib.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()),
                        f"cuModuleGetFunction({name})")
            self._functions[name] = f
        return f

    def launch(self, name, grid, block, *args, shared=0):
        """Queues a kernel. args: device pointers as int via Ptr, Python int -> int32,
        float -> float32."""
        holders = []
        for a in args:
            if isinstance(a, Ptr):
                holders.append(ctypes.c_uint64(a.value))
            elif isinstance(a, bool) or isinstance(a, int):
                holders.append(ctypes.c_int32(int(a)))
            elif isinstance(a, float):
                holders.append(ctypes.c_float(a))
            else:
                raise TypeError(f"unsupported kernel argument {a!r}")
        params = (ctypes.c_void_p * len(holders))(*[ctypes.addressof(h) for h in holders])
        gx, gy, gz = (grid + (1, 1))[:3] if isinstance(grid, tuple) else (grid, 1, 1)
        bx, by, bz = (block + (1, 1))[:3] if isinstance(block, tuple) else (block, 1, 1)
        self._check(self.lib.cuLaunchKernel(self.function(name), gx, gy, gz, bx, by, bz, shared,
                                            None, params, None), f"launch {name}")

    def launch1d(self, name, n, *args):
        """Grid-stride kernel over n elements."""
        block = 256
        grid = max(1, min((n + block - 1) // block, 1024))
        self.launch(name, grid, block, *args)


class Ptr:
    """Marks a device address in a kernel argument list (0 = NULL)."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


_device = None


def device():
    global _device
    if _device is None:
        _device = Device()
    return _device


def float_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]
