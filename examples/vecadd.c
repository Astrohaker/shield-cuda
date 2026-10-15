// Adds two vectors on the tablet GPU and reports where the time goes:
// USB transfers vs. kernel time measured on the GPU with events.
// Usage: vecadd [elements] [path/to/vecadd.cubin]
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CK(call)                                                               \
    do {                                                                       \
        CUresult r_ = (call);                                                  \
        if (r_ != CUDA_SUCCESS) {                                              \
            const char* s_;                                                    \
            cuGetErrorName(r_, &s_);                                           \
            fprintf(stderr, "%s failed: %s (%d)\n", #call, s_ ? s_ : "?", r_); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static double nowMs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1 << 20;
    const char* cubin = argc > 2 ? argv[2] : "vecadd.cubin";
    size_t bytes = (size_t) n * sizeof(float);
    char name[256];
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUdeviceptr da, db, dc;
    CUevent start, stop;

    CK(cuInit(0));
    CK(cuDeviceGet(&dev, 0));
    CK(cuDeviceGetName(name, sizeof name, dev));
    CK(cuCtxCreate(&ctx, 0, dev));
    CK(cuModuleLoad(&mod, cubin));
    CK(cuModuleGetFunction(&fn, mod, "vecadd"));
    printf("%s, %d elements (%.1f MB per vector)\n", name, n, bytes / 1048576.0);

    float* a = malloc(bytes);
    float* b = malloc(bytes);
    float* c = malloc(bytes);
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = 2.0f * i;
    }

    CK(cuMemAlloc(&da, bytes));
    CK(cuMemAlloc(&db, bytes));
    CK(cuMemAlloc(&dc, bytes));
    CK(cuEventCreate(&start, 0));
    CK(cuEventCreate(&stop, 0));

    double t0 = nowMs();
    CK(cuMemcpyHtoD(da, a, bytes));
    CK(cuMemcpyHtoD(db, b, bytes));
    CK(cuCtxSynchronize());  // copies are queued asynchronously; wait so the timing is honest
    double t1 = nowMs();
    void* args[] = {&da, &db, &dc, &n};
    CK(cuEventRecord(start, 0));
    CK(cuLaunchKernel(fn, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuEventRecord(stop, 0));
    CK(cuEventSynchronize(stop));
    double t2 = nowMs();
    CK(cuMemcpyDtoH(c, dc, bytes));
    double t3 = nowMs();

    float gpuMs;
    CK(cuEventElapsedTime(&gpuMs, start, stop));
    int bad = 0;
    for (int i = 0; i < n; i++) bad += c[i] != 3.0f * i;

    printf("  host -> GPU (USB)   %8.2f ms  %6.1f MB/s\n", t1 - t0, 2 * bytes / 1048576.0 / ((t1 - t0) / 1e3));
    printf("  kernel (GPU timer)  %8.2f ms\n", gpuMs);
    printf("  kernel + round trip %8.2f ms\n", t2 - t1);
    printf("  GPU -> host (USB)   %8.2f ms  %6.1f MB/s\n", t3 - t2, bytes / 1048576.0 / ((t3 - t2) / 1e3));
    printf("  result: %s (%d/%d correct)\n", bad ? "FAIL" : "OK", n - bad, n);

    cuMemFree(da);
    cuMemFree(db);
    cuMemFree(dc);
    cuEventDestroy(start);
    cuEventDestroy(stop);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(a);
    free(b);
    free(c);
    return bad != 0;
}
