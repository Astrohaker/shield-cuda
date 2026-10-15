// Launches a tiny kernel many times in a row and measures the cost per launch.
// Compare: launchbench  vs  CUDA_USB_SYNC=1 launchbench  (one USB round trip per call).
// Usage: launchbench [launches] [path/to/launchbench.cubin]
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
    int launches = argc > 1 ? atoi(argv[1]) : 1000;
    const char* cubin = argc > 2 ? argv[2] : "launchbench.cubin";
    int n = 4096;
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUdeviceptr dx;
    CUevent start, stop;

    CK(cuInit(0));
    CK(cuDeviceGet(&dev, 0));
    CK(cuCtxCreate(&ctx, 0, dev));
    CK(cuModuleLoad(&mod, cubin));
    CK(cuModuleGetFunction(&fn, mod, "inc"));
    CK(cuMemAlloc(&dx, n * sizeof(float)));
    CK(cuMemsetD32(dx, 0, n));
    CK(cuEventCreate(&start, 0));
    CK(cuEventCreate(&stop, 0));
    CK(cuCtxSynchronize());

    void* args[] = {&dx, &n};
    double t0 = nowMs();
    CK(cuEventRecord(start, 0));
    for (int i = 0; i < launches; i++) {
        CK(cuLaunchKernel(fn, n / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    }
    CK(cuEventRecord(stop, 0));
    double t1 = nowMs();
    CK(cuEventSynchronize(stop));
    double t2 = nowMs();

    float gpuMs, x[4096];
    CK(cuEventElapsedTime(&gpuMs, start, stop));
    CK(cuMemcpyDtoH(x, dx, sizeof x));
    int bad = 0;
    for (int i = 0; i < n; i++) bad += x[i] != (float) launches;

    printf("%d launches (%s)\n", launches, getenv("CUDA_USB_SYNC") ? "sync: reply per call" : "async: batched");
    printf("  issuing on the PC   %8.2f ms\n", t1 - t0);
    printf("  until GPU finished  %8.2f ms  (%.1f us per launch)\n", t2 - t0, (t2 - t0) * 1e3 / launches);
    printf("  GPU busy (events)   %8.2f ms\n", gpuMs);
    printf("  result: %s (x = %g, expected %d)\n", bad ? "FAIL" : "OK", x[0], launches);

    cuMemFree(dx);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return bad != 0;
}
