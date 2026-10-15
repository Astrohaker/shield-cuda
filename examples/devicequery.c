// Prints the properties of the GPU reached through cuda-usb (or any real CUDA driver).
#include <cuda.h>
#include <stdio.h>

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

// Clock attributes are in kHz on desktop drivers but in Hz on the Tegra K1 Android driver.
static int mhz(int clock) {
    return clock > 10000000 ? clock / 1000000 : clock / 1000;
}

static int attr(CUdevice dev, CUdevice_attribute a) {
    int v = 0;
    cuDeviceGetAttribute(&v, a, dev);
    return v;
}

int main(void) {
    int count, version, major, minor;
    char name[256];
    size_t total, freeMem;
    CUdevice dev;
    CUcontext ctx;

    CK(cuInit(0));
    CK(cuDriverGetVersion(&version));
    CK(cuDeviceGetCount(&count));
    printf("CUDA driver %d.%d, %d device(s)\n\n", version / 1000, version % 1000 / 10, count);

    for (int i = 0; i < count; i++) {
        CK(cuDeviceGet(&dev, i));
        CK(cuDeviceGetName(name, sizeof name, dev));
        CK(cuDeviceComputeCapability(&major, &minor, dev));
        CK(cuDeviceTotalMem(&total, dev));
        printf("Device %d: %s\n", i, name);
        printf("  compute capability     sm_%d%d\n", major, minor);
        printf("  multiprocessors        %d\n", attr(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
        printf("  GPU clock              %d MHz\n", mhz(attr(dev, CU_DEVICE_ATTRIBUTE_CLOCK_RATE)));
        printf("  memory clock           %d MHz\n", mhz(attr(dev, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)));
        printf("  memory bus             %d bit\n", attr(dev, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH));
        printf("  total memory           %zu MB\n", total >> 20);
        printf("  max threads per block  %d\n", attr(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK));
        printf("  shared memory / block  %d KB\n", attr(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) >> 10);
        printf("  integrated (shared RAM) %s\n", attr(dev, CU_DEVICE_ATTRIBUTE_INTEGRATED) ? "yes" : "no");
        printf("  managed memory         %s\n", attr(dev, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY) ? "yes" : "no");

        CK(cuCtxCreate(&ctx, 0, dev));
        CK(cuMemGetInfo(&freeMem, &total));
        printf("  free memory            %zu MB\n", freeMem >> 20);
        CK(cuCtxDestroy(ctx));
    }
    return 0;
}
