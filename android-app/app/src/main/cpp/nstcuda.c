// JNI bridge: runs sqrt_kernel through the CUDA Driver API from the device's libcuda.so
// and times it against a plain CPU loop. Tegra K1 shares DRAM between CPU and GPU, so buffers
// are managed (unified) memory: CPU and GPU use the same pointers, with no explicit copies.
#include <android/asset_manager_jni.h>
#include <cuda.h>
#include <dlfcn.h>
#include <jni.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef CUresult (*cuInit_t)(unsigned);
typedef CUresult (*cuDriverGetVersion_t)(int*);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuDeviceGetName_t)(char*, int, CUdevice);
typedef CUresult (*cuDeviceGetAttribute_t)(int*, CUdevice_attribute, CUdevice);
typedef CUresult (*cuCtxCreate_t)(CUcontext*, unsigned, CUdevice);
typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuMemAllocManaged_t)(CUdeviceptr*, size_t, unsigned);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                                     unsigned, unsigned, CUstream, void**, void**);
typedef CUresult (*cuCtxSynchronize_t)(void);

static struct {
    cuInit_t init;
    cuDriverGetVersion_t driverGetVersion;
    cuDeviceGet_t deviceGet;
    cuDeviceGetName_t deviceGetName;
    cuDeviceGetAttribute_t deviceGetAttribute;
    cuCtxCreate_t ctxCreate;
    cuModuleLoadData_t moduleLoadData;
    cuModuleGetFunction_t moduleGetFunction;
    cuMemAllocManaged_t memAllocManaged;
    cuMemFree_t memFree;
    cuLaunchKernel_t launchKernel;
    cuCtxSynchronize_t ctxSynchronize;
} cu;

static CUfunction sqrtKernel;
static char deviceName[128];
static int managedSupported;
static int driverVersion;
static char initError[256];
static int initialized;

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult r_ = (call);                                                  \
        if (r_ != CUDA_SUCCESS) {                                              \
            snprintf(err, errSize, "%s failed: CUresult %d", #call, r_);       \
            goto fail;                                                         \
        }                                                                      \
    } while (0)

static double nowMs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int loadDriver(void* lib, char* err, size_t errSize) {
#define LOAD(field, sym)                                                       \
    if (!(*(void**)&cu.field = dlsym(lib, sym))) {                             \
        snprintf(err, errSize, "libcuda.so has no %s", sym);                   \
        return 0;                                                              \
    }
    LOAD(init, "cuInit")
    LOAD(driverGetVersion, "cuDriverGetVersion")
    LOAD(deviceGet, "cuDeviceGet")
    LOAD(deviceGetName, "cuDeviceGetName")
    LOAD(deviceGetAttribute, "cuDeviceGetAttribute")
    LOAD(ctxCreate, "cuCtxCreate_v2")
    LOAD(moduleLoadData, "cuModuleLoadData")
    LOAD(moduleGetFunction, "cuModuleGetFunction")
    LOAD(memAllocManaged, "cuMemAllocManaged")
    LOAD(memFree, "cuMemFree_v2")
    LOAD(launchKernel, "cuLaunchKernel")
    LOAD(ctxSynchronize, "cuCtxSynchronize")
#undef LOAD
    return 1;
}

// One-time setup: driver, context and module. Returns 0 and fills initError on failure.
static int initCuda(JNIEnv* env, jobject assetManager) {
    char* err = initError;
    size_t errSize = sizeof initError;
    void* image = NULL;
    AAsset* asset = NULL;
    CUdevice dev;
    CUcontext ctx;
    CUmodule module;

    void* lib = dlopen("libcuda.so", RTLD_NOW);
    if (!lib) {
        snprintf(err, errSize, "CUDA driver not found: %s", dlerror());
        return 0;
    }
    if (!loadDriver(lib, err, errSize)) return 0;

    CHECK(cu.init(0));
    CHECK(cu.driverGetVersion(&driverVersion));
    CHECK(cu.deviceGet(&dev, 0));
    CHECK(cu.deviceGetName(deviceName, sizeof deviceName, dev));
    CHECK(cu.deviceGetAttribute(&managedSupported, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, dev));
    CHECK(cu.ctxCreate(&ctx, 0, dev));

    asset = AAssetManager_open(AAssetManager_fromJava(env, assetManager), "sqrt.cubin",
                               AASSET_MODE_BUFFER);
    if (!asset) {
        snprintf(err, errSize, "asset sqrt.cubin not found");
        goto fail;
    }
    size_t size = AAsset_getLength(asset);
    image = malloc(size);
    AAsset_read(asset, image, size);
    CHECK(cu.moduleLoadData(&module, image));
    CHECK(cu.moduleGetFunction(&sqrtKernel, module, "sqrt_kernel"));
    free(image);
    AAsset_close(asset);
    return 1;

fail:
    free(image);
    if (asset) AAsset_close(asset);
    return 0;
}

static CUresult launchSqrt(CUdeviceptr in, CUdeviceptr out, int n) {
    void* args[] = {&in, &out, &n};
    CUresult r = cu.launchKernel(sqrtKernel, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL);
    return r == CUDA_SUCCESS ? cu.ctxSynchronize() : r;
}

static float maxDiff(const float* a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

JNIEXPORT jstring JNICALL
Java_com_astrohacker_nstcuda_MainActivity_runSqrt(JNIEnv* env, jobject thiz, jobject assetManager,
                                                  jint n) {
    char out[1024];
    char err[256];
    size_t errSize = sizeof err;
    size_t bytes = (size_t) n * sizeof(float);
    float* cpu = NULL;
    CUdeviceptr dIn = 0, dOut = 0;

    if (!initialized) initialized = initCuda(env, assetManager) ? 1 : -1;
    if (initialized < 0) return (*env)->NewStringUTF(env, initError);
    if (!managedSupported) return (*env)->NewStringUTF(env, "GPU не поддерживает managed memory");

    cpu = malloc(bytes);
    if (!cpu) {
        snprintf(err, errSize, "out of memory for %d elements", n);
        goto fail;
    }
    CHECK(cu.memAllocManaged(&dIn, bytes, CU_MEM_ATTACH_GLOBAL));
    CHECK(cu.memAllocManaged(&dOut, bytes, CU_MEM_ATTACH_GLOBAL));
    // 32-bit process: the managed device address is also the CPU address.
    float* in = (float*) (uintptr_t) dIn;
    float* gpu = (float*) (uintptr_t) dOut;
    for (int i = 0; i < n; i++) in[i] = (float) i;

    double t0 = nowMs();
    for (int i = 0; i < n; i++) cpu[i] = sqrtf(in[i]);
    double cpuMs = nowMs() - t0;

    // The CPU must not touch managed buffers while a kernel runs (Kepler has no concurrent
    // access), so launchSqrt synchronizes before the results are read.
    double t1 = nowMs();
    CHECK(launchSqrt(dIn, dOut, n));
    double t2 = nowMs();
    float maxErr = maxDiff(gpu, cpu, n);
    double t3 = nowMs();

    snprintf(out, sizeof out,
             "%s\n"
             "Драйвер CUDA %d.%d, тулкит CUDA %d.%d\n"
             "Элементов: %d\n\n"
             "GPU (CUDA, managed memory)\n"
             "  ядро sqrt + синхр. кешей: %.2f мс\n\n"
             "CPU: %.2f мс\n\n"
             "Проверка результата на CPU: %.2f мс\n"
             "Макс. расхождение GPU/CPU: %g\n\n"
             "sqrt(%d) = %.4f",
             deviceName, driverVersion / 1000, driverVersion % 1000 / 10,
             CUDA_VERSION / 1000, CUDA_VERSION % 1000 / 10, n, t2 - t1, cpuMs, t3 - t2, maxErr, n - 1, gpu[n - 1]);
    cu.memFree(dIn);
    cu.memFree(dOut);
    free(cpu);
    return (*env)->NewStringUTF(env, out);

fail:
    if (dIn) cu.memFree(dIn);
    if (dOut) cu.memFree(dOut);
    free(cpu);
    return (*env)->NewStringUTF(env, err);
}
