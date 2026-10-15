// cudausb-server: runs on the SHIELD Tablet (adb shell), executes CUDA Driver API calls
// received from the host shim through the device's own libcuda.so.
// Usage: cudausb-server [-d] [port]   (listens on 127.0.0.1, reached via `adb forward`)
//   -d  detach from the adb shell session, so the server outlives it
// Built with -DCUDAUSB_LIBRARY it has no main() and exposes the API of server.h instead
// (used by the server-app GUI).
#include <arpa/inet.h>
#include <fcntl.h>
#include <cuda.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "../common/protocol.h"
#include "server.h"

#define MAX_TRACKED 256
#define MAX_ALLOCS 4096

static struct {
    CUresult (*init)(unsigned);
    CUresult (*driverGetVersion)(int*);
    CUresult (*deviceGetCount)(int*);
    CUresult (*deviceGet)(CUdevice*, int);
    CUresult (*deviceGetName)(char*, int, CUdevice);
    CUresult (*deviceGetAttribute)(int*, CUdevice_attribute, CUdevice);
    CUresult (*deviceTotalMem)(size_t*, CUdevice);
    CUresult (*ctxCreate)(CUcontext*, unsigned, CUdevice);
    CUresult (*ctxDestroy)(CUcontext);
    CUresult (*ctxSynchronize)(void);
    CUresult (*memGetInfo)(size_t*, size_t*);
    CUresult (*memAlloc)(CUdeviceptr*, size_t);
    CUresult (*memFree)(CUdeviceptr);
    CUresult (*memcpyHtoD)(CUdeviceptr, const void*, size_t);
    CUresult (*memcpyDtoH)(void*, CUdeviceptr, size_t);
    CUresult (*memcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*memsetD8)(CUdeviceptr, unsigned char, size_t);
    CUresult (*memsetD32)(CUdeviceptr, unsigned int, size_t);
    CUresult (*moduleLoadData)(CUmodule*, const void*);
    CUresult (*moduleUnload)(CUmodule);
    CUresult (*moduleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*launchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned, CUstream, void**, void**);
    CUresult (*eventCreate)(CUevent*, unsigned);
    CUresult (*eventRecord)(CUevent, CUstream);
    CUresult (*eventSynchronize)(CUevent);
    CUresult (*eventElapsedTime)(float*, CUevent, CUevent);
    CUresult (*eventDestroy)(CUevent);
} cu;

// Contexts created by the current client, destroyed on disconnect so a crashed
// host program does not leak GPU memory.
static CUcontext contexts[MAX_TRACKED];
static int contextCount;

// The request being handled is fire-and-forget: reply() records errors instead of sending.
static int asyncRequest;
static CUresult deferredError;
static uint32_t currentOp;

// Allocations of the current client, for the "client memory" statistic.
static struct {
    uint32_t ptr, size;
} allocs[MAX_ALLOCS];
static int allocCount;

static ServerStats stats;  // updated with __atomic builtins, read from other threads
#define STAT_ADD(field, v) __atomic_fetch_add(&stats.field, (v), __ATOMIC_RELAXED)
#define STAT_SET(field, v) __atomic_store_n(&stats.field, (v), __ATOMIC_RELAXED)

static volatile int stopping;
static int listenFd = -1;
static volatile int clientFd = -1;

#ifdef CUDAUSB_LIBRARY
static void (*logSink)(const char* line);
void cudausb_set_log_sink(void (*sink)(const char* line)) { logSink = sink; }
#endif

static void logf_(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
#ifdef CUDAUSB_LIBRARY
    char line[512];
    vsnprintf(line, sizeof line, fmt, ap);
    if (logSink) logSink(line);
#else
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
#endif
    va_end(ap);
}

static const char* opName(uint32_t op) {
    static const char* names[] = {
        "?", "Hello", "cuInit", "cuDriverGetVersion", "cuDeviceGetCount", "cuDeviceGet",
        "cuDeviceGetName", "cuDeviceGetAttribute", "cuDeviceTotalMem", "cuCtxCreate",
        "cuCtxDestroy", "cuCtxSynchronize", "cuMemGetInfo", "cuMemAlloc", "cuMemFree",
        "cuMemcpyHtoD", "cuMemcpyDtoH", "cuMemcpyDtoD", "cuMemsetD8", "cuMemsetD32",
        "cuModuleLoadData", "cuModuleUnload", "cuModuleGetFunction", "cuLaunchKernel",
        "cuEventCreate", "cuEventRecord", "cuEventSynchronize", "cuEventElapsedTime",
        "cuEventDestroy",
    };
    return op < sizeof names / sizeof *names ? names[op] : "?";
}

static uint64_t nowNs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000u + ts.tv_nsec;
}

static int loadDriver(void) {
    static int loaded;
    if (loaded) return 1;
    void* lib = dlopen("libcuda.so", RTLD_NOW);
    if (!lib) {
        logf_("dlopen libcuda.so: %s", dlerror());
        return 0;
    }
#define LOAD(field, sym)                                                       \
    if (!(*(void**)&cu.field = dlsym(lib, sym))) {                             \
        logf_("libcuda.so has no %s", sym);                                    \
        return 0;                                                              \
    }
    LOAD(init, "cuInit")
    LOAD(driverGetVersion, "cuDriverGetVersion")
    LOAD(deviceGetCount, "cuDeviceGetCount")
    LOAD(deviceGet, "cuDeviceGet")
    LOAD(deviceGetName, "cuDeviceGetName")
    LOAD(deviceGetAttribute, "cuDeviceGetAttribute")
    LOAD(deviceTotalMem, "cuDeviceTotalMem_v2")
    LOAD(ctxCreate, "cuCtxCreate_v2")
    LOAD(ctxDestroy, "cuCtxDestroy_v2")
    LOAD(ctxSynchronize, "cuCtxSynchronize")
    LOAD(memGetInfo, "cuMemGetInfo_v2")
    LOAD(memAlloc, "cuMemAlloc_v2")
    LOAD(memFree, "cuMemFree_v2")
    LOAD(memcpyHtoD, "cuMemcpyHtoD_v2")
    LOAD(memcpyDtoH, "cuMemcpyDtoH_v2")
    LOAD(memcpyDtoD, "cuMemcpyDtoD_v2")
    LOAD(memsetD8, "cuMemsetD8_v2")
    LOAD(memsetD32, "cuMemsetD32_v2")
    LOAD(moduleLoadData, "cuModuleLoadData")
    LOAD(moduleUnload, "cuModuleUnload")
    LOAD(moduleGetFunction, "cuModuleGetFunction")
    LOAD(launchKernel, "cuLaunchKernel")
    LOAD(eventCreate, "cuEventCreate")
    LOAD(eventRecord, "cuEventRecord")
    LOAD(eventSynchronize, "cuEventSynchronize")
    LOAD(eventElapsedTime, "cuEventElapsedTime")
    LOAD(eventDestroy, "cuEventDestroy_v2")
#undef LOAD
    return loaded = 1;
}

static int readAll(int fd, void* buf, size_t n) {
    char* p = buf;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r <= 0) return 0;
        p += r;
        n -= r;
        STAT_ADD(bytesIn, r);
    }
    return 1;
}

static int writeAll(int fd, const void* buf, size_t n) {
    const char* p = buf;
    while (n) {
        ssize_t r = write(fd, p, n);
        if (r <= 0) return 0;
        p += r;
        n -= r;
        STAT_ADD(bytesOut, r);
    }
    return 1;
}

static int reply(int fd, CUresult result, const void* payload, uint32_t len) {
    if (result != CUDA_SUCCESS) {
        STAT_ADD(errors, 1);
        logf_("%s failed: CUresult %d", opName(currentOp), result);
    }
    if (asyncRequest) {
        if (result != CUDA_SUCCESS && deferredError == CUDA_SUCCESS) deferredError = result;
        return 1;
    }
    ResponseHeader h = {result, deferredError, result == CUDA_SUCCESS ? len : 0};
    deferredError = CUDA_SUCCESS;
    return writeAll(fd, &h, sizeof h) && (!h.len || writeAll(fd, payload, h.len));
}

static int replyU32(int fd, CUresult result, uint32_t value) {
    return reply(fd, result, &value, sizeof value);
}

static void trackContext(CUcontext ctx) {
    if (contextCount < MAX_TRACKED) contexts[contextCount++] = ctx;
    STAT_SET(contexts, contextCount);
}

static void untrackContext(CUcontext ctx) {
    for (int i = 0; i < contextCount; i++) {
        if (contexts[i] == ctx) {
            contexts[i] = contexts[--contextCount];
            break;
        }
    }
    STAT_SET(contexts, contextCount);
}

static void trackAlloc(uint32_t ptr, uint32_t size) {
    STAT_ADD(clientMem, size);
    if (allocCount < MAX_ALLOCS) allocs[allocCount++].ptr = ptr, allocs[allocCount - 1].size = size;
}

static void untrackAlloc(uint32_t ptr) {
    for (int i = 0; i < allocCount; i++) {
        if (allocs[i].ptr == ptr) {
            __atomic_fetch_sub(&stats.clientMem, allocs[i].size, __ATOMIC_RELAXED);
            allocs[i] = allocs[--allocCount];
            return;
        }
    }
}

// Handles one request; returns 0 when the connection should be closed.
static int handle(int fd, uint32_t op, uint8_t* in, uint32_t len) {
    uint32_t* a = (uint32_t*) in;  // most requests are a few u32 arguments
    CUresult r;

#define NEED(bytes)                                                            \
    if (len < (bytes)) return reply(fd, CUDA_ERROR_INVALID_VALUE, NULL, 0);

    switch (op) {
    case OP_HELLO:
        return replyU32(fd, CUDA_SUCCESS, CUDAUSB_MAGIC);
    case OP_INIT:
        NEED(4);
        return reply(fd, cu.init(a[0]), NULL, 0);
    case OP_DRIVER_VERSION: {
        int v = 0;
        r = cu.driverGetVersion(&v);
        return replyU32(fd, r, (uint32_t) v);
    }
    case OP_DEVICE_COUNT: {
        int v = 0;
        r = cu.deviceGetCount(&v);
        return replyU32(fd, r, (uint32_t) v);
    }
    case OP_DEVICE_GET: {
        NEED(4);
        CUdevice d = 0;
        r = cu.deviceGet(&d, (int) a[0]);
        return replyU32(fd, r, (uint32_t) d);
    }
    case OP_DEVICE_NAME: {
        NEED(4);
        char name[256] = {0};
        r = cu.deviceGetName(name, sizeof name, (CUdevice) a[0]);
        return reply(fd, r, name, strlen(name) + 1);
    }
    case OP_DEVICE_ATTRIBUTE: {
        NEED(8);
        int v = 0;
        r = cu.deviceGetAttribute(&v, (CUdevice_attribute) a[0], (CUdevice) a[1]);
        return replyU32(fd, r, (uint32_t) v);
    }
    case OP_DEVICE_TOTAL_MEM: {
        NEED(4);
        size_t v = 0;
        r = cu.deviceTotalMem(&v, (CUdevice) a[0]);
        return replyU32(fd, r, (uint32_t) v);
    }
    case OP_CTX_CREATE: {
        NEED(8);
        CUcontext ctx = NULL;
        r = cu.ctxCreate(&ctx, a[0], (CUdevice) a[1]);
        if (r == CUDA_SUCCESS) trackContext(ctx);
        return replyU32(fd, r, (uint32_t) (uintptr_t) ctx);
    }
    case OP_CTX_DESTROY: {
        NEED(4);
        CUcontext ctx = (CUcontext) (uintptr_t) a[0];
        r = cu.ctxDestroy(ctx);
        if (r == CUDA_SUCCESS) untrackContext(ctx);
        return reply(fd, r, NULL, 0);
    }
    case OP_CTX_SYNCHRONIZE:
        return reply(fd, cu.ctxSynchronize(), NULL, 0);
    case OP_MEM_GET_INFO: {
        size_t f = 0, t = 0;
        r = cu.memGetInfo(&f, &t);
        uint32_t out[2] = {(uint32_t) f, (uint32_t) t};
        return reply(fd, r, out, sizeof out);
    }
    case OP_MEM_ALLOC: {
        NEED(4);
        CUdeviceptr p = 0;
        r = cu.memAlloc(&p, a[0]);
        if (r == CUDA_SUCCESS) trackAlloc((uint32_t) p, a[0]);
        return replyU32(fd, r, (uint32_t) p);
    }
    case OP_MEM_FREE:
        NEED(4);
        r = cu.memFree(a[0]);
        if (r == CUDA_SUCCESS) untrackAlloc(a[0]);
        return reply(fd, r, NULL, 0);
    case OP_MEMCPY_HTOD:
        NEED(4);
        return reply(fd, cu.memcpyHtoD(a[0], in + 4, len - 4), NULL, 0);
    case OP_MEMCPY_DTOH: {
        NEED(8);
        void* buf = malloc(a[1]);
        if (!buf) return reply(fd, CUDA_ERROR_OUT_OF_MEMORY, NULL, 0);
        r = cu.memcpyDtoH(buf, a[0], a[1]);
        int ok = reply(fd, r, buf, a[1]);
        free(buf);
        return ok;
    }
    case OP_MEMCPY_DTOD:
        NEED(12);
        return reply(fd, cu.memcpyDtoD(a[0], a[1], a[2]), NULL, 0);
    case OP_MEMSET_D8:
        NEED(12);
        return reply(fd, cu.memsetD8(a[0], (unsigned char) a[1], a[2]), NULL, 0);
    case OP_MEMSET_D32:
        NEED(12);
        return reply(fd, cu.memsetD32(a[0], a[1], a[2]), NULL, 0);
    case OP_MODULE_LOAD: {
        CUmodule m = NULL;
        r = cu.moduleLoadData(&m, in);
        if (r == CUDA_SUCCESS) STAT_ADD(modules, 1);
        return replyU32(fd, r, (uint32_t) (uintptr_t) m);
    }
    case OP_MODULE_UNLOAD:
        NEED(4);
        r = cu.moduleUnload((CUmodule) (uintptr_t) a[0]);
        if (r == CUDA_SUCCESS) __atomic_fetch_sub(&stats.modules, 1, __ATOMIC_RELAXED);
        return reply(fd, r, NULL, 0);
    case OP_MODULE_GET_FUNCTION: {
        NEED(5);
        in[len - 1] = 0;
        CUfunction f = NULL;
        r = cu.moduleGetFunction(&f, (CUmodule) (uintptr_t) a[0], (const char*) in + 4);
        return replyU32(fd, r, (uint32_t) (uintptr_t) f);
    }
    case OP_LAUNCH_KERNEL: {
        NEED(sizeof(LaunchRequest));
        LaunchRequest* req = (LaunchRequest*) in;
        if (req->nparams > 256 || req->paramBytes > 4096) {
            return reply(fd, CUDA_ERROR_INVALID_VALUE, NULL, 0);
        }
        NEED(sizeof *req + req->nparams * 2 + req->paramBytes);
        uint16_t* offsets = (uint16_t*) (req + 1);
        uint8_t* params = (uint8_t*) (offsets + req->nparams);
        void* args[256];
        for (uint32_t i = 0; i < req->nparams; i++) {
            if (offsets[i] >= req->paramBytes) return reply(fd, CUDA_ERROR_INVALID_VALUE, NULL, 0);
            args[i] = params + offsets[i];
        }
        r = cu.launchKernel((CUfunction) (uintptr_t) req->function, req->grid[0], req->grid[1],
                            req->grid[2], req->block[0], req->block[1], req->block[2],
                            req->sharedMemBytes, NULL, args, NULL);
        STAT_ADD(launches, 1);
        return reply(fd, r, NULL, 0);
    }
    case OP_EVENT_CREATE: {
        NEED(4);
        CUevent e = NULL;
        r = cu.eventCreate(&e, a[0]);
        return replyU32(fd, r, (uint32_t) (uintptr_t) e);
    }
    case OP_EVENT_RECORD:
        NEED(4);
        return reply(fd, cu.eventRecord((CUevent) (uintptr_t) a[0], NULL), NULL, 0);
    case OP_EVENT_SYNCHRONIZE:
        NEED(4);
        return reply(fd, cu.eventSynchronize((CUevent) (uintptr_t) a[0]), NULL, 0);
    case OP_EVENT_ELAPSED: {
        NEED(8);
        float ms = 0;
        r = cu.eventElapsedTime(&ms, (CUevent) (uintptr_t) a[0], (CUevent) (uintptr_t) a[1]);
        return reply(fd, r, &ms, sizeof ms);
    }
    case OP_EVENT_DESTROY:
        NEED(4);
        return reply(fd, cu.eventDestroy((CUevent) (uintptr_t) a[0]), NULL, 0);
    default:
        logf_("unknown op %u", op);
        return reply(fd, CUDA_ERROR_NOT_SUPPORTED, NULL, 0);
    }
#undef NEED
}

static void serveClient(int fd) {
    RequestHeader h;
    uint8_t* buf = NULL;
    size_t cap = 0;

    while (readAll(fd, &h, sizeof h)) {
        if (h.len + 1 > cap) {
            free(buf);
            cap = h.len + 1;
            buf = malloc(cap);
            if (!buf) {
                logf_("out of memory for %u-byte request", h.len);
                break;
            }
        }
        if (!readAll(fd, buf, h.len)) break;
        asyncRequest = (h.op & OP_ASYNC) != 0;
        currentOp = h.op & ~OP_ASYNC;
        uint64_t t0 = nowNs();
        int ok = handle(fd, currentOp, buf, h.len);
        STAT_ADD(busyNs, nowNs() - t0);
        STAT_ADD(requests, 1);
        if (!ok) break;
    }
    free(buf);

    for (int i = 0; i < contextCount; i++) cu.ctxDestroy(contexts[i]);
    if (contextCount) logf_("released %d context(s) left by the client", contextCount);
    contextCount = 0;
    allocCount = 0;
    deferredError = CUDA_SUCCESS;
    STAT_SET(contexts, 0);
    STAT_SET(clientMem, 0);
    STAT_SET(modules, 0);
}

// Binds 127.0.0.1:port; returns 0 or -errno.
static int listenOn(int port) {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) return -errno;
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(srv, (struct sockaddr*) &addr, sizeof addr) || listen(srv, 4)) {
        int err = errno;
        close(srv);
        return -err;
    }
    stopping = 0;
    listenFd = srv;
    STAT_SET(port, port);
    logf_("cudausb-server listening on 127.0.0.1:%d", port);
    return 0;
}

// Serves clients one at a time until stopServer().
static void acceptLoop(void) {
    int one = 1;
    while (!stopping) {
        int fd = accept(listenFd, NULL, NULL);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        clientFd = fd;
        if (stopping) shutdown(fd, SHUT_RDWR);  // stopServer() ran before clientFd was set
        logf_("client connected");
        STAT_ADD(sessions, 1);
        STAT_SET(clientConnected, 1);
        serveClient(fd);
        STAT_SET(clientConnected, 0);
        clientFd = -1;
        close(fd);
        logf_("client disconnected");
    }
    close(listenFd);
    listenFd = -1;
    logf_("cudausb-server stopped");
}

#ifdef CUDAUSB_LIBRARY

// Wakes acceptLoop() (blocked in accept or in the client's read) and makes it return.
static void stopServer(void) {
    stopping = 1;
    if (listenFd >= 0) shutdown(listenFd, SHUT_RDWR);
    int fd = clientFd;
    if (fd >= 0) shutdown(fd, SHUT_RDWR);
}

static pthread_t serverThread;
static int serverRunning;

static void* serverMain(void* unused) {
    acceptLoop();
    __atomic_store_n(&stats.running, 0, __ATOMIC_RELAXED);
    return NULL;
}

int cudausb_device_info(DeviceInfo* info) {
    memset(info, 0, sizeof *info);
    CUdevice dev;
    if (!loadDriver() || cu.init(0) || cu.deviceGet(&dev, 0)) return 0;
    size_t total = 0;
    cu.driverGetVersion(&info->driverVersion);
    cu.deviceGetName(info->name, sizeof info->name, dev);
    cu.deviceTotalMem(&total, dev);
    info->totalMem = total;
    cu.deviceGetAttribute(&info->ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cu.deviceGetAttribute(&info->ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cu.deviceGetAttribute(&info->multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cu.deviceGetAttribute(&info->clockHz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    return 1;
}

int cudausb_start(int port) {
    if (serverRunning) return 0;
    if (!loadDriver()) return -ENOENT;
    int err = listenOn(port);
    if (err) {
        logf_("cannot listen on 127.0.0.1:%d: %s", port, strerror(-err));
        return err;
    }
    memset(&stats, 0, sizeof stats);
    STAT_SET(port, port);
    STAT_SET(running, 1);
    if ((err = pthread_create(&serverThread, NULL, serverMain, NULL))) {
        close(listenFd);
        listenFd = -1;
        STAT_SET(running, 0);
        return -err;
    }
    serverRunning = 1;
    return 0;
}

void cudausb_stop(void) {
    if (!serverRunning) return;
    stopServer();
    pthread_join(serverThread, NULL);
    serverRunning = 0;
}

void cudausb_get_stats(ServerStats* out) {
    uint64_t* dst = (uint64_t*) out;
    uint64_t* src = (uint64_t*) &stats;
    for (size_t i = 0; i < sizeof stats / sizeof(uint64_t); i++) {
        dst[i] = __atomic_load_n(&src[i], __ATOMIC_RELAXED);
    }
}

#endif

#ifndef CUDAUSB_LIBRARY
int main(int argc, char** argv) {
    int detach = argc > 1 && strcmp(argv[1], "-d") == 0;
    int port = argc > 1 + detach ? atoi(argv[1 + detach]) : CUDAUSB_DEFAULT_PORT;
    signal(SIGPIPE, SIG_IGN);

    // adb kills the shell's process group on exit; a new session survives it. The parent
    // waits until the child is listening, otherwise adb may kill it before setsid().
    int ready[2] = {-1, -1};
    if (detach) {
        signal(SIGHUP, SIG_IGN);
        if (pipe(ready)) return 1;
        pid_t pid = fork();
        if (pid < 0) return 1;
        if (pid > 0) {
            char ok = 0;
            close(ready[1]);
            return read(ready[0], &ok, 1) == 1 ? 0 : 1;
        }
        close(ready[0]);
        setsid();
        int null = open("/dev/null", O_RDWR);
        dup2(null, STDIN_FILENO);
        close(null);
    }
    if (!loadDriver()) return 1;  // after fork: libcuda starts threads when loaded
    int err = listenOn(port);
    if (err) {
        fprintf(stderr, "bind/listen 127.0.0.1:%d: %s\n", port, strerror(-err));
        return 1;
    }
    if (detach) {
        write(ready[1], "1", 1);
        close(ready[1]);
    }
    acceptLoop();
    return 0;
}
#endif
