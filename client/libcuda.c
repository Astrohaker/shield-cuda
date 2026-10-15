// libcuda.so shim for the host PC: implements a subset of the CUDA Driver API by forwarding
// every call over TCP (adb forward -> USB) to cudausb-server on the SHIELD Tablet.
//
// The tablet GPU runs 32-bit code: kernels must be sm_32 cubins built with -m32 (nvcc-k1).
// cuLaunchKernel needs each parameter's size, so it is read from the cubin's .nv.info section
// when a function is looked up; 64-bit host values are truncated to the 32-bit kernel layout
// (little-endian, so a CUdeviceptr's low half is the device address).
//
// Calls that the real driver may run asynchronously (launches, memsets, HtoD/DtoD copies, event
// records, frees) are queued and sent in batches without waiting for a reply. Their errors come
// back from the next synchronous call (DtoH copy, synchronize, ...), as with real CUDA.
//
// Configuration: CUDA_USB_ADDR=host:port (default 127.0.0.1:7777), CUDA_USB_DEBUG=1,
// CUDA_USB_SYNC=1 (wait for every call: slower, but errors point at the failing call).
#include <arpa/inet.h>
#include <cuda.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../common/protocol.h"

#define EXPORT __attribute__((visibility("default")))

// Host-side view of a remote module: keeps the cubin to look up kernel parameter layouts.
typedef struct {
    uint32_t remote;
    uint8_t* image;
    size_t size;
} Module;

typedef struct {
    uint32_t remote;
    uint32_t nparams;
    uint32_t paramBytes;
    uint16_t offset[256];
    uint16_t size[256];
} Function;

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int sock = -1;
static int debug;
static int syncMode;

// Queued async requests, flushed when full and before every synchronous request.
static uint8_t queue[64 * 1024];
static size_t queued;

static void dbg(const char* fmt, ...) {
    if (!debug) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[cuda-usb] ");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static int readAll(int fd, void* buf, size_t n) {
    char* p = buf;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r <= 0) return 0;
        p += r;
        n -= r;
    }
    return 1;
}

static int writeAll(int fd, const void* buf, size_t n) {
    const char* p = buf;
    while (n) {
        ssize_t r = send(fd, p, n, MSG_NOSIGNAL);
        if (r <= 0) return 0;
        p += r;
        n -= r;
    }
    return 1;
}

static int connectServer(void) {
    const char* addrEnv = getenv("CUDA_USB_ADDR");
    char host[128] = "127.0.0.1";
    int port = CUDAUSB_DEFAULT_PORT;
    debug = getenv("CUDA_USB_DEBUG") != NULL;
    syncMode = getenv("CUDA_USB_SYNC") != NULL;
    if (addrEnv) {
        const char* colon = strrchr(addrEnv, ':');
        if (colon) {
            snprintf(host, sizeof host, "%.*s", (int) (colon - addrEnv), addrEnv);
            port = atoi(colon + 1);
        } else {
            snprintf(host, sizeof host, "%s", addrEnv);
        }
    }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1 ||
        connect(fd, (struct sockaddr*) &addr, sizeof addr)) {
        fprintf(stderr, "[cuda-usb] cannot reach the tablet at %s:%d - run scripts/connect.sh\n",
                host, port);
        close(fd);
        return -1;
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    dbg("connected to %s:%d", host, port);
    return fd;
}

static int flushQueue(void) {
    int ok = !queued || writeAll(sock, queue, queued);
    queued = 0;
    return ok;
}

// Appends to the queue; large payloads (memcpy data, cubins) go straight to the socket.
static int enqueue(const void* data, size_t n) {
    if (!n) return 1;
    if (queued + n > sizeof queue && !flushQueue()) return 0;
    if (n >= sizeof queue) return writeAll(sock, data, n);
    memcpy(queue + queued, data, n);
    queued += n;
    return 1;
}

static CUresult connectionLost(void) {
    fprintf(stderr, "[cuda-usb] connection to the tablet lost\n");
    close(sock);
    sock = -1;
    queued = 0;
    return CUDA_ERROR_LAUNCH_FAILED;
}

// Queues a request without waiting for its result. The payload is two parts so bulk data
// (memcpy, cubins) needs no extra copy.
static CUresult post(uint32_t op, const void* a, uint32_t alen, const void* b, uint32_t blen);

// One synchronous round trip; also flushes queued async requests. The response payload is
// written to out (up to outCap bytes). Errors of earlier async requests take precedence.
static CUresult rpc2(uint32_t op, const void* a, uint32_t alen, const void* b, uint32_t blen,
                     void* out, uint32_t outCap) {
    CUresult result;
    pthread_mutex_lock(&lock);
    if (sock < 0) {
        result = CUDA_ERROR_NOT_INITIALIZED;
        goto done;
    }
    RequestHeader req = {op, alen + blen};
    ResponseHeader resp;
    if (!enqueue(&req, sizeof req) || !enqueue(a, alen) || !enqueue(b, blen) || !flushQueue() ||
        !readAll(sock, &resp, sizeof resp)) {
        result = connectionLost();
        goto done;
    }
    if (resp.len) {
        uint32_t keep = resp.len < outCap ? resp.len : outCap;
        if (keep && !readAll(sock, out, keep)) {
            result = connectionLost();
            goto done;
        }
        for (uint32_t rest = resp.len - keep; rest;) {  // drain anything unexpected
            char junk[256];
            uint32_t n = rest < sizeof junk ? rest : sizeof junk;
            if (!readAll(sock, junk, n)) {
                result = connectionLost();
                goto done;
            }
            rest -= n;
        }
    }
    result = resp.deferred ? (CUresult) resp.deferred : (CUresult) resp.result;
    if (resp.deferred) dbg("error %d from an earlier async call", resp.deferred);
    dbg("op %u -> %d", op, resp.result);
done:
    pthread_mutex_unlock(&lock);
    return result;
}

static CUresult post(uint32_t op, const void* a, uint32_t alen, const void* b, uint32_t blen) {
    if (syncMode) return rpc2(op, a, alen, b, blen, NULL, 0);
    CUresult result = CUDA_SUCCESS;
    pthread_mutex_lock(&lock);
    RequestHeader req = {op | OP_ASYNC, alen + blen};
    if (sock < 0) {
        result = CUDA_ERROR_NOT_INITIALIZED;
    } else if (!enqueue(&req, sizeof req) || !enqueue(a, alen) || !enqueue(b, blen)) {
        result = connectionLost();
    }
    pthread_mutex_unlock(&lock);
    return result;
}

static CUresult rpc(uint32_t op, const void* args, uint32_t alen, void* out, uint32_t outCap) {
    return rpc2(op, args, alen, NULL, 0, out, outCap);
}

// A program may end right after queuing work (e.g. free + exit); send what is left.
__attribute__((destructor)) static void flushAtExit(void) {
    pthread_mutex_lock(&lock);
    if (sock >= 0) flushQueue();
    pthread_mutex_unlock(&lock);
}

// ---- cubin parsing ---------------------------------------------------------------------

typedef struct {
    const uint8_t* data;
    size_t size;
    int is64;
} Elf;

static uint64_t rd(const uint8_t* p, int n) {
    uint64_t v = 0;
    for (int i = n - 1; i >= 0; i--) v = v << 8 | p[i];
    return v;
}

// Total length of an ELF image held in memory (cuModuleLoadData gets no size).
static size_t elfSize(const uint8_t* p) {
    int is64 = p[4] == 2;
    uint64_t phoff = is64 ? rd(p + 32, 8) : rd(p + 28, 4);
    uint64_t shoff = is64 ? rd(p + 40, 8) : rd(p + 32, 4);
    uint64_t phentsize = rd(p + (is64 ? 54 : 42), 2), phnum = rd(p + (is64 ? 56 : 44), 2);
    uint64_t shentsize = rd(p + (is64 ? 58 : 46), 2), shnum = rd(p + (is64 ? 60 : 48), 2);
    uint64_t end = shoff + shentsize * shnum;
    if (phoff + phentsize * phnum > end) end = phoff + phentsize * phnum;
    for (uint64_t i = 0; i < shnum; i++) {
        const uint8_t* sh = p + shoff + i * shentsize;
        uint32_t type = rd(sh + 4, 4);
        uint64_t off = is64 ? rd(sh + 24, 8) : rd(sh + 16, 4);
        uint64_t size = is64 ? rd(sh + 32, 8) : rd(sh + 20, 4);
        if (type != 8 /* SHT_NOBITS */ && off + size > end) end = off + size;
    }
    return end;
}

static int isElf(const void* image) {
    return memcmp(image, "\177ELF", 4) == 0;
}

// Finds the section called `name`; returns its bytes or NULL.
static const uint8_t* findSection(const Elf* e, const char* name, size_t* len) {
    const uint8_t* p = e->data;
    uint64_t shoff = e->is64 ? rd(p + 40, 8) : rd(p + 32, 4);
    uint64_t shentsize = rd(p + (e->is64 ? 58 : 46), 2), shnum = rd(p + (e->is64 ? 60 : 48), 2);
    uint64_t shstrndx = rd(p + (e->is64 ? 62 : 50), 2);
#define SH(i) (p + shoff + (i) * shentsize)
#define OFF(sh) (e->is64 ? rd((sh) + 24, 8) : rd((sh) + 16, 4))
#define SIZE(sh) (e->is64 ? rd((sh) + 32, 8) : rd((sh) + 20, 4))
    const char* strtab = (const char*) p + OFF(SH(shstrndx));
    for (uint64_t i = 0; i < shnum; i++) {
        if (strcmp(strtab + rd(SH(i), 4), name) == 0) {
            *len = SIZE(SH(i));
            return p + OFF(SH(i));
        }
    }
#undef SH
#undef OFF
#undef SIZE
    return NULL;
}

// Reads parameter offsets/sizes of `kernel` from .nv.info.<kernel> (EIATTR_KPARAM_INFO).
static int parseParams(const Module* m, const char* kernel, Function* f) {
    Elf e = {m->image, m->size, m->image[4] == 2};
    char name[512];
    size_t len;
    snprintf(name, sizeof name, ".nv.info.%s", kernel);
    const uint8_t* p = findSection(&e, name, &len);
    if (!p) return 0;

    f->nparams = 0;
    f->paramBytes = 0;
    for (size_t i = 0; i + 4 <= len;) {
        uint8_t format = p[i], attr = p[i + 1];
        uint16_t size = rd(p + i + 2, 2);
        if (format == 0x03 /* EIFMT_HVAL */ || format == 0x01 /* EIFMT_NVAL */) {
            if (attr == 0x19 /* EIATTR_CBANK_PARAM_SIZE */) f->paramBytes = size;
            i += 4;
            continue;
        }
        if (attr == 0x17 /* EIATTR_KPARAM_INFO */ && size >= 12) {
            uint16_t ordinal = rd(p + i + 8, 2);
            uint16_t offset = rd(p + i + 10, 2);
            uint32_t flags = rd(p + i + 12, 4);
            if (ordinal < 256) {
                f->offset[ordinal] = offset;
                f->size[ordinal] = (flags >> 18) & 0x3fff;
                if (ordinal + 1u > f->nparams) f->nparams = ordinal + 1;
            }
        }
        i += 4 + size;
    }
    return 1;
}

// ---- Driver API ------------------------------------------------------------------------

EXPORT CUresult CUDAAPI cuInit(unsigned int flags) {
    pthread_mutex_lock(&lock);
    if (sock < 0) sock = connectServer();
    pthread_mutex_unlock(&lock);
    if (sock < 0) return CUDA_ERROR_NO_DEVICE;

    uint32_t magic = 0;
    CUresult r = rpc(OP_HELLO, NULL, 0, &magic, sizeof magic);
    if (r != CUDA_SUCCESS || magic != CUDAUSB_MAGIC) return CUDA_ERROR_NO_DEVICE;
    return rpc(OP_INIT, &flags, 4, NULL, 0);
}

EXPORT CUresult CUDAAPI cuDriverGetVersion(int* version) {
    if (!version) return CUDA_ERROR_INVALID_VALUE;
    // Before cuInit report the shim's API level, like the real driver does without a GPU.
    if (sock < 0) {
        *version = CUDA_VERSION;
        return CUDA_SUCCESS;
    }
    return rpc(OP_DRIVER_VERSION, NULL, 0, version, 4);
}

EXPORT CUresult CUDAAPI cuDeviceGetCount(int* count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    return rpc(OP_DEVICE_COUNT, NULL, 0, count, 4);
}

EXPORT CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    return rpc(OP_DEVICE_GET, &ordinal, 4, device, 4);
}

EXPORT CUresult CUDAAPI cuDeviceGetName(char* name, int len, CUdevice dev) {
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    char buf[256] = {0};
    CUresult r = rpc(OP_DEVICE_NAME, &dev, 4, buf, sizeof buf - 1);
    if (r == CUDA_SUCCESS) snprintf(name, len, "%s", buf);
    return r;
}

EXPORT CUresult CUDAAPI cuDeviceGetAttribute(int* value, CUdevice_attribute attrib, CUdevice dev) {
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    int32_t args[2] = {attrib, dev};
    return rpc(OP_DEVICE_ATTRIBUTE, args, sizeof args, value, 4);
}

EXPORT CUresult CUDAAPI cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) {
    CUresult r = cuDeviceGetAttribute(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    return r != CUDA_SUCCESS ? r
                             : cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
}

EXPORT CUresult CUDAAPI cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    uint32_t v = 0;
    CUresult r = rpc(OP_DEVICE_TOTAL_MEM, &dev, 4, &v, 4);
    *bytes = v;
    return r;
}

EXPORT CUresult CUDAAPI cuCtxCreate(CUcontext* ctx, unsigned int flags, CUdevice dev) {
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    uint32_t args[2] = {flags, (uint32_t) dev}, h = 0;
    CUresult r = rpc(OP_CTX_CREATE, args, sizeof args, &h, 4);
    *ctx = (CUcontext) (uintptr_t) h;
    return r;
}

EXPORT CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
    uint32_t h = (uint32_t) (uintptr_t) ctx;
    return rpc(OP_CTX_DESTROY, &h, 4, NULL, 0);
}

EXPORT CUresult CUDAAPI cuCtxSynchronize(void) {
    return rpc(OP_CTX_SYNCHRONIZE, NULL, 0, NULL, 0);
}

EXPORT CUresult CUDAAPI cuMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
    uint32_t v[2] = {0, 0};
    CUresult r = rpc(OP_MEM_GET_INFO, NULL, 0, v, sizeof v);
    if (freeBytes) *freeBytes = v[0];
    if (totalBytes) *totalBytes = v[1];
    return r;
}

EXPORT CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytes) {
    if (!dptr || bytes > UINT32_MAX) return CUDA_ERROR_INVALID_VALUE;
    uint32_t size = bytes, p = 0;
    CUresult r = rpc(OP_MEM_ALLOC, &size, 4, &p, 4);
    *dptr = p;
    return r;
}

EXPORT CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    uint32_t p = dptr;
    return post(OP_MEM_FREE, &p, 4, NULL, 0);
}

EXPORT CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t bytes) {
    if (bytes > UINT32_MAX - 4) return CUDA_ERROR_INVALID_VALUE;
    uint32_t p = dst;
    return post(OP_MEMCPY_HTOD, &p, 4, src, bytes);
}

EXPORT CUresult CUDAAPI cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t bytes) {
    if (bytes > UINT32_MAX) return CUDA_ERROR_INVALID_VALUE;
    uint32_t args[2] = {(uint32_t) src, (uint32_t) bytes};
    return rpc(OP_MEMCPY_DTOH, args, sizeof args, dst, bytes);
}

EXPORT CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    uint32_t args[3] = {(uint32_t) dst, (uint32_t) src, (uint32_t) bytes};
    return post(OP_MEMCPY_DTOD, args, sizeof args, NULL, 0);
}

EXPORT CUresult CUDAAPI cuMemsetD8(CUdeviceptr dst, unsigned char value, size_t count) {
    uint32_t args[3] = {(uint32_t) dst, value, (uint32_t) count};
    return post(OP_MEMSET_D8, args, sizeof args, NULL, 0);
}

EXPORT CUresult CUDAAPI cuMemsetD32(CUdeviceptr dst, unsigned int value, size_t count) {
    uint32_t args[3] = {(uint32_t) dst, value, (uint32_t) count};
    return post(OP_MEMSET_D32, args, sizeof args, NULL, 0);
}

EXPORT CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image) {
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    if (!isElf(image)) {
        fprintf(stderr, "[cuda-usb] only sm_32 cubins are supported (build with nvcc-k1 -cubin)\n");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    Module* m = calloc(1, sizeof *m);
    m->size = elfSize(image);
    m->image = malloc(m->size);
    memcpy(m->image, image, m->size);
    if (m->image[4] != 1) {
        fprintf(stderr, "[cuda-usb] cubin is 64-bit; the tablet needs -m32 (use nvcc-k1)\n");
        free(m->image);
        free(m);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    CUresult r = rpc(OP_MODULE_LOAD, m->image, m->size, &m->remote, 4);
    if (r != CUDA_SUCCESS) {
        free(m->image);
        free(m);
        return r;
    }
    *module = (CUmodule) m;
    return CUDA_SUCCESS;
}

EXPORT CUresult CUDAAPI cuModuleLoad(CUmodule* module, const char* fname) {
    FILE* f = fopen(fname, "rb");
    if (!f) return CUDA_ERROR_FILE_NOT_FOUND;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    void* data = malloc(size);
    size_t got = fread(data, 1, size, f);
    fclose(f);
    CUresult r = got == (size_t) size ? cuModuleLoadData(module, data) : CUDA_ERROR_FILE_NOT_FOUND;
    free(data);
    return r;
}

EXPORT CUresult CUDAAPI cuModuleUnload(CUmodule module) {
    Module* m = (Module*) module;
    if (!m) return CUDA_ERROR_INVALID_HANDLE;
    CUresult r = rpc(OP_MODULE_UNLOAD, &m->remote, 4, NULL, 0);
    free(m->image);
    free(m);
    return r;
}

EXPORT CUresult CUDAAPI cuModuleGetFunction(CUfunction* func, CUmodule module, const char* name) {
    Module* m = (Module*) module;
    if (!func || !m || !name) return CUDA_ERROR_INVALID_VALUE;
    Function* f = calloc(1, sizeof *f);
    if (!parseParams(m, name, f)) {
        free(f);
        return CUDA_ERROR_NOT_FOUND;
    }
    size_t nameLen = strlen(name) + 1;
    CUresult r = rpc2(OP_MODULE_GET_FUNCTION, &m->remote, 4, name, nameLen, &f->remote, 4);
    if (r != CUDA_SUCCESS) {
        free(f);
        return r;
    }
    dbg("%s: %u params, %u bytes", name, f->nparams, f->paramBytes);
    *func = (CUfunction) f;  // leaked with the module, like the real driver's handles
    return CUDA_SUCCESS;
}

EXPORT CUresult CUDAAPI cuLaunchKernel(CUfunction func, unsigned gx, unsigned gy, unsigned gz,
                                       unsigned bx, unsigned by, unsigned bz, unsigned sharedMem,
                                       CUstream stream, void** params, void** extra) {
    Function* f = (Function*) func;
    if (!f) return CUDA_ERROR_INVALID_HANDLE;
    if (extra || (f->nparams && !params)) {
        fprintf(stderr, "[cuda-usb] cuLaunchKernel: only kernelParams are supported\n");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (stream) dbg("streams are not supported, using the default stream");

    size_t total = sizeof(LaunchRequest) + f->nparams * 2 + f->paramBytes;
    uint8_t* buf = calloc(1, total);
    LaunchRequest* req = (LaunchRequest*) buf;
    *req = (LaunchRequest){f->remote, {gx, gy, gz}, {bx, by, bz}, sharedMem, f->nparams, f->paramBytes};
    uint16_t* offsets = (uint16_t*) (req + 1);
    uint8_t* packed = (uint8_t*) (offsets + f->nparams);
    for (uint32_t i = 0; i < f->nparams; i++) {
        offsets[i] = f->offset[i];
        memcpy(packed + f->offset[i], params[i], f->size[i]);
    }
    CUresult r = post(OP_LAUNCH_KERNEL, buf, total, NULL, 0);
    free(buf);
    return r;
}

EXPORT CUresult CUDAAPI cuEventCreate(CUevent* event, unsigned int flags) {
    if (!event) return CUDA_ERROR_INVALID_VALUE;
    uint32_t h = 0;
    CUresult r = rpc(OP_EVENT_CREATE, &flags, 4, &h, 4);
    *event = (CUevent) (uintptr_t) h;
    return r;
}

EXPORT CUresult CUDAAPI cuEventRecord(CUevent event, CUstream stream) {
    uint32_t h = (uint32_t) (uintptr_t) event;
    return post(OP_EVENT_RECORD, &h, 4, NULL, 0);
}

EXPORT CUresult CUDAAPI cuEventSynchronize(CUevent event) {
    uint32_t h = (uint32_t) (uintptr_t) event;
    return rpc(OP_EVENT_SYNCHRONIZE, &h, 4, NULL, 0);
}

EXPORT CUresult CUDAAPI cuEventElapsedTime(float* ms, CUevent start, CUevent end) {
    if (!ms) return CUDA_ERROR_INVALID_VALUE;
    uint32_t args[2] = {(uint32_t) (uintptr_t) start, (uint32_t) (uintptr_t) end};
    return rpc(OP_EVENT_ELAPSED, args, sizeof args, ms, sizeof *ms);
}

EXPORT CUresult CUDAAPI cuEventDestroy(CUevent event) {
    uint32_t h = (uint32_t) (uintptr_t) event;
    return post(OP_EVENT_DESTROY, &h, 4, NULL, 0);
}

EXPORT CUresult CUDAAPI cuGetErrorName(CUresult error, const char** str) {
    static const struct { CUresult code; const char* name; } names[] = {
        {CUDA_SUCCESS, "CUDA_SUCCESS"},
        {CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE"},
        {CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY"},
        {CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED"},
        {CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE"},
        {CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE"},
        {CUDA_ERROR_INVALID_IMAGE, "CUDA_ERROR_INVALID_IMAGE"},
        {CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT"},
        {CUDA_ERROR_NO_BINARY_FOR_GPU, "CUDA_ERROR_NO_BINARY_FOR_GPU"},
        {CUDA_ERROR_FILE_NOT_FOUND, "CUDA_ERROR_FILE_NOT_FOUND"},
        {CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE"},
        {CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND"},
        {CUDA_ERROR_ILLEGAL_ADDRESS, "CUDA_ERROR_ILLEGAL_ADDRESS"},
        {CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"},
        {CUDA_ERROR_LAUNCH_TIMEOUT, "CUDA_ERROR_LAUNCH_TIMEOUT"},
        {CUDA_ERROR_LAUNCH_FAILED, "CUDA_ERROR_LAUNCH_FAILED"},
        {CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED"},
        {CUDA_ERROR_UNKNOWN, "CUDA_ERROR_UNKNOWN"},
    };
    if (!str) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        if (names[i].code == error) {
            *str = names[i].name;
            return CUDA_SUCCESS;
        }
    }
    *str = NULL;
    return CUDA_ERROR_INVALID_VALUE;
}

EXPORT CUresult CUDAAPI cuGetErrorString(CUresult error, const char** str) {
    return cuGetErrorName(error, str);
}
