// Wire protocol between the host libcuda.so shim and the tablet server.
// RPC over TCP (adb forward), little-endian on both ends.
//   request:  RequestHeader + payload
//   response: ResponseHeader + payload (payload only meaningful when result == CUDA_SUCCESS)
// Requests with OP_ASYNC set get no response; the server keeps the first error among them and
// returns it as `deferred` in the next response, like CUDA reports launch errors at the next
// synchronizing call.
// Device-side handles and pointers come from a 32-bit process and travel as uint32_t.
#pragma once
#include <stdint.h>

#define CUDAUSB_DEFAULT_PORT 7777
#define CUDAUSB_MAGIC 0x32425355u  // "USB2", sent by OP_HELLO
#define OP_ASYNC 0x80000000u

typedef struct {
    uint32_t op;
    uint32_t len;
} RequestHeader;

typedef struct {
    int32_t result;    // CUresult of this request
    int32_t deferred;  // first error of async requests since the previous response, or 0
    uint32_t len;
} ResponseHeader;

enum {
    OP_HELLO = 1,         // -> u32 magic
    OP_INIT,              // u32 flags
    OP_DRIVER_VERSION,    // -> i32
    OP_DEVICE_COUNT,      // -> i32
    OP_DEVICE_GET,        // i32 ordinal -> i32 device
    OP_DEVICE_NAME,       // i32 device -> char[]
    OP_DEVICE_ATTRIBUTE,  // i32 attrib, i32 device -> i32
    OP_DEVICE_TOTAL_MEM,  // i32 device -> u32
    OP_CTX_CREATE,        // u32 flags, i32 device -> u32 ctx
    OP_CTX_DESTROY,       // u32 ctx
    OP_CTX_SYNCHRONIZE,
    OP_MEM_GET_INFO,      // -> u32 free, u32 total
    OP_MEM_ALLOC,         // u32 size -> u32 dptr
    OP_MEM_FREE,          // u32 dptr
    OP_MEMCPY_HTOD,       // u32 dst, bytes...
    OP_MEMCPY_DTOH,       // u32 src, u32 size -> bytes
    OP_MEMCPY_DTOD,       // u32 dst, u32 src, u32 size
    OP_MEMSET_D8,         // u32 dst, u32 value, u32 count
    OP_MEMSET_D32,        // u32 dst, u32 value, u32 count
    OP_MODULE_LOAD,       // cubin bytes -> u32 module
    OP_MODULE_UNLOAD,     // u32 module
    OP_MODULE_GET_FUNCTION,  // u32 module, char[] name -> u32 function
    OP_LAUNCH_KERNEL,     // LaunchRequest, u16 offsets[nparams], u8 params[paramBytes]
    OP_EVENT_CREATE,      // u32 flags -> u32 event
    OP_EVENT_RECORD,      // u32 event
    OP_EVENT_SYNCHRONIZE, // u32 event
    OP_EVENT_ELAPSED,     // u32 start, u32 end -> float ms
    OP_EVENT_DESTROY,     // u32 event
};

typedef struct {
    uint32_t function;
    uint32_t grid[3];
    uint32_t block[3];
    uint32_t sharedMemBytes;
    uint32_t nparams;
    uint32_t paramBytes;
} LaunchRequest;
