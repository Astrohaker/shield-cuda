// Library API of cudausb-server (server.c built with -DCUDAUSB_LIBRARY), used by server-app.
#pragma once
#include <stdint.h>

// Counters since cudausb_start(); all fields are uint64_t (copied as an array).
typedef struct {
    uint64_t running;          // 1 while the server thread is alive
    uint64_t port;
    uint64_t clientConnected;  // 1 while a host program is connected
    uint64_t sessions;         // host programs served
    uint64_t requests;         // CUDA calls handled
    uint64_t launches;         // cuLaunchKernel calls
    uint64_t errors;           // calls that returned an error
    uint64_t bytesIn;          // received from the PC
    uint64_t bytesOut;         // sent to the PC
    uint64_t busyNs;           // time spent executing calls
    uint64_t clientMem;        // bytes the current client has cuMemAlloc'ed
    uint64_t contexts;         // contexts of the current client
    uint64_t modules;          // modules of the current client
} ServerStats;

typedef struct {
    char name[128];
    int driverVersion;  // e.g. 7050
    int ccMajor, ccMinor;
    int multiprocessors;
    int clockHz;  // the Tegra driver reports Hz, not kHz
    uint64_t totalMem;
} DeviceInfo;

int cudausb_start(int port);  // 0 or -errno; serves in a background thread
void cudausb_stop(void);      // waits until the server thread has exited
void cudausb_get_stats(ServerStats* out);
int cudausb_device_info(DeviceInfo* info);  // 1 on success
void cudausb_set_log_sink(void (*sink)(const char* line));  // called from the server thread
