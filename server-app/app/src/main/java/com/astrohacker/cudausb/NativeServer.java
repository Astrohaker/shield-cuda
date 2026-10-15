package com.astrohacker.cudausb;

/** cudausb-server (../server/server.c) running inside this process. */
final class NativeServer {
    static final int DEFAULT_PORT = 7777;

    // Indices into the array filled by stats(), in the order of ServerStats in server.h.
    static final int RUNNING = 0, PORT = 1, CLIENT_CONNECTED = 2, SESSIONS = 3, REQUESTS = 4,
            LAUNCHES = 5, ERRORS = 6, BYTES_IN = 7, BYTES_OUT = 8, BUSY_NS = 9, CLIENT_MEM = 10,
            CONTEXTS = 11, MODULES = 12, STAT_COUNT = 13;

    static {
        System.loadLibrary("cudausbserver");
    }

    private NativeServer() {}

    /** Starts listening on 127.0.0.1:port; returns 0 or -errno. */
    static native int start(int port);

    /** Stops the server; blocks until the current CUDA call returns. */
    static native void stop();

    static native void stats(long[] out);

    /** Newline-separated fields (see jni.c), or null if the device has no CUDA driver. */
    static native String deviceInfo();

    /** Log lines since the previous call, or null. */
    static native String takeLog();
}
