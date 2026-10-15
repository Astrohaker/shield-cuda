// JNI bridge between NativeServer.java and server.c (built with CUDAUSB_LIBRARY).
#include <android/log.h>
#include <jni.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "server.h"

#define TAG "cudausb"
#define LOG_CAPACITY 32768

// Log lines from the server thread wait here until the UI takes them.
static pthread_mutex_t logLock = PTHREAD_MUTEX_INITIALIZER;
static char logBuf[LOG_CAPACITY];
static size_t logLen;
static unsigned logDropped;

static void logSink(const char* line) {
    __android_log_write(ANDROID_LOG_INFO, TAG, line);
    size_t n = strlen(line);
    pthread_mutex_lock(&logLock);
    if (logLen + n + 1 < LOG_CAPACITY) {
        memcpy(logBuf + logLen, line, n);
        logLen += n;
        logBuf[logLen++] = '\n';
    } else {
        logDropped++;
    }
    pthread_mutex_unlock(&logLock);
}

JNIEXPORT jint JNICALL Java_com_astrohacker_cudausb_NativeServer_start(JNIEnv* env, jclass cls,
                                                                       jint port) {
    cudausb_set_log_sink(logSink);
    return cudausb_start(port);
}

JNIEXPORT void JNICALL Java_com_astrohacker_cudausb_NativeServer_stop(JNIEnv* env, jclass cls) {
    cudausb_stop();
}

JNIEXPORT void JNICALL Java_com_astrohacker_cudausb_NativeServer_stats(JNIEnv* env, jclass cls,
                                                                      jlongArray out) {
    ServerStats s;
    cudausb_get_stats(&s);
    jsize n = (*env)->GetArrayLength(env, out);
    jsize count = sizeof s / sizeof(uint64_t);
    (*env)->SetLongArrayRegion(env, out, 0, n < count ? n : count, (const jlong*) &s);
}

// "name\ndriverVersion\nccMajor\nccMinor\nSMs\nclockHz\ntotalMem", or null without a CUDA driver.
JNIEXPORT jstring JNICALL Java_com_astrohacker_cudausb_NativeServer_deviceInfo(JNIEnv* env,
                                                                              jclass cls) {
    DeviceInfo d;
    cudausb_set_log_sink(logSink);
    if (!cudausb_device_info(&d)) return NULL;
    char text[256];
    snprintf(text, sizeof text, "%s\n%d\n%d\n%d\n%d\n%d\n%llu", d.name, d.driverVersion, d.ccMajor,
             d.ccMinor, d.multiprocessors, d.clockHz, (unsigned long long) d.totalMem);
    return (*env)->NewStringUTF(env, text);
}

// New log lines since the previous call, or null.
JNIEXPORT jstring JNICALL Java_com_astrohacker_cudausb_NativeServer_takeLog(JNIEnv* env,
                                                                           jclass cls) {
    char text[LOG_CAPACITY + 64];
    pthread_mutex_lock(&logLock);
    size_t n = logLen;
    memcpy(text, logBuf, n);
    if (logDropped) n += snprintf(text + n, 64, "... %u line(s) dropped\n", logDropped);
    logLen = 0;
    logDropped = 0;
    pthread_mutex_unlock(&logLock);
    if (!n) return NULL;
    text[n] = 0;
    return (*env)->NewStringUTF(env, text);
}
