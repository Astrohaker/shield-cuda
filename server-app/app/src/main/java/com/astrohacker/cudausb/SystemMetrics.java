package com.astrohacker.cudausb;

import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/** Readings of the Tegra K1 GPU (gk20a sysfs), CPU, RAM and battery; -1 when unavailable. */
final class SystemMetrics {
    private static final String GK20A = "/sys/devices/platform/host1x/gk20a.0/";
    private static final String DEVFREQ = GK20A + "devfreq/gk20a.0/";

    private final Context context;
    private final ActivityManager activityManager;
    private final ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
    private long prevCpuTotal = -1, prevCpuIdle;

    float gpuLoad = -1;      // percent
    float gpuClockMhz = -1;
    float gpuMaxClockMhz = -1;
    float cpuLoad = -1;      // percent, all cores
    long ramTotal, ramAvailable;
    float batteryTemp = -1;  // °C

    SystemMetrics(Context context) {
        this.context = context;
        activityManager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
    }

    void sample() {
        long load = readLong(GK20A + "load");  // per mille
        gpuLoad = load < 0 ? -1 : Math.min(100f, load / 10f);
        long freq = readLong(DEVFREQ + "cur_freq");
        gpuClockMhz = freq < 0 ? -1 : freq / 1e6f;
        long max = readLong(DEVFREQ + "max_freq");
        gpuMaxClockMhz = max < 0 ? -1 : max / 1e6f;

        sampleCpu();

        activityManager.getMemoryInfo(memoryInfo);
        ramTotal = memoryInfo.totalMem;
        ramAvailable = memoryInfo.availMem;

        Intent battery = context.registerReceiver(null,
                new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        int temp = battery != null ? battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) : -1;
        batteryTemp = temp < 0 ? -1 : temp / 10f;
    }

    private void sampleCpu() {
        String line = readLine("/proc/stat");
        if (line == null || !line.startsWith("cpu ")) {
            cpuLoad = -1;
            return;
        }
        String[] f = line.trim().split("\\s+");
        long total = 0;
        for (int i = 1; i < f.length; i++) total += Long.parseLong(f[i]);
        long idle = Long.parseLong(f[4]) + (f.length > 5 ? Long.parseLong(f[5]) : 0);
        if (prevCpuTotal >= 0 && total > prevCpuTotal) {
            cpuLoad = 100f * (1f - (float) (idle - prevCpuIdle) / (total - prevCpuTotal));
        }
        prevCpuTotal = total;
        prevCpuIdle = idle;
    }

    private static long readLong(String path) {
        String s = readLine(path);
        if (s == null) return -1;
        try {
            return Long.parseLong(s.trim());
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    private static String readLine(String path) {
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            return r.readLine();
        } catch (IOException | SecurityException e) {
            return null;
        }
    }
}
