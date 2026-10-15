package com.astrohacker.cudausb;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/** Start/stop buttons and live monitoring of the server and the Tegra K1. */
public class MainActivity extends Activity {
    private static final long SAMPLE_MS = 500;
    private static final int LOG_MAX_CHARS = 20000;
    /** Typical adb throughput on Android 6 is ~7 MB/s, so the USB gauge tops out at 10. */
    private static final float USB_GAUGE_MAX = 10;

    // Survive the activity being recreated while the server keeps running.
    private static long serverStartedAt = -1;
    private static final SpannableStringBuilder logText = new SpannableStringBuilder();

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final long[] stats = new long[NativeServer.STAT_COUNT];
    private final long[] prevStats = new long[NativeServer.STAT_COUNT];
    private long prevSampleAt = -1;
    private final SimpleDateFormat clock = new SimpleDateFormat("HH:mm:ss", Locale.US);

    private SystemMetrics metrics;
    private long gpuTotalMem;
    private boolean stopping;

    private GaugeView gaugeGpu, gaugeClock, gaugeUsb, gaugeCpu, gaugeRam;
    private ChartView chartGpu, chartClock, chartUsb, chartCalls, chartCpu, chartMem;
    private MeterView meterBusy, meterClientMem, meterTemp;
    private TextView status, uptime, errorText, logView;
    private TextView statClient, statSessions, statCalls, statLaunches, statErrors, statIn, statOut,
            statObjects;
    private PulseDot statusDot;
    private Button startButton, stopButton;
    private ScrollView logScroll;
    private final Peak peakGpu = new Peak(), peakClock = new Peak(), peakUsb = new Peak(),
            peakCpu = new Peak(), peakRam = new Peak();

    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            sample();
            handler.postDelayed(this, SAMPLE_MS);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        metrics = new SystemMetrics(this);

        status = findViewById(R.id.status);
        statusDot = findViewById(R.id.statusDot);
        uptime = findViewById(R.id.uptime);
        errorText = findViewById(R.id.error);
        logView = findViewById(R.id.log);
        logScroll = findViewById(R.id.logScroll);
        logScroll.setNestedScrollingEnabled(true);
        startButton = findViewById(R.id.startButton);
        stopButton = findViewById(R.id.stopButton);
        startButton.setOnClickListener(v -> startServer());
        stopButton.setOnClickListener(v -> stopServer());
        ((TextView) findViewById(R.id.address))
                .setText(getString(R.string.address, NativeServer.DEFAULT_PORT));

        gaugeGpu = findViewById(R.id.gaugeGpu);
        gaugeClock = findViewById(R.id.gaugeClock);
        gaugeUsb = findViewById(R.id.gaugeUsb);
        gaugeCpu = findViewById(R.id.gaugeCpu);
        gaugeRam = findViewById(R.id.gaugeRam);
        gaugeGpu.setup("ЗАГРУЗКА GPU", "%", 100, 0);
        gaugeClock.setup("ЧАСТОТА GPU", "МГц", 852, 0);
        gaugeUsb.setup("USB", "МБ/с", USB_GAUGE_MAX, 2);
        gaugeCpu.setup("ЗАГРУЗКА CPU", "%", 100, 0);
        gaugeRam.setup("ПАМЯТЬ", "%", 100, 0);

        chartGpu = ((ChartView) findViewById(R.id.chartGpu))
                .setup("ЗАГРУЗКА GPU", "%", 100, 0).addSeries("GPU", Palette.GREEN);
        chartClock = ((ChartView) findViewById(R.id.chartClock))
                .setup("ЧАСТОТА GPU", "МГц", 852, 0).addSeries("GPU", Palette.TEAL);
        chartUsb = ((ChartView) findViewById(R.id.chartUsb))
                .setup("USB-ТРАФИК", "МБ/с", 0, 2)
                .addSeries("ПК → GPU", Palette.GREEN).addSeries("GPU → ПК", Palette.TEAL);
        chartCalls = ((ChartView) findViewById(R.id.chartCalls))
                .setup("CUDA-ВЫЗОВЫ", "/с", 0, 0)
                .addSeries("вызовы", Palette.GREEN).addSeries("запуски ядер", Palette.LIME);
        chartCpu = ((ChartView) findViewById(R.id.chartCpu))
                .setup("ЗАГРУЗКА CPU", "%", 100, 0).addSeries("CPU", Palette.GREEN);
        chartMem = ((ChartView) findViewById(R.id.chartMem))
                .setup("ПАМЯТЬ", "МБ", 0, 0)
                .addSeries("ОЗУ занято", Palette.GREEN).addSeries("выделено клиентом", Palette.TEAL);

        meterBusy = findViewById(R.id.meterBusy);
        meterBusy.setLabel("Сервер занят вызовами");
        meterClientMem = findViewById(R.id.meterClientMem);
        meterClientMem.setLabel("Память GPU у клиента");
        meterTemp = findViewById(R.id.meterTemp);
        meterTemp.setLabel("Температура батареи");

        LinearLayout rows = findViewById(R.id.statRows);
        statClient = addStatRow(rows, "Клиент");
        statSessions = addStatRow(rows, "Сеансов");
        statCalls = addStatRow(rows, "CUDA-вызовов");
        statLaunches = addStatRow(rows, "Запусков ядер");
        statErrors = addStatRow(rows, "Ошибок");
        statIn = addStatRow(rows, "Принято с ПК");
        statOut = addStatRow(rows, "Отправлено на ПК");
        statObjects = addStatRow(rows, "Контексты / модули");

        logView.setText(logText);
        loadDeviceInfo();
        if (Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                        != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {Manifest.permission.POST_NOTIFICATIONS}, 1);
        }
    }

    private TextView addStatRow(LinearLayout parent, String label) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(0, (int) Palette.dp(3), 0, (int) Palette.dp(3));
        TextView name = new TextView(this);
        name.setText(label);
        name.setTextColor(Palette.TEXT_DIM);
        name.setTextSize(13);
        row.addView(name, new LinearLayout.LayoutParams(0, -2, 1));
        TextView value = new TextView(this);
        value.setTypeface(Palette.MONO);
        value.setTextColor(Palette.TEXT);
        value.setTextSize(13);
        value.setText("—");
        row.addView(value);
        parent.addView(row);
        return value;
    }

    private void loadDeviceInfo() {
        TextView view = findViewById(R.id.deviceInfo);
        new Thread(() -> {
            String info = NativeServer.deviceInfo();
            runOnUiThread(() -> {
                if (info == null) {
                    view.setText(R.string.no_cuda);
                    view.setTextColor(Palette.RED);
                    startButton.setEnabled(false);
                    return;
                }
                String[] f = info.split("\n");
                gpuTotalMem = Long.parseLong(f[6]);
                int driver = Integer.parseInt(f[1]);
                view.setText(getString(R.string.device_info, f[0], Integer.parseInt(f[2]),
                        Integer.parseInt(f[3]), Integer.parseInt(f[4]),
                        Integer.parseInt(f[5]) / 1000000, (int) (gpuTotalMem >> 20),
                        driver / 1000, driver % 1000 / 10));
            });
        }, "device-info").start();
    }

    @Override
    protected void onResume() {
        super.onResume();
        handler.post(tick);
    }

    @Override
    protected void onPause() {
        handler.removeCallbacks(tick);
        super.onPause();
    }

    private void startServer() {
        errorText.setVisibility(View.GONE);
        int port = NativeServer.DEFAULT_PORT;
        int result = NativeServer.start(port);
        if (result != 0) {
            errorText.setText(result == -98 /* EADDRINUSE */
                    ? getString(R.string.port_busy, port)
                    : getString(R.string.start_failed, result));
            errorText.setVisibility(View.VISIBLE);
            sample();
            return;
        }
        serverStartedAt = SystemClock.elapsedRealtime();
        prevSampleAt = -1;
        ServerService.start(this, port);
        sample();
    }

    private void stopServer() {
        stopping = true;
        stopButton.setEnabled(false);
        startButton.setEnabled(false);
        stopButton.setText(R.string.stopping);
        // Blocks until a CUDA call in progress returns, so not on the UI thread.
        new Thread(() -> {
            NativeServer.stop();
            runOnUiThread(() -> {
                stopping = false;
                serverStartedAt = -1;
                stopButton.setText(R.string.stop);
                ServerService.stop(this);
                sample();
            });
        }, "server-stop").start();
    }

    private void sample() {
        long now = SystemClock.elapsedRealtime();
        metrics.sample();
        NativeServer.stats(stats);
        boolean running = stats[NativeServer.RUNNING] != 0;
        boolean client = stats[NativeServer.CLIENT_CONNECTED] != 0;
        if (!running && serverStartedAt >= 0 && !stopping) {
            serverStartedAt = -1;  // the server thread ended by itself
            ServerService.stop(this);
        }

        // Per-second rates from the counters.
        float dt = prevSampleAt < 0 ? 0 : (now - prevSampleAt) / 1000f;
        float inRate = 0, outRate = 0, callRate = 0, launchRate = 0, busy = 0;
        if (dt > 0 && running) {
            inRate = delta(NativeServer.BYTES_IN) / dt / 1e6f;
            outRate = delta(NativeServer.BYTES_OUT) / dt / 1e6f;
            callRate = delta(NativeServer.REQUESTS) / dt;
            launchRate = delta(NativeServer.LAUNCHES) / dt;
            busy = Math.min(1f, delta(NativeServer.BUSY_NS) / (dt * 1e9f));
        }
        System.arraycopy(stats, 0, prevStats, 0, stats.length);
        prevSampleAt = now;

        // Status line and buttons.
        if (!stopping) {
            startButton.setEnabled(!running && gpuTotalMem > 0);
            stopButton.setEnabled(running);
        }
        if (running) {
            status.setText(client ? R.string.status_client : R.string.status_running);
            status.setTextColor(client ? Palette.GREEN_BRIGHT : Palette.GREEN);
            statusDot.set(client ? Palette.GREEN_BRIGHT : Palette.GREEN, true);
            uptime.setText(getString(R.string.uptime, serverStartedAt < 0 ? "—"
                    : formatDuration((now - serverStartedAt) / 1000)));
        } else {
            status.setText(R.string.status_stopped);
            status.setTextColor(Palette.TEXT_DIM);
            statusDot.set(Palette.RED, false);
            uptime.setText("");
        }

        // Gauges.
        if (metrics.gpuMaxClockMhz > 0) {
            gaugeClock.setMax(metrics.gpuMaxClockMhz);
            chartClock.setFixedMax(metrics.gpuMaxClockMhz);
        }
        float usb = inRate + outRate;
        float ramUsedMb = (metrics.ramTotal - metrics.ramAvailable) / 1048576f;
        float ramPct = metrics.ramTotal > 0 ? 100f * ramUsedMb * 1048576f / metrics.ramTotal : -1;
        gaugeGpu.setValue(metrics.gpuLoad, peakGpu.add(metrics.gpuLoad));
        gaugeClock.setValue(metrics.gpuClockMhz, peakClock.add(metrics.gpuClockMhz));
        gaugeUsb.setValue(running ? usb : -1, peakUsb.add(usb));
        gaugeCpu.setValue(metrics.cpuLoad, peakCpu.add(metrics.cpuLoad));
        gaugeRam.setValue(ramPct, peakRam.add(ramPct));

        // Charts.
        long clientMem = stats[NativeServer.CLIENT_MEM];
        chartGpu.push(metrics.gpuLoad);
        chartClock.push(metrics.gpuClockMhz);
        chartUsb.push(inRate, outRate);
        chartCalls.push(callRate, launchRate);
        chartCpu.push(metrics.cpuLoad);
        chartMem.setFixedMax((float) Math.ceil(metrics.ramTotal / 1048576f / 256) * 256);
        chartMem.push(ramUsedMb, clientMem / 1048576f);

        // Server panel.
        meterBusy.setValue(running ? busy : -1, running ? Math.round(busy * 100) + " %" : "—");
        meterClientMem.setValue(gpuTotalMem > 0 ? (float) clientMem / gpuTotalMem : -1,
                formatBytes(clientMem));
        meterTemp.setValue(metrics.batteryTemp < 0 ? -1 : metrics.batteryTemp / 60f,
                Palette.format(metrics.batteryTemp, 1) + " °C");
        statClient.setText(!running ? "—" : client ? "подключён" : "ожидание");
        statClient.setTextColor(client ? Palette.GREEN_BRIGHT : Palette.TEXT);
        statSessions.setText(String.valueOf(stats[NativeServer.SESSIONS]));
        statCalls.setText(String.format("%,d", stats[NativeServer.REQUESTS]));
        statLaunches.setText(String.format("%,d", stats[NativeServer.LAUNCHES]));
        long errors = stats[NativeServer.ERRORS];
        statErrors.setText(String.valueOf(errors));
        statErrors.setTextColor(errors > 0 ? Palette.ORANGE : Palette.TEXT);
        statIn.setText(formatBytes(stats[NativeServer.BYTES_IN]));
        statOut.setText(formatBytes(stats[NativeServer.BYTES_OUT]));
        statObjects.setText(stats[NativeServer.CONTEXTS] + " / " + stats[NativeServer.MODULES]);

        appendLog(NativeServer.takeLog());
    }

    private long delta(int index) {
        return Math.max(0, stats[index] - prevStats[index]);
    }

    private void appendLog(String lines) {
        if (lines == null) return;
        String time = clock.format(new Date());
        for (String line : lines.split("\n")) {
            if (line.isEmpty()) continue;
            int start = logText.length();
            logText.append(time).append("  ");
            logText.setSpan(new ForegroundColorSpan(Palette.TEXT_DIM), start, logText.length(),
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
            int color = line.contains("failed") || line.contains("cannot") ? Palette.ORANGE
                    : line.contains("connected") && !line.contains("dis") ? Palette.GREEN_BRIGHT
                    : Palette.TEXT;
            start = logText.length();
            logText.append(line).append('\n');
            logText.setSpan(new ForegroundColorSpan(color), start, logText.length(),
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        if (logText.length() > LOG_MAX_CHARS) {
            int cut = logText.length() - LOG_MAX_CHARS;
            while (cut < logText.length() && logText.charAt(cut - 1) != '\n') cut++;
            logText.delete(0, cut);
        }
        logView.setText(logText);
        // scrollTo, not fullScroll: fullScroll moves focus, and the outer ScrollView follows it.
        logScroll.post(() -> logScroll.scrollTo(0, logView.getHeight()));
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " Б";
        if (bytes < 1048576) return String.format("%.1f КБ", bytes / 1024.0);
        if (bytes < 1073741824) return String.format("%.1f МБ", bytes / 1048576.0);
        return String.format("%.2f ГБ", bytes / 1073741824.0);
    }

    private static String formatDuration(long seconds) {
        return String.format(Locale.US, "%02d:%02d:%02d", seconds / 3600, seconds / 60 % 60,
                seconds % 60);
    }

    /** Maximum over the last minute of samples, for the gauges' peak markers. */
    private static final class Peak {
        private final float[] values = new float[ChartView.CAPACITY];
        private int next;

        float add(float v) {
            values[next] = v;
            next = (next + 1) % values.length;
            float max = 0;
            for (float x : values) max = Math.max(max, x);
            return max;
        }
    }
}
