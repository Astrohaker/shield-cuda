package com.astrohacker.cudausb;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

/**
 * Keeps the process (and the CPU, via a wake lock) alive while the server runs, so host
 * programs keep working with the screen off or another app in front. The server itself is
 * started and stopped by MainActivity through NativeServer.
 */
public class ServerService extends Service {
    private static final String CHANNEL = "server";
    private static final int NOTIFICATION_ID = 1;

    private PowerManager.WakeLock wakeLock;

    static void start(Context context, int port) {
        Intent intent = new Intent(context, ServerService.class).putExtra("port", port);
        if (Build.VERSION.SDK_INT >= 26) context.startForegroundService(intent);
        else context.startService(intent);
    }

    static void stop(Context context) {
        context.stopService(new Intent(context, ServerService.class));
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        int port = intent != null ? intent.getIntExtra("port", NativeServer.DEFAULT_PORT)
                : NativeServer.DEFAULT_PORT;
        Notification notification = buildNotification(port);
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIFICATION_ID, notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
        if (wakeLock == null) {
            PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "cudausb:server");
            wakeLock.acquire();
        }
        return START_NOT_STICKY;
    }

    private Notification buildNotification(int port) {
        Notification.Builder builder;
        if (Build.VERSION.SDK_INT >= 26) {
            NotificationManager nm = getSystemService(NotificationManager.class);
            nm.createNotificationChannel(new NotificationChannel(CHANNEL,
                    getString(R.string.notification_channel), NotificationManager.IMPORTANCE_LOW));
            builder = new Notification.Builder(this, CHANNEL);
        } else {
            builder = new Notification.Builder(this);
        }
        PendingIntent open = PendingIntent.getActivity(this, 0,
                new Intent(this, MainActivity.class), PendingIntent.FLAG_IMMUTABLE);
        return builder.setSmallIcon(R.drawable.ic_stat_server)
                .setContentTitle(getString(R.string.notification_title))
                .setContentText(getString(R.string.notification_text, port))
                .setContentIntent(open)
                .setOngoing(true)
                .build();
    }

    @Override
    public void onDestroy() {
        if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
