package com.astrohacker.cudausb;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.os.SystemClock;
import android.util.AttributeSet;
import android.view.View;

/** Status LED; pulses while active. */
public class PulseDot extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private int color = Palette.TEXT_DIM;
    private boolean pulsing;

    public PulseDot(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    void set(int color, boolean pulsing) {
        if (color == this.color && pulsing == this.pulsing) return;
        this.color = color;
        this.pulsing = pulsing;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas c) {
        float cx = getWidth() / 2f, cy = getHeight() / 2f, r = Math.min(cx, cy);
        paint.setColor(color);
        if (pulsing) {
            float t = (SystemClock.uptimeMillis() % 1600) / 1600f;
            paint.setAlpha((int) (110 * (1 - t)));
            c.drawCircle(cx, cy, r * (0.45f + 0.55f * t), paint);
            postInvalidateOnAnimation();
        }
        paint.setAlpha(255);
        c.drawCircle(cx, cy, r * 0.42f, paint);
    }
}
