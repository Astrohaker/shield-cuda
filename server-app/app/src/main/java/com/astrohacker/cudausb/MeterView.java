package com.astrohacker.cudausb;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

/** Horizontal segmented bar: label on the left, value text on the right. */
public class MeterView extends View {
    private static final int SEGMENTS = 32;

    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF rect = new RectF();
    private String label = "", valueText = "—";
    private float fraction = -1, shown;

    public MeterView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    void setLabel(String label) {
        this.label = label;
    }

    /** fraction in 0..1, or < 0 for "not available". */
    void setValue(float fraction, String valueText) {
        this.fraction = fraction;
        this.valueText = valueText;
        postInvalidateOnAnimation();
    }

    @Override
    protected void onMeasure(int widthSpec, int heightSpec) {
        setMeasuredDimension(MeasureSpec.getSize(widthSpec),
                resolveSize((int) Palette.dp(38), heightSpec));
    }

    @Override
    protected void onDraw(Canvas c) {
        float w = getWidth();
        float goal = Math.max(0, Math.min(1, fraction));
        shown += (goal - shown) * 0.2f;
        if (Math.abs(goal - shown) > 0.002f) postInvalidateOnAnimation();
        else shown = goal;

        paint.setTypeface(Palette.MEDIUM);
        paint.setTextSize(Palette.dp(12));
        paint.setColor(Palette.TEXT_DIM);
        paint.setTextAlign(Paint.Align.LEFT);
        c.drawText(label, 0, Palette.dp(13), paint);
        paint.setTypeface(Palette.CONDENSED);
        paint.setTextSize(Palette.dp(14));
        paint.setColor(Palette.TEXT);
        paint.setTextAlign(Paint.Align.RIGHT);
        c.drawText(valueText, w, Palette.dp(14), paint);

        float top = Palette.dp(22), bottom = Palette.dp(34);
        float step = w / SEGMENTS, gap = Palette.dp(2.5f);
        int lit = fraction < 0 ? 0 : Math.round(shown * SEGMENTS);
        if (fraction > 0 && lit == 0) lit = 1;
        for (int i = 0; i < SEGMENTS; i++) {
            paint.setColor(i < lit ? Palette.levelColor((i + 1f) / SEGMENTS) : Palette.SEGMENT_OFF);
            rect.set(i * step, top, (i + 1) * step - gap, bottom);
            c.drawRoundRect(rect, Palette.dp(1.5f), Palette.dp(1.5f), paint);
        }
    }
}
