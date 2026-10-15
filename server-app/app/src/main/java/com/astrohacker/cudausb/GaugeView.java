package com.astrohacker.cudausb;

import android.content.Context;
import android.graphics.BlurMaskFilter;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

/** Round gauge made of LED-like segments along a 240° arc, with a peak marker. */
public class GaugeView extends View {
    private static final int SEGMENTS = 44;
    private static final float START = 150, SWEEP = 240;

    private final Paint segment = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint glow = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint ring = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint panel = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF rect = new RectF();

    private String label = "", unit = "";
    private float max = 100, target = -1, shown, peak;
    private int decimals;

    public GaugeView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setLayerType(LAYER_TYPE_SOFTWARE, null);  // BlurMaskFilter needs it
        segment.setStyle(Paint.Style.STROKE);
        segment.setStrokeCap(Paint.Cap.BUTT);
        glow.setStyle(Paint.Style.STROKE);
        glow.setMaskFilter(new BlurMaskFilter(Palette.dp(9), BlurMaskFilter.Blur.NORMAL));
        ring.setStyle(Paint.Style.STROKE);
        ring.setStrokeCap(Paint.Cap.ROUND);
        text.setTextAlign(Paint.Align.CENTER);
    }

    void setup(String label, String unit, float max, int decimals) {
        this.label = label;
        this.unit = unit;
        this.max = max;
        this.decimals = decimals;
        invalidate();
    }

    void setMax(float max) {
        if (max > 0 && max != this.max) {
            this.max = max;
            invalidate();
        }
    }

    /** value < 0 means "not available". */
    void setValue(float value, float peak) {
        target = value;
        this.peak = peak;
        postInvalidateOnAnimation();
    }

    @Override
    protected void onMeasure(int widthSpec, int heightSpec) {
        int w = MeasureSpec.getSize(widthSpec);
        setMeasuredDimension(w, resolveSize((int) (w * 0.92f), heightSpec));
    }

    @Override
    protected void onDraw(Canvas c) {
        float w = getWidth(), h = getHeight();
        float corner = Palette.dp(10);

        panel.setStyle(Paint.Style.FILL);
        panel.setColor(Palette.PANEL);
        rect.set(0, 0, w, h);
        c.drawRoundRect(rect, corner, corner, panel);
        panel.setStyle(Paint.Style.STROKE);
        panel.setStrokeWidth(Palette.dp(1));
        panel.setColor(Palette.PANEL_BORDER);
        rect.inset(Palette.dp(0.5f), Palette.dp(0.5f));
        c.drawRoundRect(rect, corner, corner, panel);

        // Ease the shown value towards the latest sample.
        float goal = Math.max(0, target);
        shown += (goal - shown) * 0.18f;
        if (Math.abs(goal - shown) > max * 0.002f) postInvalidateOnAnimation();
        else shown = goal;
        float fraction = Math.min(1f, shown / max);

        float thickness = Math.min(w, h) * 0.085f;
        float radius = Math.min(w / 2f, h / 1.75f) - thickness - Palette.dp(8);
        float cx = w / 2f, cy = h * 0.53f;
        rect.set(cx - radius, cy - radius, cx + radius, cy + radius);

        float step = SWEEP / SEGMENTS, gap = step * 0.28f;
        int lit = target < 0 ? 0 : Math.round(fraction * SEGMENTS);
        int peakSeg = Math.min(SEGMENTS - 1, Math.round(Math.min(1f, peak / max) * SEGMENTS) - 1);
        segment.setStrokeWidth(thickness);
        glow.setStrokeWidth(thickness);
        for (int i = 0; i < SEGMENTS; i++) {
            float a = START + i * step;
            int color = Palette.levelColor((i + 1f) / SEGMENTS);
            if (i < lit) {
                glow.setColor(color);
                glow.setAlpha(110);
                c.drawArc(rect, a, step - gap, false, glow);
                segment.setColor(color);
            } else if (i == peakSeg && target >= 0) {
                segment.setColor(color);
                segment.setAlpha(150);
            } else {
                segment.setColor(Palette.SEGMENT_OFF);
            }
            c.drawArc(rect, a, step - gap, false, segment);
        }

        // Thin inner ring with the exact value.
        float inner = radius - thickness * 0.95f;
        rect.set(cx - inner, cy - inner, cx + inner, cy + inner);
        ring.setStrokeWidth(Palette.dp(2));
        ring.setColor(Palette.GRID);
        c.drawArc(rect, START, SWEEP, false, ring);
        if (target >= 0 && fraction > 0) {
            ring.setColor(Palette.GREEN_BRIGHT);
            c.drawArc(rect, START, SWEEP * fraction, false, ring);
        }

        text.setTypeface(Palette.CONDENSED);
        text.setColor(target < 0 ? Palette.TEXT_DIM : Palette.TEXT);
        text.setTextSize(radius * 0.52f);
        c.drawText(target < 0 ? "N/A" : Palette.format(shown, decimals), cx, cy + radius * 0.16f, text);

        text.setTypeface(Palette.MEDIUM);
        text.setColor(Palette.GREEN);
        text.setTextSize(radius * 0.17f);
        c.drawText(unit, cx, cy + radius * 0.42f, text);

        text.setColor(Palette.TEXT_DIM);
        text.setTextSize(Math.max(Palette.dp(11), radius * 0.15f));
        text.setLetterSpacing(0.12f);
        c.drawText(label, cx, h - Palette.dp(12), text);
        text.setLetterSpacing(0);

        // Scale ends, under the ends of the arc (at ±30° below the horizontal).
        text.setTextSize(Palette.dp(10));
        float ty = cy + radius * 0.5f + thickness * 0.5f + Palette.dp(14);
        c.drawText("0", cx - radius * 0.87f, ty, text);
        c.drawText(Palette.format(max, 0), cx + radius * 0.87f, ty, text);
    }
}
