package com.astrohacker.cudausb;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.DashPathEffect;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Shader;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/** Scrolling time-series chart (newest sample on the right) with a grid and gradient fill. */
public class ChartView extends View {
    static final int CAPACITY = 120;

    private static final class Series {
        final String name;
        final int color;
        final float[] values = new float[CAPACITY];
        Series(String name, int color) {
            this.name = name;
            this.color = color;
        }
    }

    private final List<Series> series = new ArrayList<>();
    private final Paint panel = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint grid = new Paint();
    private final Paint line = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path path = new Path();
    private final RectF rect = new RectF();

    private String title = "", unit = "";
    private float fixedMax;  // 0 = autoscale
    private int decimals;
    private int count;       // samples stored, up to CAPACITY
    private int head;        // index of the next sample
    private long total;      // samples ever pushed, moves the vertical grid
    private float samplesPerGridLine = 20;

    public ChartView(Context context, AttributeSet attrs) {
        super(context, attrs);
        grid.setStrokeWidth(Palette.dp(1));
        grid.setPathEffect(new DashPathEffect(new float[] {Palette.dp(3), Palette.dp(4)}, 0));
        line.setStyle(Paint.Style.STROKE);
        line.setStrokeWidth(Palette.dp(2));
        line.setStrokeJoin(Paint.Join.ROUND);
    }

    ChartView setup(String title, String unit, float fixedMax, int decimals) {
        this.title = title;
        this.unit = unit;
        this.fixedMax = fixedMax;
        this.decimals = decimals;
        return this;
    }

    ChartView addSeries(String name, int color) {
        series.add(new Series(name, color));
        return this;
    }

    void setFixedMax(float max) {
        fixedMax = max;
    }

    /** One value per series; negative values are drawn as 0. */
    void push(float... values) {
        for (int i = 0; i < series.size(); i++) {
            series.get(i).values[head] = i < values.length ? Math.max(0, values[i]) : 0;
        }
        head = (head + 1) % CAPACITY;
        count = Math.min(CAPACITY, count + 1);
        total++;
        invalidate();
    }

    void setSamplesPerGridLine(float n) {
        samplesPerGridLine = n;
    }

    private float value(Series s, int age) {  // age 0 = newest
        return s.values[(head - 1 - age + 2 * CAPACITY) % CAPACITY];
    }

    private float scaleMax() {
        if (fixedMax > 0) return fixedMax;
        float m = 0;
        for (Series s : series) for (int i = 0; i < count; i++) m = Math.max(m, value(s, i));
        if (m <= 0) return 1;
        float mag = (float) Math.pow(10, Math.floor(Math.log10(m)));
        for (float k : new float[] {1, 2, 2.5f, 5, 10}) if (k * mag >= m * 1.1f) return k * mag;
        return 10 * mag;
    }

    @Override
    protected void onMeasure(int widthSpec, int heightSpec) {
        int w = MeasureSpec.getSize(widthSpec);
        setMeasuredDimension(w, resolveSize((int) Palette.dp(170), heightSpec));
    }

    @Override
    protected void onDraw(Canvas c) {
        float w = getWidth(), h = getHeight(), pad = Palette.dp(12), corner = Palette.dp(10);

        panel.setStyle(Paint.Style.FILL);
        panel.setColor(Palette.PANEL);
        rect.set(0, 0, w, h);
        c.drawRoundRect(rect, corner, corner, panel);
        panel.setStyle(Paint.Style.STROKE);
        panel.setStrokeWidth(Palette.dp(1));
        panel.setColor(Palette.PANEL_BORDER);
        rect.inset(Palette.dp(0.5f), Palette.dp(0.5f));
        c.drawRoundRect(rect, corner, corner, panel);

        // Header: title on the left, current values on the right.
        text.setTypeface(Palette.MEDIUM);
        text.setTextSize(Palette.dp(11));
        text.setLetterSpacing(0.1f);
        text.setColor(Palette.TEXT_DIM);
        text.setTextAlign(Paint.Align.LEFT);
        c.drawText(title, pad, pad + Palette.dp(10), text);
        text.setLetterSpacing(0);

        float x = w - pad;
        text.setTextAlign(Paint.Align.RIGHT);
        for (int i = series.size() - 1; i >= 0; i--) {
            Series s = series.get(i);
            String v = (count > 0 ? Palette.format(value(s, 0), decimals) : "—") + " " + unit;
            text.setTypeface(Palette.CONDENSED);
            text.setTextSize(Palette.dp(18));
            text.setColor(s.color);
            c.drawText(v, x, pad + Palette.dp(14), text);
            float vw = text.measureText(v);
            if (series.size() > 1) {
                text.setTypeface(Palette.MEDIUM);
                text.setTextSize(Palette.dp(10));
                text.setColor(Palette.TEXT_DIM);
                c.drawText(s.name, x, pad + Palette.dp(28), text);
                vw = Math.max(vw, text.measureText(s.name));
            }
            x -= vw + Palette.dp(16);
        }

        float top = pad + Palette.dp(series.size() > 1 ? 36 : 26), bottom = h - pad;
        float left = pad, right = w - pad;
        float max = scaleMax();

        // Horizontal grid with labels.
        text.setTypeface(Palette.MONO);
        text.setTextSize(Palette.dp(9));
        text.setTextAlign(Paint.Align.LEFT);
        text.setColor(Palette.TEXT_DIM);
        grid.setColor(Palette.GRID);
        for (int i = 0; i <= 4; i++) {
            float y = bottom - (bottom - top) * i / 4f;
            c.drawLine(left, y, right, y, grid);
        }
        // Vertical grid scrolls with the data.
        float dx = (right - left) / (CAPACITY - 1);
        int every = Math.round(samplesPerGridLine);
        for (int age = (int) (total % every); age < CAPACITY; age += every) {
            float gx = right - age * dx;
            c.drawLine(gx, top, gx, bottom, grid);
        }
        // Axis labels on a backing so grid lines do not cross them.
        panel.setStyle(Paint.Style.FILL);
        panel.setColor(Palette.PANEL);
        for (int i = 1; i <= 4; i++) {
            float y = bottom - (bottom - top) * i / 4f;
            float v = max * i / 4f;
            String label = Palette.format(v, v == Math.round(v) ? 0 : 1);
            rect.set(left, y + Palette.dp(1), left + text.measureText(label) + Palette.dp(6),
                    y + Palette.dp(13));
            c.drawRect(rect, panel);
            c.drawText(label, left + Palette.dp(3), y + Palette.dp(10), text);
        }

        if (count < 2) return;
        for (int si = series.size() - 1; si >= 0; si--) {
            Series s = series.get(si);
            path.reset();
            for (int age = 0; age < count; age++) {
                float px = right - age * dx;
                float py = bottom - (bottom - top) * Math.min(1f, value(s, age) / max);
                if (age == 0) path.moveTo(px, py);
                else path.lineTo(px, py);
            }
            line.setColor(s.color);
            c.drawPath(path, line);

            path.lineTo(right - (count - 1) * dx, bottom);
            path.lineTo(right, bottom);
            path.close();
            int alpha = si == 0 ? 0x70 : 0x38;
            fill.setShader(new LinearGradient(0, top, 0, bottom,
                    (s.color & 0x00FFFFFF) | (alpha << 24), s.color & 0x00FFFFFF,
                    Shader.TileMode.CLAMP));
            c.drawPath(path, fill);
        }
    }
}
