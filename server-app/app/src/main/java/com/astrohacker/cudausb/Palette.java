package com.astrohacker.cudausb;

import android.content.res.Resources;
import android.graphics.Typeface;

/** Colors and helpers shared by the custom views (mirror res/values/colors.xml). */
final class Palette {
    static final int PANEL = 0xFF0F1510;
    static final int PANEL_BORDER = 0xFF1D2A1C;
    static final int GRID = 0xFF1A2419;
    static final int SEGMENT_OFF = 0xFF1C271A;
    static final int GREEN = 0xFF76B900;
    static final int GREEN_BRIGHT = 0xFFA6F000;
    static final int TEAL = 0xFF2FD3A0;
    static final int LIME = 0xFFD4E600;
    static final int ORANGE = 0xFFFF8A3D;
    static final int RED = 0xFFFF4D4D;
    static final int TEXT = 0xFFE3ECDB;
    static final int TEXT_DIM = 0xFF7D8B76;

    static final Typeface CONDENSED = Typeface.create("sans-serif-condensed", Typeface.BOLD);
    static final Typeface MEDIUM = Typeface.create("sans-serif-medium", Typeface.NORMAL);
    static final Typeface MONO = Typeface.MONOSPACE;

    private Palette() {}

    static float dp(float v) {
        return v * Resources.getSystem().getDisplayMetrics().density;
    }

    /** Green while there is headroom, lime near the top, orange when saturated. */
    static int levelColor(float fraction) {
        if (fraction >= 0.95f) return ORANGE;
        if (fraction >= 0.8f) return LIME;
        return GREEN;
    }

    static String format(float v, int decimals) {
        if (v < 0) return "—";
        return decimals == 0 ? String.valueOf(Math.round(v)) : String.format("%." + decimals + "f", v);
    }
}
