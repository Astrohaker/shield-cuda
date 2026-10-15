package com.astrohacker.nstcuda;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int MAX_COUNT = 50_000_000;

    static {
        System.loadLibrary("nstcuda");
    }

    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    /** Computes sqrt of 0..n-1 on the GPU and returns a timing report (or an error message). */
    private native String runSqrt(AssetManager assets, int n);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        EditText count = findViewById(R.id.count);
        Button run = findViewById(R.id.run);
        TextView result = findViewById(R.id.result);

        run.setOnClickListener(v -> {
            int n;
            try {
                n = Integer.parseInt(count.getText().toString().trim());
            } catch (NumberFormatException e) {
                n = 0;
            }
            if (n < 1 || n > MAX_COUNT) {
                result.setText(R.string.bad_count);
                return;
            }
            getSystemService(InputMethodManager.class).hideSoftInputFromWindow(count.getWindowToken(), 0);
            run.setEnabled(false);
            result.setText(R.string.running);
            int size = n;
            executor.execute(() -> {
                String report = runSqrt(getAssets(), size);
                runOnUiThread(() -> {
                    result.setText(report);
                    run.setEnabled(true);
                });
            });
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdown();
    }
}
