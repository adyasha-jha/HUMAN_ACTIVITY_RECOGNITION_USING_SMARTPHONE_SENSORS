package com.example.harapp.ui

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import kotlin.math.sqrt

class MainActivity : ComponentActivity(), SensorEventListener {

    // ML model
    private lateinit var classifier: HARClassifier

    // Sensors
    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null

    // Sliding window of accelerometer samples (x, y, z)
    private val windowSize = 32   // smaller window → more frequent updates
    private val window: MutableList<FloatArray> = mutableListOf()

    // How many predictions we've made
    private var predictionCount = 0

    // UI state text
    private var predictionText by mutableStateOf("Move the phone/emulator to start…")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 1) Load TF-Lite model
        classifier = HARClassifier(assets)

        // 2) Set up accelerometer
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        // 3) Compose UI
        setContent {
            Surface(
                modifier = Modifier.fillMaxSize(),
                color = MaterialTheme.colorScheme.background
            ) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(text = predictionText)
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Start listening to accelerometer when app is visible
        accelSensor?.let {
            sensorManager.registerListener(
                this,
                it,
                SensorManager.SENSOR_DELAY_GAME
            )
        }
    }

    override fun onPause() {
        super.onPause()
        // Stop listening to save battery
        sensorManager.unregisterListener(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }

    // Called for every accelerometer reading
    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return
        if (event.sensor.type != Sensor.TYPE_ACCELEROMETER) return

        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]

        // Add sample to sliding window
        window.add(floatArrayOf(x, y, z))

        // When we have enough samples, compute features and predict
        if (window.size >= windowSize) {
            val features = buildFeatureVector(window)
            val label = classifier.predict(features)

            predictionCount += 1

            runOnUiThread {
                predictionText = "Prediction #$predictionCount\nActivity: $label"
            }

            // Clear window for the next prediction
            window.clear()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not used
    }

    /**
     * Build a 561-dim vector:
     *  - first part: raw magnitude time series (scaled)
     *  - next part: simple stats (mean, std, min, max) for x, y, z, mag (scaled)
     *  - rest: zeros (padding)
     */
    private fun buildFeatureVector(window: List<FloatArray>): FloatArray {
        val size = 561
        val features = FloatArray(size) { 0f }

        // Separate x, y, z and magnitude
        val xs = window.map { it[0] }
        val ys = window.map { it[1] }
        val zs = window.map { it[2] }
        val mags = window.map { sqrt(it[0] * it[0] + it[1] * it[1] + it[2] * it[2]) }

        fun mean(values: List<Float>): Float {
            if (values.isEmpty()) return 0f
            var sum = 0f
            for (v in values) sum += v
            return sum / values.size
        }

        fun std(values: List<Float>, m: Float): Float {
            if (values.isEmpty()) return 0f
            var sumSq = 0f
            for (v in values) {
                val d = v - m
                sumSq += d * d
            }
            return sqrt(sumSq / values.size)
        }

        val mx = mean(xs)
        val my = mean(ys)
        val mz = mean(zs)
        val mmag = mean(mags)

        val sx = std(xs, mx)
        val sy = std(ys, my)
        val sz = std(zs, mz)
        val smag = std(mags, mmag)

        val minx = xs.minOrNull() ?: 0f
        val miny = ys.minOrNull() ?: 0f
        val minz = zs.minOrNull() ?: 0f
        val maxx = xs.maxOrNull() ?: 0f
        val maxy = ys.maxOrNull() ?: 0f
        val maxz = zs.maxOrNull() ?: 0f

        val scale = 20f
        var i = 0

        // 1) Put raw magnitude time series into first part of vector
        val tsLen = minOf(windowSize, size)
        for (k in 0 until tsLen) {
            features[i++] = mags[k] / scale
        }

        // 2) Append summary stats after the time series
        if (i + 15 < size) {
            features[i++] = mx / scale
            features[i++] = my / scale
            features[i++] = mz / scale
            features[i++] = mmag / scale

            features[i++] = sx / scale
            features[i++] = sy / scale
            features[i++] = sz / scale
            features[i++] = smag / scale

            features[i++] = minx / scale
            features[i++] = miny / scale
            features[i++] = minz / scale
            features[i++] = maxx / scale
            features[i++] = maxy / scale
            features[i++] = maxz / scale
        }

        // remaining positions stay 0f as padding
        return features
    }
}
