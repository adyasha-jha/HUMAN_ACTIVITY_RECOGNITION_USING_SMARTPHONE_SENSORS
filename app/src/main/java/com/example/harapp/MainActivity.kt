package com.example.harapp

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.util.Log
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
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

// ------------------------------------------------------------
//                    HAR CLASSIFIER
// ------------------------------------------------------------
class HARClassifier(
    private val context: Context,
    private val fileName: String = "har_model.tflite"
) {

    private var interpreter: Interpreter? = null
    var lastError: String? = null
        private set

    private val windowSize = 128
    private val numChannels = 3
    private val numClasses = 6

    // === NORMALIZATION CONSTANTS FROM PYTHON SCRIPT ===
    private val mean = floatArrayOf(
        0.808330f,
        0.021772f,
        0.084831f
    )

    private val std = floatArrayOf(
        0.411991f,
        0.397284f,
        0.344254f
    )
    // ==================================================

    init {
        try {
            val options = Interpreter.Options().apply {
                // Being safe, some devices behave better with XNNPACK off
                setUseXNNPACK(false)
            }
            interpreter = Interpreter(loadModelFileFromAssets(), options)
            Log.d("HARClassifier", "TFLite model loaded successfully")
        } catch (e: Exception) {
            lastError = e.toString()
            Log.e("HARClassifier", "Error loading TFLite model", e)
            interpreter = null
        }
    }

    fun isReady(): Boolean = interpreter != null

    /**
     * Load model from assets into a direct ByteBuffer.
     * Works even if the asset is compressed.
     */
    private fun loadModelFileFromAssets(): ByteBuffer {
        val inputStream = context.assets.open(fileName)
        val bytes = inputStream.readBytes()
        inputStream.close()

        val buffer = ByteBuffer.allocateDirect(bytes.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        buffer.rewind()
        return buffer
    }

    /**
     * window: list of 128 float[3] (x,y,z) samples
     * returns class index 0..5 or -1 on error.
     */
    fun predict(window: List<FloatArray>): Int {
        val interp = interpreter ?: return -1

        if (window.size != windowSize) {
            lastError = "Bad window size: ${window.size}, expected $windowSize"
            Log.e("HARClassifier", lastError!!)
            return -1
        }

        val input = Array(1) { Array(windowSize) { FloatArray(numChannels) } }
        val output = Array(1) { FloatArray(numClasses) }

        // normalize and copy into input tensor
        for (i in 0 until windowSize) {
            val sample = window[i]
            for (c in 0 until numChannels) {
                val raw = sample.getOrNull(c) ?: 0f
                input[0][i][c] = (raw - mean[c]) / std[c]
            }
        }

        return try {
            interp.run(input, output)

            // argmax
            var bestIdx = 0
            var bestProb = output[0][0]
            for (i in 1 until numClasses) {
                if (output[0][i] > bestProb) {
                    bestProb = output[0][i]
                    bestIdx = i
                }
            }

            Log.d("HARClassifier", "probs=${output[0].contentToString()}, pred=$bestIdx")
            bestIdx
        } catch (e: Exception) {
            lastError = e.toString()
            Log.e("HARClassifier", "Error during inference", e)
            -1
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

// ------------------------------------------------------------
//                       MAIN ACTIVITY
// ------------------------------------------------------------
class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null

    private val windowSize = 128
    private val window: MutableList<FloatArray> = mutableListOf()

    private var classifier: HARClassifier? = null
    private var currentActivity by mutableStateOf("Loading model...")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // sensors
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        // classifier
        classifier = HARClassifier(this)

        currentActivity = if (classifier?.isReady() == true) {
            "Move the phone to start..."
        } else {
            "Model load failed:\n${classifier?.lastError ?: "unknown error"}"
        }

        // UI
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = currentActivity,
                            style = MaterialTheme.typography.headlineMedium
                        )
                    }
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        accelSensor?.also { sensor ->
            sensorManager.registerListener(
                this,
                sensor,
                SensorManager.SENSOR_DELAY_GAME
            )
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // not used
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ACCELEROMETER) return

        val clf = classifier
        if (clf == null || !clf.isReady()) {
            currentActivity = "Model NOT loaded:\n${clf?.lastError ?: "unknown error"}"
            return
        }

        // collect accelerometer sample
        window.add(
            floatArrayOf(
                event.values[0],
                event.values[1],
                event.values[2]
            )
        )

        if (window.size >= windowSize) {
            val recentWindow = window.takeLast(windowSize)

            val prediction = clf.predict(recentWindow)

            val label = when (prediction) {
                0 -> "WALKING"
                1 -> "WALKING UPSTAIRS"
                2 -> "WALKING DOWNSTAIRS"
                3 -> "SITTING"
                4 -> "STANDING"
                5 -> "LAYING"
                else -> "UNKNOWN\n${clf.lastError ?: ""}"
            }

            currentActivity = label

            // keep 50% overlap
            while (window.size > windowSize / 2) {
                window.removeAt(0)
            }
        }
    }
}
