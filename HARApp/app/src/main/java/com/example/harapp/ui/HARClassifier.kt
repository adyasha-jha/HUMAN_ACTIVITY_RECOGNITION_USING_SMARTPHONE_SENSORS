package com.example.harapp.ui

import android.content.res.AssetManager
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class HARClassifier(private val assetManager: AssetManager) {

    companion object {
        private const val TAG = "HARClassifier"
        private const val MODEL_NAME = "har_model.tflite"
        private const val TIMESTEPS = 561
        private const val FEATURES = 1
        private const val NUM_CLASSES = 6
    }

    private val interpreter: Interpreter

    private val activityLabels = arrayOf(
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    )

    init {
        val modelBuffer = loadModelFile(assetManager, MODEL_NAME)
        interpreter = Interpreter(modelBuffer)
    }

    /**
     * Load the .tflite model from assets into a direct ByteBuffer.
     * This works even if the asset is stored compressed.
     */
    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, modelPath: String): ByteBuffer {
        val inputStream = assetManager.open(modelPath, AssetManager.ACCESS_BUFFER)
        val bytes = inputStream.readBytes()
        inputStream.close()

        val byteBuffer = ByteBuffer.allocateDirect(bytes.size)
        byteBuffer.order(ByteOrder.nativeOrder())
        byteBuffer.put(bytes)
        byteBuffer.rewind()
        return byteBuffer
    }

    // input561 must be length 561 and normalized
    fun predict(input561: FloatArray): String {
        if (input561.size != TIMESTEPS) {
            Log.e(TAG, "Input length must be 561, got: ${input561.size}")
            return "INVALID_INPUT"
        }

        // Shape: [1, 561, 1]
        val input = Array(1) { Array(TIMESTEPS) { FloatArray(FEATURES) } }
        for (i in 0 until TIMESTEPS) {
            input[0][i][0] = input561[i]
        }

        val output = Array(1) { FloatArray(NUM_CLASSES) }
        interpreter.run(input, output)

        // Argmax
        var bestIndex = 0
        var bestProb = output[0][0]
        for (i in 1 until NUM_CLASSES) {
            if (output[0][i] > bestProb) {
                bestProb = output[0][i]
                bestIndex = i
            }
        }

        return activityLabels[bestIndex]
    }

    fun close() {
        interpreter.close()
    }
}
