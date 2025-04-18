package com.example.myapplication

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.myapplication.ui.theme.MyApplicationTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
class MainActivity : ComponentActivity() {

    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val imageView = ImageView(this)
        setContentView(imageView)

        // Load the TFLite model
        interpreter = Interpreter(loadModelFile("palm_detection_full.tflite"))
        Log.d("MODEL_INPUT", interpreter.getInputTensor(0).shape().joinToString())

        // Load and preprocess the image
        val rawBitmap = BitmapFactory.decodeStream(assets.open("hand3.jpg"))
        val resizedBitmap = Bitmap.createScaledBitmap(rawBitmap, 192, 192, true)
        val input = preprocessImage(resizedBitmap)

        // Prepare output buffer
        val output = Array(1) { Array(2016) { FloatArray(18) } }
        val scores = Array(1) { Array(2016) { FloatArray(1) } }

        // Run inference
        interpreter.runForMultipleInputsOutputs(
            arrayOf(input),
            mapOf(
                0 to output,
                1 to scores
            )
        )

        // Decode the bounding box
        val anchors = PalmUtils.generateAnchors()
        Log.d("ANCHORS", "anchors.size = ${anchors.size}")
        Log.d("OUTPUT", "output[0].size = ${output[0].size}")

        val decodedBoxes = PalmUtils.decodeBoxes(
            rawBoxes = output,
            scores = scores,
            anchors = anchors,
            scoreThreshold = 0.5f
        )
        val finalBoxes = PalmUtils.nonMaxSuppressionFast(decodedBoxes, overlapThreshold = 0.3f)
        val finalBitmap = drawBoxesOnBitmap(resizedBitmap, finalBoxes)
        imageView.setImageBitmap(finalBitmap)
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Updated to 192 × 192
    private fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val resized = Bitmap.createScaledBitmap(bitmap, 192, 192, true)
        val input = Array(1) { Array(192) { Array(192) { FloatArray(3) } } }
        for (y in 0 until 192) {
            for (x in 0 until 192) {
                val pixel = resized.getPixel(x, y)
                input[0][y][x][0] = ((pixel shr 16) and 0xFF) / 255.0f
                input[0][y][x][1] = ((pixel shr 8) and 0xFF) / 255.0f
                input[0][y][x][2] = (pixel and 0xFF) / 255.0f
            }
        }
        return input
    }

    private fun drawBoxesOnBitmap(bitmap: Bitmap, boxes: List<BoundingBox>): Bitmap {
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val paint = Paint().apply {
            color = Color.CYAN
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }

        for (box in boxes) {
            Log.d("longnh found Box", "xMin=${box.xMin}, yMin=${box.yMin}, xMax=${box.xMax}, yMax=${box.yMax}, score=${box.score}")
            // Directly use pixel coordinates
            val left = box.xMin
            val top = box.yMin
            val right = box.xMax
            val bottom = box.yMax

            canvas.drawRect(left, top, right, bottom, paint)
            Log.d("longnh DRAW", "Draw rect: ($left, $top, $right, $bottom)")
        }

        return output
    }
}