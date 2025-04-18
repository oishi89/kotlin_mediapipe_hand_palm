package com.example.myapplication

import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.sqrt

data class Anchor(val xCenter: Float, val yCenter: Float, val width: Float, val height: Float)
data class BoundingBox(val xMin: Float, val yMin: Float, val xMax: Float, val yMax: Float, val score: Float)

object PalmUtils {

    fun generateAnchors(): List<Anchor> {
        data class AnchorParams(
            val numLayers: Int = 4,
            val minScale: Float = 0.1484375f,
            val maxScale: Float = 0.75f,
            val inputWidth: Int = 192,
            val inputHeight: Int = 192,
            val anchorOffsetX: Float = 0.5f,
            val anchorOffsetY: Float = 0.5f,
            val strides: List<Int> = listOf(8, 16, 16, 16),
            val aspectRatios: List<Float> = listOf(1.0f)
        )

        fun calculateScale(minScale: Float, maxScale: Float, strideIndex: Int, numStrides: Int): Float {
            return minScale + (maxScale - minScale) * strideIndex / (numStrides - 1).toFloat()
        }

        val anchors = mutableListOf<Anchor>()
        val params = AnchorParams()

        var layerId = 0
        while (layerId < params.strides.size) {
            val stride = params.strides[layerId]
            val scales = mutableListOf<Float>()
            val aspectRatios = mutableListOf<Float>()

            var lastSameStrideLayer = layerId
            while (
                lastSameStrideLayer < params.strides.size &&
                params.strides[lastSameStrideLayer] == params.strides[layerId]
            ) {
                val scale = calculateScale(params.minScale, params.maxScale, lastSameStrideLayer, params.strides.size)
                for (ar in params.aspectRatios) {
                    aspectRatios.add(ar)
                    scales.add(scale)
                }

                // Extra scale for interpolated anchor
                if (lastSameStrideLayer == params.strides.size - 1) {
                    scales.add(1.0f)
                } else {
                    val nextScale = calculateScale(params.minScale, params.maxScale, lastSameStrideLayer + 1, params.strides.size)
                    scales.add(sqrt(scale * nextScale))
                }
                aspectRatios.add(1.0f)
                lastSameStrideLayer++
            }

            val featureMapHeight = ceil(params.inputHeight / stride.toDouble()).toInt()
            val featureMapWidth = ceil(params.inputWidth / stride.toDouble()).toInt()

            for (y in 0 until featureMapHeight) {
                for (x in 0 until featureMapWidth) {
                    val xCenter = (x + params.anchorOffsetX) / featureMapWidth
                    val yCenter = (y + params.anchorOffsetY) / featureMapHeight

                    for (i in scales.indices) {
                        val ratio = sqrt(aspectRatios[i])
                        val anchorWidth = scales[i] * ratio
                        val anchorHeight = scales[i] / ratio

                        anchors.add(
                            Anchor(
                                xCenter = xCenter,
                                yCenter = yCenter,
                                width = anchorWidth,
                                height = anchorHeight
                            )
                        )
                    }
                }
            }

            layerId = lastSameStrideLayer
        }

        return anchors
    }

    fun decodeBoxes(
        rawBoxes: Array<Array<FloatArray>>,  // [1][2016][18]
        scores: Array<Array<FloatArray>>,    // [1][2016][1]
        anchors: List<Anchor>,
        inputWidth: Int = 192,
        inputHeight: Int = 192,
        scoreThreshold: Float = 0.5f
    ): List<BoundingBox> {
        val decodedBoxes = mutableListOf<BoundingBox>()
        for (i in rawBoxes[0].indices) {
            val rawScore = scores[0][i][0]
            val score = 1.0f / (1.0f + exp(-rawScore)) // sigmoid

            if (score < scoreThreshold) continue

            val box = rawBoxes[0][i]
            val anchor = anchors[i]

            val dx = box[0]
            val dy = box[1]
            val dw = box[2]
            val dh = box[3]

            val cx = dx + anchor.xCenter * inputWidth
            val cy = dy + anchor.yCenter * inputHeight

            val xMin = cx - dw / 2f
            val yMin = cy - dh / 2f
            val xMax = cx + dw / 2f
            val yMax = cy + dh / 2f

            decodedBoxes.add(BoundingBox(xMin, yMin, xMax, yMax, score))
        }
        return decodedBoxes
    }

    fun nonMaxSuppressionFast(
        boxes: List<BoundingBox>,
        overlapThreshold: Float
    ): List<BoundingBox> {
        if (boxes.isEmpty()) return emptyList()

        val picked = mutableListOf<BoundingBox>()

        // Sort boxes by score ascending (we'll pick the last one each loop)
        val sortedBoxes = boxes.sortedBy { it.score }.toMutableList()

        while (sortedBoxes.isNotEmpty()) {
            val last = sortedBoxes.removeAt(sortedBoxes.size - 1)
            picked.add(last)

            val toKeep = mutableListOf<BoundingBox>()

            for (box in sortedBoxes) {
                val xx1 = maxOf(last.xMin, box.xMin)
                val yy1 = maxOf(last.yMin, box.yMin)
                val xx2 = minOf(last.xMax, box.xMax)
                val yy2 = minOf(last.yMax, box.yMax)

                val w = maxOf(0f, xx2 - xx1)
                val h = maxOf(0f, yy2 - yy1)
                val intersection = w * h

                val areaBox = (box.xMax - box.xMin) * (box.yMax - box.yMin)
                val overlap = if (areaBox > 0f) intersection / areaBox else 0f

                if (overlap <= overlapThreshold) {
                    toKeep.add(box)
                }
            }

            sortedBoxes.clear()
            sortedBoxes.addAll(toKeep)
        }

        return picked
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))
}