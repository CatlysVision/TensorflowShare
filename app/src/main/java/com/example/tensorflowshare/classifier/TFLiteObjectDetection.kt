package com.example.tensorflowshare.classifier

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import java.util.logging.Logger

class TFLiteObjectDetection private constructor() : Classifier {

    private val labels = mutableListOf<String>()

    private var inputSize = 0

    private var isModelQuantized = false

    private lateinit var intValues: IntArray
    private lateinit var outputLocations: Array<Array<FloatArray>>
    private lateinit var outputClasses: Array<FloatArray>
    private lateinit var outputScores: Array<FloatArray>
    private lateinit var numDetections: FloatArray

    private lateinit var imgData: ByteBuffer
    private lateinit var tfLite: Interpreter

    override fun recognizeImage(bitmap: Bitmap): List<Recognition> {
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(
            intValues,
            0,
            bitmap.width,
            0,
            0,
            bitmap.width,
            bitmap.height
        )
        imgData.rewind()

        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[i * inputSize + j]
                if (isModelQuantized) { // Quantized model
                    imgData.put((pixelValue shr 16 and 0xFF).toByte())
                    imgData.put((pixelValue shr 8 and 0xFF).toByte())
                    imgData.put((pixelValue and 0xFF).toByte())
                } else { // Float model
                    imgData.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }
        // Copy the input data into TensorFlow.
        outputLocations = Array(1) { Array(NUM_DETECTIONS) { FloatArray(4) } }
        outputClasses = Array(1) { FloatArray(NUM_DETECTIONS) }
        outputScores = Array(1) { FloatArray(NUM_DETECTIONS) }
        numDetections = FloatArray(1)

        val inputArray = arrayOf<Any?>(imgData)
        val outputMap = mutableMapOf<Int, Any>()
        outputMap[0] = outputLocations
        outputMap[1] = outputClasses
        outputMap[2] = outputScores
        outputMap[3] = numDetections
        // Run the inference call.
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap)
        // Show the best detections.
// after scaling them back to the input size.
        val recognitions =
            ArrayList<Recognition>(NUM_DETECTIONS)
        for (i in 0 until NUM_DETECTIONS) {
            val detection = RectF(
                outputLocations[0][i][1] * inputSize,
                outputLocations[0][i][0] * inputSize,
                outputLocations[0][i][3] * inputSize,
                outputLocations[0][i][2] * inputSize
            )
            // SSD Mobilenet V1 Model assumes class 0 is background class
// in label file and class labels start from 1 to number_of_classes+1,
// while outputClasses correspond to class index from 0 to number_of_classes
            val labelOffset = 1
            val recognition = Recognition(
                i.toString(),
                labels[outputClasses[0][i].toInt() + labelOffset],
                outputScores[0][i],
                detection
            )
            recognitions.add(recognition)
        }
        return recognitions
    }

    fun recognizeImage1(bitmap: Bitmap): List<Recognition> {

        //bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        bitmap.getPixels(
            intValues,
            0,
            bitmap.width,
            0,
            0,
            bitmap.width,
            bitmap.height
        )
        imgData.rewind()

        /*for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[i * inputSize + j]
                if (isModelQuantized) { // Quantized model
                    imgData.put((pixelValue shr 16 and 0xFF).toByte())
                    imgData.put((pixelValue shr 8 and 0xFF).toByte())
                    imgData.put((pixelValue and 0xFF).toByte())
                } else { // Float model
                    imgData.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }*/

        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[i * inputSize + j]
                if (isModelQuantized) { // Quantized model
                    imgData.put((pixelValue shr 16 and 0xFF).toByte())
                    imgData.put((pixelValue shr 8 and 0xFF).toByte())
                    imgData.put((pixelValue and 0xFF).toByte())
                } else { // Float model
                    imgData.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }

        outputLocations = Array(1) { Array(NUM_DETECTIONS) { FloatArray(4) } }
        outputClasses = Array(1) { FloatArray(NUM_DETECTIONS) }
        outputScores = Array(1) { FloatArray(NUM_DETECTIONS) }
        numDetections = FloatArray(1)

        val inputArray = arrayOf<Any>(imgData)
        val outputMap = mutableMapOf<Int, Any>()
        outputMap[0] = outputLocations
        outputMap[1] = outputClasses
        outputMap[2] = outputScores
        outputMap[3] = numDetections

        tfLite.runForMultipleInputsOutputs(inputArray, outputMap)

        val recognitions =
            ArrayList<Recognition>(NUM_DETECTIONS)
        for (i in 0 until NUM_DETECTIONS) {

            val detection = RectF(
                outputLocations[0][i][1] * inputSize,
                outputLocations[0][i][0] * inputSize,
                outputLocations[0][i][3] * inputSize,
                outputLocations[0][i][2] * inputSize
            )

            val labelOffset = 1
            val recognition = Recognition(
                i.toString(),
                labels[outputClasses[0][i].toInt() + labelOffset],
                outputScores[0][i],
                detection
            )
            Log.d("ymc_test", "recognition=$recognition")
            recognitions.add(recognition)
        }

        return recognitions
    }

    companion object {

        private const val NUM_DETECTIONS = 10
        private const val IMAGE_MEAN = 128.0f
        private const val IMAGE_STD = 128.0f
        private const val NUM_THREADS = 4

        private fun loadModelFile(
            assets: AssetManager,
            modelFilename: String
        ): MappedByteBuffer {
            val fileDescriptor = assets.openFd(modelFilename)
            val inputStream =
                FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                startOffset,
                declaredLength
            )
        }

        public fun create(
            assetManager: AssetManager,
            modelFileName: String,
            labelFileName: String,
            inputSize: Int,
            isQuantized: Boolean
        ): Classifier {
            val model = TFLiteObjectDetection()
            val labelsInput = assetManager.open(labelFileName)
            labelsInput.bufferedReader().useLines { lines ->
                lines.forEach {
                    model.labels.add(it)
                }
            }
            model.inputSize = inputSize

            val delegate = GpuDelegate()
            val options = Interpreter.Options().addDelegate(delegate)
            model.tfLite = Interpreter(loadModelFile(assetManager, modelFileName), options)

            model.isModelQuantized = isQuantized
            val numBytesPerChannel = if (isQuantized) {
                1
            } else {
                4
            }
            model.imgData =
                ByteBuffer.allocateDirect(1 * model.inputSize * model.inputSize * 3 * numBytesPerChannel)
            model.imgData.order(ByteOrder.nativeOrder())
            model.intValues = IntArray(model.inputSize * model.inputSize)
            model.tfLite.setNumThreads(NUM_THREADS)
            model.outputLocations = Array(1) { Array(NUM_DETECTIONS) { FloatArray(4) } }
            model.outputClasses = Array(1) { FloatArray(NUM_DETECTIONS) }
            model.outputScores = Array(1) { FloatArray(NUM_DETECTIONS) }
            model.numDetections = FloatArray(1)
            return model
        }

    }

    override fun enableStatLogging(debug: Boolean) {

    }

    override fun getStatString(): String {
        return ""
    }

    override fun setNumThreads(num_threads: Int) {
        tfLite.setNumThreads(num_threads)
    }

    override fun setUseNNAPI(isChecked: Boolean) {
        tfLite.setUseNNAPI(isChecked)
    }

    override fun close() {

    }

}