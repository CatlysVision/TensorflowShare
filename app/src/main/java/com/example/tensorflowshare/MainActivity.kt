package com.example.tensorflowshare

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.util.TypedValue
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.tensorflowshare.classifier.*
import com.example.tensorflowshare.tools.BorderedText
import com.example.tensorflowshare.tools.ImageUtils
import com.example.tensorflowshare.view.MultiBoxTracker
import com.example.tensorflowshare.view.OverlayView
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private val TF_OD_API_MODEL_FILE = "detect.tflite"
    private val TF_OD_API_LABELS_FILE = "labelmap.txt"

    private val REQUEST_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    private lateinit var textureView: TextureView
    private lateinit var trackingView: OverlayView
    private lateinit var tracker: MultiBoxTracker
    private lateinit var borderedText: BorderedText
    private lateinit var detector: Classifier

    private lateinit var rgbFrameBitmap: Bitmap
    private lateinit var croppedBitmap: Bitmap
    private lateinit var cropCopyBitmap: Bitmap

    private var computingDetection = false
    private var isProcessingFrame = false
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var yRowStride = 0

    private lateinit var frameToCropTransform: Matrix
    private lateinit var cropToFrameTransform: Matrix

    private var previewWidth = 0
    private var previewHeight = 0

    private var rgbBytes: IntArray? = null

    private var postInferenceCallback: Runnable? = null
    private var imageConverter: Runnable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textureView = findViewById(R.id.view_finder)
        if (permissionGranted()) {
            textureView.post { startCamera() }
        } else {
            ActivityCompat.requestPermissions(this, REQUEST_PERMISSIONS, 100)
        }
        textureView.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> updateTransform() }
        initView()
    }

    private lateinit var handler: Handler
    private lateinit var handlerThread: HandlerThread

    override fun onResume() {
        super.onResume()
        handlerThread = HandlerThread("inference")
        handlerThread.start()
        handler = Handler(handlerThread.looper)
    }

    override fun onPause() {
        handlerThread.quitSafely()
        super.onPause()
    }

    private fun initView() {
        val textSizePx = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            10f,
            resources.displayMetrics
        )
        borderedText = BorderedText(textSizePx)
        borderedText.setTypeface(Typeface.MONOSPACE)

        tracker = MultiBoxTracker(this)

        detector = TFLiteObjectDetection.create(
            assets, TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE,
            300, true
        )

        val cropSize = 300
        previewWidth = 640
        previewHeight = 480

        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)

        frameToCropTransform = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            90, false
        )

        cropToFrameTransform = Matrix()
        frameToCropTransform.invert(cropToFrameTransform)

        trackingView = findViewById(R.id.tracking_view)
        trackingView.setDrawCallbac(object : OverlayView.DrawCallback {

            override fun onDraw(canvas: Canvas) {
                tracker.draw(canvas)
            }

        })
        tracker.setFrameConfiguration(previewWidth, previewHeight, 90)
    }

    private fun permissionGranted() = REQUEST_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == 100) {
            if (permissionGranted()) {
                textureView.post { startCamera() }
            }
        }
    }

    private fun startCamera() {
        val previewConfig = PreviewConfig.Builder().apply {
            setTargetAspectRatio(AspectRatio.RATIO_4_3)
            setTargetRotation(Surface.ROTATION_90)
            //setTargetResolution(Size(640, 480))
        }.build()
        val preview = Preview(previewConfig)
        preview.setOnPreviewOutputUpdateListener {
            val parent = textureView.parent as ViewGroup
            parent.removeView(textureView)
            parent.addView(textureView, 0)

            textureView.surfaceTexture = it.surfaceTexture
            updateTransform()
        }

        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            setTargetRotation(Surface.ROTATION_90)
            setTargetResolution(Size(640, 480))
        }.build()
        val imageAnalyzer = ImageAnalysis(analyzerConfig).apply {
            setAnalyzer(Executors.newSingleThreadExecutor(), TensorFlowImageAnalyzer())
        }

        CameraX.bindToLifecycle(this, preview, imageAnalyzer)
    }

    private fun updateTransform() {
        val matrix = Matrix()
        val centerX = textureView.width / 2f
        val centerY = textureView.height / 2f
        val rotationDegrees = when (textureView.display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return
        }
        matrix.postRotate(-rotationDegrees.toFloat(), centerX, centerY)
        textureView.setTransform(matrix)
    }

    private var count = 0

    private fun processImage() {
        trackingView.postInvalidate()

        if (computingDetection) {
            readyForNextImage()
            return
        }
        computingDetection = true
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)
        readyForNextImage()

        val canvas = Canvas(croppedBitmap)
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null)

        runInBackground(Runnable {
            val results = detector.recognizeImage(croppedBitmap)
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap)
            val canvas1 = Canvas(cropCopyBitmap)
            val paint = Paint()
            paint.color = Color.RED
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f

            val minimumConfidence = 0.5f

            val mappedRecognitions = mutableListOf<Recognition>()

            results.forEach {
                val location = it.location
                if (it.confidence >= minimumConfidence) {
                    canvas1.drawRect(location, paint)
                    cropToFrameTransform.mapRect(location)
                    it.location = location
                    mappedRecognitions.add(it)
                }
            }
            tracker.trackResults(mappedRecognitions)
            trackingView.postInvalidate()

            computingDetection = false
        })

    }

    private fun getRgbBytes(): IntArray {
        imageConverter?.run()
        return rgbBytes!!
    }

    private fun readyForNextImage() {
        postInferenceCallback?.run()
    }

    @Synchronized
    private fun runInBackground(runnable: Runnable) {
        handler.post(runnable)
    }

    inner class TensorFlowImageAnalyzer : ImageAnalysis.Analyzer {

        override fun analyze(image: ImageProxy?, rotationDegrees: Int) {
            if (rgbBytes == null) {
                rgbBytes = IntArray(previewWidth * previewHeight)
            }

            try {
                if (image == null) {
                    return
                }
                if (isProcessingFrame) {
                    return
                }
                isProcessingFrame = true

                val planes = image.planes
                fillBytes(planes, yuvBytes)
                yRowStride = planes[0].rowStride
                val uvRowStride = planes[1].rowStride
                val uvPixelStride = planes[1].pixelStride

                imageConverter = Runnable {
                    ImageUtils.convertYUV420ToARGB8888(
                        yuvBytes[0],
                        yuvBytes[1],
                        yuvBytes[2],
                        previewWidth,
                        previewHeight,
                        yRowStride,
                        uvRowStride,
                        uvPixelStride,
                        rgbBytes
                    )
                }

                postInferenceCallback = Runnable {
                    isProcessingFrame = false
                }

                processImage()

            } catch (e: Exception) {

            }
        }

    }

    private fun fillBytes(
        planes: Array<ImageProxy.PlaneProxy>,
        yuvBytes: Array<ByteArray?>
    ) {
        for (i in planes.indices) {
            val byteBuffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                yuvBytes[i] = ByteArray(byteBuffer.capacity())
            }
            byteBuffer.get(yuvBytes[i])
        }
    }
}
