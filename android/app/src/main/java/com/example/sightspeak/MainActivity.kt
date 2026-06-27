package com.example.sightspeak

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.*
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tts: TextToSpeech
    private var ortSession: OrtSession? = null
    private var ortEnv: OrtEnvironment? = null
    private var lastSpokenTime = 0L
    private val speakIntervalMs = 2000L

    private val cocoLabels = listOf(
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
        "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        tts = TextToSpeech(this, this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        initOnnxModel()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.CAMERA), 100)
        }
    }

    private fun initOnnxModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            Log.d("SightSpeak", "ORT env created")
            val modelBytes = assets.open("sightspeak.onnx").readBytes()
            Log.d("SightSpeak", "Model bytes loaded: ${modelBytes.size}")
            ortSession = ortEnv!!.createSession(modelBytes, OrtSession.SessionOptions())
            Log.d("SightSpeak", "Session created: ${ortSession?.outputNames}")
        } catch (e: Exception) {
            Log.e("SightSpeak", "Model load failed: ${e.message}")
            Log.e("SightSpeak", "Stack: ${e.stackTraceToString()}")
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(320, 320))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        runDetection(imageProxy)
                    }
                }
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun runDetection(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxy.toBitmap()
            val resized = android.graphics.Bitmap.createScaledBitmap(bitmap, 320, 320, true)
            val floatArray = FloatArray(1 * 3 * 320 * 320)
            // CHW layout: all R values, then all G, then all B
            var idx = 0
            for (c in 0 until 3) {
                for (y in 0 until 320) {
                    for (x in 0 until 320) {
                        val px = resized.getPixel(x, y)
                        floatArray[idx++] = when (c) {
                            0 -> android.graphics.Color.red(px) / 255f
                            1 -> android.graphics.Color.green(px) / 255f
                            else -> android.graphics.Color.blue(px) / 255f
                        }
                    }
                }
            }
            val inputTensor = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(floatArray),
                longArrayOf(1, 3, 320, 320)
            )
            val results = ortSession?.run(mapOf("images" to inputTensor))

            // Correct extraction: results.get(index) returns Optional<OnnxValue>
            val outputValue = results?.get("output0")
            val outputTensor = if (outputValue != null && outputValue.isPresent) {
                (outputValue.get() as OnnxTensor).value
            } else null

            Log.d("SightSpeak", "Output type: ${outputTensor?.javaClass?.name}")
            processDetections(outputTensor)
            inputTensor.close()
            results?.close()
        } catch (e: Exception) {
            Log.e("SightSpeak", "Detection error: ${e.message}")
            Log.e("SightSpeak", "Stack: ${e.stackTraceToString()}")
        } finally {
            imageProxy.close()
        }
    }

    private fun processDetections(output: Any?) {
        val now = System.currentTimeMillis()
        if (now - lastSpokenTime < speakIntervalMs) return
        output ?: return

        // YOLOv8 ONNX output shape: [1, 84, 2100]
        // 84 = 4 box coords (rows 0-3) + 80 class scores (rows 4-83)
        val batch = output as? Array<*> ?: return
        val data = batch[0] as? Array<*> ?: return

        var bestConf = 0.25f
        var bestLabel = ""
        val numDetections = 2100
        val numClasses = 80

        for (i in 0 until numDetections) {
            var maxScore = 0f
            var maxClassIdx = -1
            for (c in 0 until numClasses) {
                val row = data[4 + c] as? FloatArray ?: continue
                if (row.size <= i) continue
                val score = row[i]
                if (score > maxScore) {
                    maxScore = score
                    maxClassIdx = c
                }
            }
            if (maxScore > bestConf && maxClassIdx >= 0
                && maxClassIdx < cocoLabels.size) {
                bestConf = maxScore
                bestLabel = cocoLabels[maxClassIdx]
            }
        }

        if (bestLabel.isNotEmpty()) {
            Log.d("SightSpeak", "Detected: $bestLabel conf=$bestConf")
            speak(bestLabel)
            lastSpokenTime = now
        }
    }

    private fun speak(text: String) {
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
            speak("SightSpeak ready")
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tts.shutdown()
        ortSession?.close()
        ortEnv?.close()
    }
}