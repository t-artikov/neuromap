package me.tartikov.neuromap

import android.annotation.SuppressLint
import android.content.Context
import android.opengl.GLSurfaceView
import android.text.method.ScrollingMovementMethod
import android.util.AttributeSet
import android.util.Log
import android.view.MotionEvent
import android.widget.FrameLayout
import android.widget.TextView
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class MapView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : FrameLayout(context, attrs, defStyle) {

    private val coroutineScope = MainScope()
    private val glView: GLSurfaceView
    private val fpsView: TextView
    private val errorView: TextView
    private val renderer: MapRenderer
    private val cameraController: CameraController = CameraController(this)

    init {
        inflate(context, R.layout.map_view, this)

        glView = findViewById<GLSurfaceView?>(R.id.glView).apply {
            setEGLContextClientVersion(3)
            setEGLConfigChooser(8 , 8, 8, 0, 0, 0)
        }
        fpsView = findViewById(R.id.fpsView)
        errorView = findViewById<TextView>(R.id.errorView).apply {
            movementMethod = ScrollingMovementMethod()
        }

        renderer = MapRenderer(coroutineScope, ::runOnRenderThread).also {
            glView.setRenderer(it)
        }

        showErrors()
        showFps()
        initCameraController()
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        return cameraController.onTouchEvent(event) || super.onTouchEvent(event)
    }

    private fun showErrors() {
        coroutineScope.launch {
            renderer.error.collect {
                if (it != null) {
                    Log.w(LOG_TAG, "Rendering error", it)
                    errorView.text = it.stackTraceToString()
                    errorView.visibility = VISIBLE
                }
                errorView.visibility = if (it != null) VISIBLE else GONE
            }
        }
    }

    private fun showFps() {
        coroutineScope.launch {
            renderer.fps.collect {
                fpsView.text = it?.toString() ?: ""
            }
        }
    }

    private fun initCameraController() {
        coroutineScope.launch {
            cameraController.cameraState.collect {
                runOnRenderThread {
                    renderer.setCameraState(it)
                }
            }
        }
    }

    private fun runOnRenderThread(action: () -> Unit) {
        glView.queueEvent(action)
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        coroutineScope.cancel()
    }
}
