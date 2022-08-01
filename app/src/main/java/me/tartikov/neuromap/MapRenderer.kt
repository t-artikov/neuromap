package me.tartikov.neuromap

import android.opengl.GLES20.*
import android.opengl.GLSurfaceView
import android.util.Size
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.lang.Exception
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MapRenderer(
    mainCoroutineScope: CoroutineScope,
    private val runOnRenderThread: (() -> Unit) -> Unit
) : GLSurfaceView.Renderer {

    private lateinit var grid: RenderTarget
    private lateinit var quad: Quad
    private lateinit var gridShader: GridShader
    private lateinit var mapShader: MapShader
    private var viewportSize = Size(0, 0)

    private val _error = MutableStateFlow<Exception?>(null)
    val error: StateFlow<Exception?> = _error

    private val fpsMeasurer = FpsMeasurer()
    val fps: StateFlow<Int?> = fpsMeasurer.fps

    private var cameraState = CameraState.DEFAULT

    init {
        mainCoroutineScope.launch {
            while (true) {
                delay(2000)
                runOnRenderThread {
                    fpsMeasurer.flush()
                }
            }
        }
    }

    override fun onSurfaceCreated(unused: GL10, config: EGLConfig) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f)
        try {
            grid = RenderTarget(Size(GridSettings.size, GridSettings.size), 2)
            quad = Quad()
            gridShader = GridShader()
            mapShader = MapShader()
        } catch (e: Exception) {
            _error.value = e
        }
    }

    override fun onDrawFrame(unused: GL10) {
        if (_error.value != null) {
            return
        }
        if (viewportSize.width == 0 || viewportSize.height == 0) {
            return
        }

        grid.drawInto {
            gridShader.draw(quad, cameraState)
        }

        glViewport(0, 0, viewportSize.width, viewportSize.height)
        mapShader.draw(quad, grid, cameraState)

        fpsMeasurer.frame()
    }

    override fun onSurfaceChanged(unused: GL10, width: Int, height: Int) {
        viewportSize = Size(width, height)
    }

    fun setCameraState(value: CameraState) {
        cameraState = value
    }
}
