package me.tartikov.neuromap

import android.content.Context
import android.graphics.Matrix
import android.util.DisplayMetrics
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import com.otaliastudios.zoom.ZoomEngine
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.lang.RuntimeException
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

class CameraController(view: View) {
    private val zoomEngine = ZoomEngine(view.context, view)

    private val _cameraState = MutableStateFlow(CameraState.DEFAULT)
    val cameraState: StateFlow<CameraState> = _cameraState

    init {
        val size = getScreenSize(view.context)
        zoomEngine.apply {
            setOverScrollHorizontal(false)
            setOverScrollVertical(false)
            setOverPinchable(false)
            setContentSize(size, size)
            setMaxZoom(20000.0f)
            setMinZoom(2.0f)
            setTwoFingersScrollEnabled(false)
            addListener(object : ZoomEngine.Listener {
                override fun onIdle(engine: ZoomEngine) {

                }

                override fun onUpdate(engine: ZoomEngine, matrix: Matrix) {
                    val rect = getCameraRect(engine)
                    _cameraState.value = calculateCameraState(rect)
                }
            })
        }
    }

    fun onTouchEvent(event: MotionEvent): Boolean {
        return zoomEngine.onTouchEvent(event)
    }
}

private fun getCameraRect(engine: ZoomEngine): Rect {
    val w = engine.containerWidth.toDouble() / engine.contentWidth / engine.realZoom
    val h = engine.containerHeight.toDouble() / engine.contentHeight / engine.realZoom
    val x = -engine.scaledPanX.toDouble() / engine.containerWidth * w
    val y = -engine.scaledPanY.toDouble() / engine.containerHeight * h
    return Rect(x, y, w, h)
}

@Suppress("DEPRECATION")
private fun getScreenSize(context: Context): Float {
    val metrics = DisplayMetrics()
    val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    windowManager.defaultDisplay.getMetrics(metrics)
    return min(metrics.widthPixels, metrics.heightPixels).toFloat()
}

private fun getRelativeRect(rect: Rect, basis: Rect): Rect {
    val x = (rect.x - basis.x) / basis.w
    val y = (rect.y - basis.y) / basis.h
    val w = rect.w / basis.w
    val h = rect.h / basis.h
    return Rect(x, y, w, h)
}

private fun calculateCameraState(cameraRect: Rect): CameraState {
    val lowestLevel = 4
    for (level in EncodingSettings.levelCount - 1 downTo lowestLevel) {
        val resolution = EncodingSettings.resolutions[level]
        val tileSize = 1.0 / (resolution - 1)

        fun snapToTile(v: Double): Double {
            val iv = floor(v * (resolution - 1))
            return max(0.0, iv * tileSize)
        }

        val x = snapToTile(cameraRect.x)
        val y = snapToTile(cameraRect.y)
        val w = (GridSettings.size - 1) * tileSize
        val h = (GridSettings.size - 1) * tileSize

        if (x + w > cameraRect.right && y + h > cameraRect.bottom || level == lowestLevel) {
            val snappedRect =
                Rect(x - tileSize / 2.0, y - tileSize / 2.0, w + tileSize, h + tileSize)
            val opacity = if (level != lowestLevel) {
                (min(w / cameraRect.w, h / cameraRect.h) - 1.0).coerceIn(0.0, 1.0)
            } else {
                1.0
            }
            return CameraState(
                cameraRect,
                snappedRect,
                getRelativeRect(cameraRect, snappedRect),
                level,
                opacity.toFloat()
            )
        }
    }
    throw RuntimeException("Unreachable")
}
