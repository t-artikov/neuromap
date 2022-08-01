package me.tartikov.neuromap

import android.util.Size
import kotlin.math.ceil
import kotlin.math.min

data class CameraState(
    val rect: Rect,
    val gridRect: Rect,
    val screenSpaceGridRect: Rect,
    val level: Int,
    val levelOpacity: Float
) {
    val activeGridSize: Size
        get() {
            val w = min(ceil(screenSpaceGridRect.right * GridSettings.size).toInt() + 1, GridSettings.size)
            val h = min(ceil(screenSpaceGridRect.bottom * GridSettings.size).toInt() + 1, GridSettings.size)
            return Size(w, h)
        }

    companion object {
        val DEFAULT = CameraState(Rect.WORLD, Rect.WORLD, Rect(0.0, 0.0, 1.0, 1.0), 0, 1.0f)
    }
}