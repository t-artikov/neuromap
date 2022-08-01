package me.tartikov.neuromap

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlin.math.roundToInt

class FpsMeasurer {
    private val _fps = MutableStateFlow<Int?>(null)
    val fps: StateFlow<Int?> = _fps

    private var startTime = System.currentTimeMillis()
    private var frames: Long = 0

    fun frame() {
        frames++
    }

    fun flush() {
        val time = System.currentTimeMillis()
        _fps.value = ((frames * 1000.0) / (time - startTime)).roundToInt()
        startTime = System.currentTimeMillis()
        frames = 0
    }
}