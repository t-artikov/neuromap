package me.tartikov.neuromap

import kotlin.math.min

object EncodingSettings {
    const val levelCount = 12
    const val featurePerLevel = 4
    private const val baseResolution = 32
    private const val hashSize = 1 shl 22

    val resolutions: IntArray = run {
        var resolution = baseResolution
        IntArray(levelCount) {
            resolution.also {
                resolution = (resolution - 1) * 2 + 1
            }
        }
    }

    val paramCounts: IntArray = resolutions.map {
        if (it.toLong() * it.toLong() > Int.MAX_VALUE) {
            return@map hashSize
        }
        min(nextMultiple(it * it, 8), hashSize)
    }.toIntArray()

    val paramOffsets: IntArray = run {
        var offset = 0
        IntArray(levelCount) { index ->
            offset.also {
                offset += paramCounts[index]
            }
        }
    }
}

private fun divRoundUp(value: Int, divisor: Int): Int {
    return (value + divisor - 1) / divisor
}

@Suppress("SameParameterValue")
private fun nextMultiple(value: Int, divisor: Int): Int {
    return divRoundUp(value, divisor) * divisor
}
