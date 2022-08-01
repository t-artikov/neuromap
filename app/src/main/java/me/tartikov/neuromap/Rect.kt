package me.tartikov.neuromap

data class Rect(val x: Double, val y: Double, val w: Double, val h: Double) {
    companion object {
        val WORLD = Rect(0.0, 0.0, 1.0, 1.0)
    }

    val right get() = x + w
    val bottom get() = y + h
}