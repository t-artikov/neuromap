package me.tartikov.neuromap

import android.opengl.GLES20.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer

class Quad {
    private val vertexBuffer: FloatBuffer
    private val indexBuffer: ShortBuffer

    fun draw(positionAttribute: Int) {
        glEnableVertexAttribArray(positionAttribute)
        glVertexAttribPointer(
            positionAttribute, COORD_COUNT,
            GL_FLOAT, false,
            COORD_COUNT * Float.SIZE_BYTES, vertexBuffer
        )
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indexBuffer)
        glDisableVertexAttribArray(positionAttribute)
    }

    init {
        val vertices = floatArrayOf(
            0.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f,
            1.0f, 1.0f
        )
        vertexBuffer = ByteBuffer.allocateDirect(
            vertices.size * Float.SIZE_BYTES
        ).apply {
            order(ByteOrder.nativeOrder())
        }.asFloatBuffer().apply {
            put(vertices)
            position(0)
        }

        val indexes = shortArrayOf(
            0, 1, 2,
            0, 2, 3
        )
        indexBuffer = ByteBuffer.allocateDirect(
            indexes.size * Short.SIZE_BYTES
        ).apply {
            order(ByteOrder.nativeOrder())
        }.asShortBuffer().apply {
            put(indexes)
            position(0)
        }
    }

    companion object {
        private const val COORD_COUNT = 2
    }
}