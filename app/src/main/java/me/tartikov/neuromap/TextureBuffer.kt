package me.tartikov.neuromap

import android.opengl.GLES32.*
import java.nio.ShortBuffer

class TextureBuffer(data: ShortBuffer) {
    private val texture: Int

    init {
        check(getGlInt(GL_MAX_TEXTURE_BUFFER_SIZE) > 0) {
            "Texture buffer is not supported"
        }

        val buffer = run {
            val buffers = IntArray(1)
            glGenBuffers(1, buffers, 0)
            checkGlError()
            buffers[0]
        }

        glBindBuffer(GL_TEXTURE_BUFFER, buffer)
        glBufferData(GL_TEXTURE_BUFFER, data.capacity() * Short.SIZE_BYTES, data, GL_STATIC_DRAW)
        glBindBuffer(GL_TEXTURE_BUFFER, 0)
        checkGlError()

        texture = run {
            val textures = IntArray(1)
            glGenTextures(1, textures, 0)
            checkGlError()
            textures[0]
        }

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, texture)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA16F, buffer)
        glBindTexture(GL_TEXTURE_BUFFER, 0)
        checkGlError()
    }

    fun bind(slot: Int = 0) {
        glActiveTexture(GL_TEXTURE0 + slot)
        glBindTexture(GL_TEXTURE_BUFFER, texture)
    }
}
