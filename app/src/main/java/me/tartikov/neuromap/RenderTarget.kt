package me.tartikov.neuromap

import android.opengl.GLES20.*
import android.opengl.GLES30
import android.opengl.GLES30.glDrawBuffers
import android.util.Size


class RenderTarget(private val size: Size, textureCount: Int = 1) {
    private val fbo: Int = run {
        val result = IntArray(1)
        glGenFramebuffers(1, result, 0)
        result[0]
    }
    private val textures = IntArray(textureCount)

    init {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glGenTextures(textureCount, textures, 0)
        textures.forEachIndexed { index, it ->
            glBindTexture(GL_TEXTURE_2D, it)
            glTexImage2D(GL_TEXTURE_2D, 0, GLES30.GL_RGBA32F, size.width, size.height, 0, GL_RGBA, GL_FLOAT, null)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glBindTexture(GL_TEXTURE_2D, 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + index, GL_TEXTURE_2D, it, 0)
        }

        val attachments = IntArray(textureCount) {
            GL_COLOR_ATTACHMENT0 + it
        }
        glDrawBuffers(textureCount, attachments, 0)

        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw RuntimeException("Failed to create RenderTarget")
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        checkGlError()
    }

    fun drawInto(action: () -> Unit) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glViewport(0, 0, size.width, size.height)
        action()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        checkGlError()
    }

    fun bindTextures(slot: Int = 0) {
        textures.forEachIndexed { index, it ->
            glActiveTexture(GL_TEXTURE0 + slot + index)
            glBindTexture(GL_TEXTURE_2D, it)
        }
        glActiveTexture(GL_TEXTURE0)
        checkGlError()
    }

    fun unbindTextures(slot: Int = 0) {
        repeat(textures.size) {
            glActiveTexture(GL_TEXTURE0 + slot + it)
            glBindTexture(GL_TEXTURE_2D, 0)

        }
        glActiveTexture(GL_TEXTURE0)
        checkGlError()
    }
}
