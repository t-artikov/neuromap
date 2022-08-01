package me.tartikov.neuromap

import android.opengl.GLES20.*
import android.opengl.GLU.gluErrorString

fun checkGlError() {
    val error = glGetError()
    if (error != GL_NO_ERROR) {
        throw RuntimeException("OpenGL error: ${gluErrorString(error)}")
    }
}

private fun loadShaderFromAsset(type: Int, fileName: String): Int {
    val shader = glCreateShader(type)
    glShaderSource(shader, loadStringFromAsset(fileName))
    glCompileShader(shader)

    val status = IntArray(1)
    glGetShaderiv(shader, GL_COMPILE_STATUS, status, 0)
    if (status[0] != GL_TRUE) {
        throw RuntimeException("Failed to compile '$fileName':\n${glGetShaderInfoLog(shader)}")
    }

    checkGlError()
    return shader
}

fun loadProgramFromAssets(vsName: String, fsName: String): Int {
    val vs = loadShaderFromAsset(GL_VERTEX_SHADER, vsName)
    val fs = loadShaderFromAsset(GL_FRAGMENT_SHADER,fsName)

    val program = glCreateProgram().also {
        glAttachShader(it, vs)
        glAttachShader(it, fs)
        glLinkProgram(it)
    }

    val status = IntArray(1)
    glGetProgramiv(program, GL_LINK_STATUS, status, 0)
    if (status[0] != GL_TRUE) {
        throw RuntimeException("Failed to link '$vsName', '$fsName':\n${glGetProgramInfoLog(program)}")
    }

    checkGlError()
    return program
}

fun getGlInt(name: Int): Int {
    val values = IntArray(1)
    glGetIntegerv(name, values, 0)
    checkGlError()
    return values[0]
}

fun glUniformRect(location: Int, rect: Rect) = rect.apply {
    glUniform4f(location, x.toFloat(), y.toFloat(), w.toFloat(), h.toFloat())
}