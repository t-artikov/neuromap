package me.tartikov.neuromap

import android.opengl.GLES20.*

class MapShader {
    private val program = loadProgramFromAssets("map.vs", "map.fs")
    private val positionAttribute = glGetAttribLocation(program, "a_position")
    private val screenSpaceGridRectUniform = glGetUniformLocation(program, "u_screenSpaceGridRect")

    init {
        val gridTexturesUniform = glGetUniformLocation(program, "u_gridTextures")
        val gridTextureSlots = IntArray(2) { it }
        val weights = loadFloatBufferFromAsset("network.data").apply {
            position(NetworkSettings.weightCount0)
        }
        val weightsUniform = glGetUniformLocation(program, "u_networkWeights")

        glUseProgram(program)
        glUniform1iv(gridTexturesUniform, gridTextureSlots.size, gridTextureSlots, 0)
        glUniform4fv(weightsUniform, NetworkSettings.weightCount1 / 4, weights)
        glUseProgram(0)
        checkGlError()
    }

    fun draw(quad: Quad, grid: RenderTarget, cameraState: CameraState) {
        glUseProgram(program)
        grid.bindTextures()
        glUniformRect(screenSpaceGridRectUniform, cameraState.screenSpaceGridRect)
        quad.draw(positionAttribute)
        grid.unbindTextures()
    }
}
