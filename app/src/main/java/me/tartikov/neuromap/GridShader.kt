package me.tartikov.neuromap

import android.opengl.GLES20.*

class GridShader {
    private val program = loadProgramFromAssets("grid.vs", "grid.fs")
    private val positionAttribute = glGetAttribLocation(program, "a_position")
    private val encodingTexture = TextureBuffer(loadShortBufferFromAsset("encoding.data").also {
        val expectedSize = EncodingSettings.paramCounts.sum() * EncodingSettings.featurePerLevel
        check(it.capacity() == expectedSize) {
            "Invalid encoding data size. Actual: ${it.capacity()}, expected: $expectedSize"
        }
    })
    private val gridRectUniform = glGetUniformLocation(program, "u_gridRect")
    private val levelUniform = glGetUniformLocation(program, "u_level")
    private val levelOpacityUniform = glGetUniformLocation(program, "u_levelOpacity")

    init {
        val weights = loadFloatBufferFromAsset("network.data")
        val weightsUniform = glGetUniformLocation(program, "u_networkWeights")
        val encodingTextureUniform = glGetUniformLocation(program, "u_encodingTexture")
        val encodingResolutionsUniform = glGetUniformLocation(program, "u_encodingResolutions")
        val encodingParamCountsUniform = glGetUniformLocation(program, "u_encodingParamCounts")
        val encodingParamOffsetsUniform = glGetUniformLocation(program, "u_encodingParamOffsets")

        glUseProgram(program)
        glUniform4fv(weightsUniform, NetworkSettings.weightCount0 / 4, weights)
        glUniform1iv(encodingResolutionsUniform, EncodingSettings.resolutions.size, EncodingSettings.resolutions, 0)
        glUniform1iv(encodingParamCountsUniform, EncodingSettings.paramCounts.size, EncodingSettings.paramCounts, 0)
        glUniform1iv(encodingParamOffsetsUniform, EncodingSettings.paramOffsets.size, EncodingSettings.paramOffsets, 0)
        glUniform1i(encodingTextureUniform, 0)
        glUseProgram(0)
        checkGlError()
    }

    fun draw(quad: Quad, cameraState: CameraState) {
        cameraState.activeGridSize.let {
            glEnable(GL_SCISSOR_TEST)
            glScissor(0, 0, it.width, it.height)
        }
        glUseProgram(program)
        glUniformRect(gridRectUniform, cameraState.gridRect)
        glUniform1i(levelUniform, cameraState.level)
        glUniform1f(levelOpacityUniform, cameraState.levelOpacity)
        encodingTexture.bind()
        quad.draw(positionAttribute)
        glDisable(GL_SCISSOR_TEST)
    }
}
