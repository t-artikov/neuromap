package me.tartikov.neuromap

object NetworkSettings {
    private const val inputSize = EncodingSettings.levelCount * EncodingSettings.featurePerLevel
    private const val hiddenSize = 8
    private const val outputSize = 4

    const val weightCount0 = inputSize * hiddenSize
    const val weightCount1 = hiddenSize * outputSize
}
