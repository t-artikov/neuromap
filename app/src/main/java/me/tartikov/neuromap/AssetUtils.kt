package me.tartikov.neuromap

import java.io.BufferedReader
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.nio.channels.FileChannel

fun loadStringFromAsset(fileName: String): String {
    return Application.instance.assets
        .open(fileName).bufferedReader().use(BufferedReader::readText)
}

fun loadBufferFromAsset(fileName: String): ByteBuffer {
    Application.instance.assets.openFd(fileName).use { descriptor ->
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength
            )
        }
    }
}

fun loadFloatBufferFromAsset(fileName: String): FloatBuffer {
    return loadBufferFromAsset(fileName).apply {
        order(ByteOrder.nativeOrder())
    }.asFloatBuffer()
}

fun loadShortBufferFromAsset(fileName: String): ShortBuffer {
    return loadBufferFromAsset(fileName).apply {
        order(ByteOrder.nativeOrder())
    }.asShortBuffer()
}
