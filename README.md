## Map rendering using a neural network

Based on the great article [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).

![map](https://github.com/t-artikov/neuromap/blob/master/screenshot.png)

The repository consists of:

- **tools/trainer** - a program for training a neural network on raster tiles.
- **tools/tiles_to_image** - converts tiles from custom format to PNG image (for testing purposes).
- **tools/inference** - generates an image from a trained neural network (for testing purposes).
- **app** - An Android application that displays a neural map in real time.