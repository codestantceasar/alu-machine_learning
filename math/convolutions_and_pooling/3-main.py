#!/usr/bin/env python3

import numpy as np

convolve_grayscale = __import__(
    '3-convolve_grayscale'
).convolve_grayscale


images = np.zeros((50000, 28, 28))

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])


images_conv = convolve_grayscale(
    images,
    kernel,
    padding='valid',
    stride=(2, 2)
)

print(images.shape)
print(images_conv.shape)