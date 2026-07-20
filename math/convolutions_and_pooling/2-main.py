#!/usr/bin/env python3

import numpy as np

convolve_grayscale_padding = __import__(
    '2-convolve_grayscale_padding'
).convolve_grayscale_padding

dataset = np.load("animals_gray.npz")
images = dataset["data"]

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

images_conv = convolve_grayscale_padding(
    images,
    kernel,
    (2, 4)
)

print(images.shape)
print(images_conv.shape)