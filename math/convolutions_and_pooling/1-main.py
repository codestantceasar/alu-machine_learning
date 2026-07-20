#!/usr/bin/env python3

import numpy as np

convolve_grayscale_same = __import__(
    '1-convolve_grayscale_same'
).convolve_grayscale_same

dataset = np.load("animals_gray.npz")

images = dataset["data"]

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

images_conv = convolve_grayscale_same(images, kernel)

print(images.shape)
print(images_conv.shape)