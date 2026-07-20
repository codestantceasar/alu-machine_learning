#!/usr/bin/env python3

import numpy as np

convolve_channels = __import__(
    '4-convolve_channels'
).convolve_channels


dataset = np.load("animals_1.npz")

images = dataset["data"]

print(images.shape)


kernel = np.array(
[
    [
        [0,0,0],
        [-1,-1,-1],
        [0,0,0]
    ],
    [
        [-1,-1,-1],
        [5,5,5],
        [-1,-1,-1]
    ],
    [
        [0,0,0],
        [-1,-1,-1],
        [0,0,0]
    ]
])


images_conv = convolve_channels(
    images,
    kernel,
    padding='valid'
)


print(images_conv.shape)