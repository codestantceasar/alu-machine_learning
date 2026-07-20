#!/usr/bin/env python3

import numpy as np

convolve = __import__(
    "5-convolve"
).convolve


data = np.load("animals_1.npz")

images = data["data"]


kernels = np.random.randn(
    3,3,3,3
)


result = convolve(
    images,
    kernels,
    padding='valid'
)


print(result.shape)