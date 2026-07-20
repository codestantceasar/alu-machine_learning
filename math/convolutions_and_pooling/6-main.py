#!/usr/bin/env python3

import numpy as np

pool = __import__(
    '6-pool'
).pool


data = np.load("animals_1.npz")

images = data["data"]


print(images.shape)


images_pool = pool(
    images,
    (2, 2),
    (2, 2),
    mode='avg'
)


print(images_pool.shape)