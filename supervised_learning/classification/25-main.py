#!/usr/bin/env python3

import numpy as np

oh_decode = __import__('25-one_hot_decode').one_hot_decode

lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:10]

one_hot = np.zeros((10, Y.shape[0]))
one_hot[Y, np.arange(Y.shape[0])] = 1

print(Y)
print(one_hot)
print(oh_decode(one_hot))