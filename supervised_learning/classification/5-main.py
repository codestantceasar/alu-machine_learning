#!/usr/bin/env python3

import numpy as np

Neuron = __import__('5-neuron').Neuron

np.random.seed(0)

neuron = Neuron(5)

X = np.random.randn(5, 3)
Y = np.array([[1, 0, 1]])

A = neuron.forward_prop(X)

print("Before:")
print(neuron.W)
print(neuron.b)

neuron.gradient_descent(X, Y, A, 0.05)

print("After:")
print(neuron.W)
print(neuron.b)
