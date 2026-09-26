#!/usr/bin/env python3

import numpy as np

Neuron = __import__('4-neuron').Neuron

np.random.seed(0)

neuron = Neuron(5)

X = np.random.randn(5, 3)
Y = np.array([[1, 0, 1]])

prediction, cost = neuron.evaluate(X, Y)

print(prediction)
print(cost)
