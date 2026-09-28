#!/usr/bin/env python3
"""15-main"""

import numpy as np

NeuralNetwork = __import__('15-neural_network').NeuralNetwork

if __name__ == "__main__":
    np.random.seed(0)

    nx = 784
    nodes = 3

    X = np.random.randn(nx, 100)
    Y = np.random.randint(0, 2, (1, 100))

    nn = NeuralNetwork(nx, nodes)

    nn.train(
        X,
        Y,
        iterations=1000,
        alpha=0.05,
        verbose=True,
        graph=True,
        step=100
    )

    prediction, cost = nn.evaluate(X, Y)

    print("Cost:", cost)
    print("Prediction shape:", prediction.shape)