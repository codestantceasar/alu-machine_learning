#!/usr/bin/env python3
"""Deep Neural Network"""


import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network"""

    def __init__(self, nx, layers):
        """Class constructor"""

        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError(
                "layers must be a list of positive integers"
            )

        for nodes in layers:
            if type(nodes) is not int or nodes <= 0:
                raise TypeError(
                    "layers must be a list of positive integers"
                )

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if i == 0:
                self.weights["W1"] = (
                    np.random.randn(layers[0], nx)
                    * np.sqrt(2 / nx)
                )
                self.weights["b1"] = np.zeros((layers[0], 1))
            else:
                self.weights["W{}".format(i + 1)] = (
                    np.random.randn(
                        layers[i],
                        layers[i - 1]
                    )
                    * np.sqrt(2 / layers[i - 1])
                )

                self.weights["b{}".format(i + 1)] = (
                    np.zeros((layers[i], 1))
                )
