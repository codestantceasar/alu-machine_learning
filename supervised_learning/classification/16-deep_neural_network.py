#!/usr/bin/env python3
"""Deep Neural Network class"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

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
            layer = i + 1

            if i == 0:
                self.weights["W{}".format(layer)] = (
                    np.random.randn(layers[i], nx)
                    * np.sqrt(2 / nx)
                )
            else:
                self.weights["W{}".format(layer)] = (
                    np.random.randn(
                        layers[i],
                        layers[i - 1]
                    ) * np.sqrt(2 / layers[i - 1])
                )

            self.weights["b{}".format(layer)] = np.zeros(
                (layers[i], 1)
            )
