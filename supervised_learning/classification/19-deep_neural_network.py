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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                self.__weights["W1"] = (
                    np.random.randn(layers[0], nx)
                    * np.sqrt(2 / nx)
                )
                self.__weights["b1"] = np.zeros(
                    (layers[0], 1)
                )
            else:
                self.__weights["W{}".format(i + 1)] = (
                    np.random.randn(
                        layers[i],
                        layers[i - 1]
                    )
                    * np.sqrt(2 / layers[i - 1])
                )
                self.__weights["b{}".format(i + 1)] = (
                    np.zeros((layers[i], 1))
                )

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation"""

        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]

            Z = np.matmul(
                W,
                self.__cache["A{}".format(layer - 1)]
            ) + b

            self.__cache["A{}".format(layer)] = (
                1 / (1 + np.exp(-Z))
            )
        return (
            self.__cache["A{}".format(self.__L)],
            self.__cache
        )

    def cost(self, Y, A):
        """Calculates the cost of the model"""

        m = Y.shape[1]

        return -np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        ) / m
