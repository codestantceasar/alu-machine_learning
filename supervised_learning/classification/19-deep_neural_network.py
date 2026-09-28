#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initialize the deep neural network"""

        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError(
                "layers must be a list of positive integers"
            )

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx

        for i, nodes in enumerate(layers):
            if type(nodes) is not int or nodes <= 0:
                raise TypeError(
                    "layers must be a list of positive integers"
                )

            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, prev)
                * np.sqrt(2 / prev)
            )

            self.__weights["b{}".format(i + 1)] = np.zeros(
                (nodes, 1)
            )

            prev = nodes

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

        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]

            A_prev = self.__cache["A{}".format(i - 1)]

            Z = np.matmul(W, A_prev) + b

            A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(i)] = A

        return A, self.__cache
    
    def cost(self, Y, A):
        """Calculates the cost of the model"""

        m = Y.shape[1]

        cost = -np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        ) / m
        
        return cost