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

    def evaluate(self, X, Y):
        """Evaluates the neural network"""

        A, _ = self.forward_prop(X)

        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent"""

        m = Y.shape[1]
        weights_copy = self.__weights.copy()

        for layer in range(self.__L, 0, -1):

            A_curr = cache["A{}".format(layer)]
            A_prev = cache["A{}".format(layer - 1)]

            if layer == self.__L:
                dZ = A_curr - Y
            else:
                dZ = (
                    np.matmul(
                        weights_copy["W{}".format(layer + 1)].T,
                        dZ
                    )
                    * A_curr
                    * (1 - A_curr)
                )

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(
                dZ,
                axis=1,
                keepdims=True
            ) / m

            self.__weights["W{}".format(layer)] = (
                self.__weights["W{}".format(layer)]
                - alpha * dW
            )

            self.__weights["b{}".format(layer)] = (
                self.__weights["b{}".format(layer)]
                - alpha * db
            )

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the deep neural network"""

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError(
                "iterations must be a positive integer"
            )

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(
                Y,
                cache,
                alpha
            )

        return self.evaluate(X, Y)
