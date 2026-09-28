#!/usr/bin/env python3
"""Creates a layer for a neural network"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer

    prev: output of previous layer
    n: number of nodes
    activation: activation function

    Returns:
        output tensor
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )

    return layer(prev)
