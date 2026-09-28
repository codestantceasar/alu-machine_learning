#!/usr/bin/env python3
"""Batch normalization layer"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer"""

    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    dense = tf.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )

    Z = dense(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    mean, variance = tf.nn.moments(Z, axes=[0])

    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        beta,
        gamma,
        1e-8
    )

    if activation is None:
        return Z_norm

    return activation(Z_norm)


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates forward propagation graph"""

    output = x

    for i in range(len(layer_sizes)):
        if i < len(layer_sizes) - 1:
            # Hidden layers use Batch Normalization
            output = create_batch_norm_layer(
                output,
                layer_sizes[i],
                activations[i]
            )
        else:
            # Output layer must be a standard dense layer (with bias, no BN)
            initializer = tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"
            )
            dense = tf.layers.Dense(
                units=layer_sizes[i],
                kernel_initializer=initializer,
                use_bias=True
            )
            Z = dense(output)
            if activations[i] is None:
                output = Z
            else:
                output = activations[i](Z)

    return output
