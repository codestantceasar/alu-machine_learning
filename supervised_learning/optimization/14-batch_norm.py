#!/usr/bin/env python3
"""Batch normalization layer"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation, epsilon=1e-8):
    """creates a batch normalization layer for hidden layers"""

    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    # Hidden layers use use_bias=False because Batch Norm has its own beta offset
    dense = tf.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )

    Z = dense(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True, name="gamma")
    beta = tf.Variable(tf.zeros([n]), trainable=True, name="beta")

    mean, variance = tf.nn.moments(Z, axes=[0])

    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        beta,
        gamma,
        epsilon
    )

    if activation is None:
        return Z_norm

    return activation(Z_norm)


def forward_prop(x, layer_sizes=[], activations=[], epsilon=1e-8):
    """creates forward propagation graph"""

    output = x
    num_layers = len(layer_sizes)

    for i in range(num_layers):
        if i < num_layers - 1:
            # Hidden layers get Batch Normalization
            output = create_batch_norm_layer(
                output,
                layer_sizes[i],
                activations[i],
                epsilon=epsilon
            )
        else:
            # Output layer is a standard Dense layer (with bias, no BN)
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
