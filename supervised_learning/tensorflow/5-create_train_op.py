#!/usr/bin/env python3
"""Creates training operation"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates training operation

    loss: network loss
    alpha: learning rate

    Returns:
        training operation
    """
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=alpha
    )

    train_op = optimizer.minimize(loss)

    return train_op
