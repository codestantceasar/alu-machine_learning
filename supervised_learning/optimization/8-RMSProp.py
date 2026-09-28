#!/usr/bin/env python3
"""RMSProp operation"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates RMSProp training operation"""
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        decay=beta2,
        epsilon=epsilon
    )

    return optimizer.minimize(loss)
