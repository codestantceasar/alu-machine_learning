#!/usr/bin/env python3
"""Momentum optimization operation"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates momentum training operation"""
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=alpha,
        momentum=beta1
    )

    return optimizer.minimize(loss)
