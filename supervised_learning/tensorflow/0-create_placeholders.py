#!/usr/bin/env python3
"""Creates placeholders for a neural network"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for the neural network

    nx: number of feature columns
    classes: number of classes

    Returns:
        x, y placeholders
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return x, y
