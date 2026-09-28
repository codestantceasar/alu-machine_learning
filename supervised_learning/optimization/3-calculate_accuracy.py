#!/usr/bin/env python3
"""Calculates prediction accuracy"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates accuracy

    y: labels
    y_pred: predictions

    Returns:
        accuracy tensor
    """
    correct = tf.equal(
        tf.argmax(y, 1),
        tf.argmax(y_pred, 1)
    )

    accuracy = tf.reduce_mean(
        tf.cast(correct, tf.float32)
    )

    return accuracy
