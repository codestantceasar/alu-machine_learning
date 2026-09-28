#!/usr/bin/env python3
"""Calculates loss"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates softmax cross-entropy loss

    y: labels
    y_pred: predictions

    Returns:
        loss tensor
    """
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred
    )

    return loss
