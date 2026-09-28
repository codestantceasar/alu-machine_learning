#!/usr/bin/env python3
"""One-hot encoding module"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Args:
        Y: numpy.ndarray of shape (m,) containing numeric class labels
        classes: maximum number of classes

    Returns:
        One-hot matrix of shape (classes, m), or None on failure
    """
    try:
        if not isinstance(Y, np.ndarray):
            return None

        if not isinstance(classes, int) or classes <= 0:
            return None

        m = Y.shape[0]

        one_hot = np.zeros((classes, m))

        one_hot[Y, np.arange(m)] = 1

        return one_hot

    except Exception:
        return None
