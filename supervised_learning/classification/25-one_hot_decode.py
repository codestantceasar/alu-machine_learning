#!/usr/bin/env python3
"""One-hot decode module"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels

    Args:
        one_hot: numpy.ndarray of shape (classes, m)

    Returns:
        numpy.ndarray containing the numeric labels for each example,
        or None on failure
    """
    if not isinstance(one_hot, np.ndarray):
        return None

    if len(one_hot.shape) != 2:
        return None

    return np.argmax(one_hot, axis=0)
