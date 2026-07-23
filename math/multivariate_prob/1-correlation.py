#!/usr/bin/env python3
"""Correlation module"""

import numpy as np


def correlation(C):
    """Calculates the correlation matrix from a covariance matrix"""

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]

    D = np.sqrt(np.diag(C))

    correlation_matrix = C / np.outer(D, D)

    return correlation_matrix
