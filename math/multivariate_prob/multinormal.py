#!/usr/bin/env python3
"""Multivariate Normal distribution module"""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Initialize MultiNormal with data"""

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError(
                "data must be a 2D numpy.ndarray"
            )

        if data.shape[1] < 2:
            raise ValueError(
                "data must contain multiple data points"
            )

        self.mean = np.mean(data, axis=1, keepdims=True)

        centered = data - self.mean

        self.cov = np.dot(
            centered,
            centered.T
        ) / (data.shape[1] - 1)
