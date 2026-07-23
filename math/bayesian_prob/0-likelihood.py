#!/usr/bin/env python3
"""Likelihood module"""

import numpy as np
import math


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining the data"""

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    combination = (
        math.factorial(n) /
        (math.factorial(x) * math.factorial(n - x))
    )

    return combination * (P ** x) * ((1 - P) ** (n - x))
