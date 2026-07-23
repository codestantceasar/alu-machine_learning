#!/usr/bin/env python3
"""
This module contains the summation_i_squared function.
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from i=1 to n without using any loops.
    """
    if not isinstance(n, int) or n < 1:
        return None

    # Apply the mathematical formula for the sum of squares
    total_sum = (n * (n + 1) * (2 * n + 1)) // 6
    return total_sum
