#!/usr/bin/env python3
"""
This module contains the poly_derivative function.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Validate that all items in the list are integers or floats
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # A constant polynomial (e.g., [5]) has a derivative of 0
    if len(poly) == 1:
        return [0]

    derivative = []
    for power in range(1, len(poly)):
        derivative.append(poly[power] * power)

    # Trim trailing zeros if the highest degree coefficients became zero
    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative
