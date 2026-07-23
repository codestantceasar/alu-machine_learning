#!/usr/bin/env python3
"""
This module contains the poly_integral function.
"""


def poly_integral(poly, C=0):
    """
    Calculates the indefinite integral of a polynomial.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None
    if isinstance(C, float) and C.is_integer():
        C = int(C)

    # Validate that all items in the list are integers or floats
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # Pre-populate the list with the integration constant C
    integral = [C]

    # Apply the integration power rule term by term
    for power, coef in enumerate(poly):
        new_power = power + 1
        new_coef = coef / new_power

        # Convert to integer if it's a whole number
        if new_coef.is_integer():
            new_coef = int(new_coef)

        integral.append(new_coef)

    # Trim trailing zeros so the list is as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
