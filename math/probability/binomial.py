#!/usr/bin/env python3
"""
This module contains the Binomial class that represents
a binomial distribution.
"""


class Binomial:
    """
    Class that represents a binomial distribution.
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate sample mean (mu) and variance (sigma^2)
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimate initial p using: variance = mean * (1 - p)
            initial_p = 1.0 - (variance / mean)

            # Estimate and round n to nearest integer
            calculated_n = round(mean / initial_p)

            # Recalculate p based on the exact integer n
            calculated_p = mean / calculated_n

            self.n = int(calculated_n)
            self.p = float(calculated_p)
