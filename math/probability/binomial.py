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

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes".
        """
        # Convert k to an integer if it's not one
        k = int(k)

        # If k is out of range, return 0
        if k < 0 or k > self.n:
            return 0

        # Helper function to calculate factorials
        def factorial(num):
            fact = 1
            for i in range(1, num + 1):
                fact *= i
            return fact

        n_fact = factorial(self.n)
        k_fact = factorial(k)
        nk_fact = factorial(self.n - k)

        # Calculate binomial coefficient: n! / (k! * (n - k)!)
        combination = n_fact / (k_fact * nk_fact)

        # Calculate PMF value (Broken into 2 lines for PEP8 compliance)
        pmf_value = combination * (self.p ** k) * (
            (1 - self.p) ** (self.n - k)
        )
        return pmf_value
