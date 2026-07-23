#!/usr/bin/env python3
"""
This module contains the Poisson class that represents a Poisson distribution.
"""


class Poisson:
    """
    Class that represents a Poisson distribution.
    """
    def __init__(self, data=None, lambtha=1.0):
        """
        Initializes the Poisson distribution.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes".
        """
        # Convert k to an integer if it's not one
        k = int(k)

        # If k is out of range (less than 0), return 0
        if k < 0:
            return 0

        # Define Euler's number explicitly as requested by the project manual
        e = 2.7182818285

        # Calculate the factorial of k (k!)
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Calculate the Poisson Probability Mass Function (PMF)
        pmf_value = ((e ** -self.lambtha) * (self.lambtha ** k)) / factorial
        return pmf_value
