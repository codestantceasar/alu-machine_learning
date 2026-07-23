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

        # Define Euler's number explicitly
        e = 2.7182818285

        # Calculate the factorial of k (k!)
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Calculate the Poisson Probability Mass Function (PMF)
        pmf_value = ((e ** -self.lambtha) * (self.lambtha ** k)) / factorial
        return pmf_value

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of "successes".
        """
        # Convert k to an integer if it's not one
        k = int(k)

        # If k is out of range (less than 0), return 0
        if k < 0:
            return 0

        # Sum up PMF values from 0 up to and including k
        cdf_value = 0.0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
