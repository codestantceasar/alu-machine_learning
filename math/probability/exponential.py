#!/usr/bin/env python3
"""
This module contains the Exponential class that represents
an exponential distribution.
"""


class Exponential:
    """
    Class that represents an exponential distribution.
    """
    def __init__(self, data=None, lambtha=1.0):
        """
        Initializes the Exponential distribution.
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
            # For exponential distribution, lambtha = 1 / mean
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period.
        """
        # If x is out of range (less than 0), return 0
        if x < 0:
            return 0

        # Define Euler's number explicitly as required by the project
        e = 2.7182818285

        # Calculate and return Exponential PDF: lambtha * e^(-lambtha * x)
        pdf_value = self.lambtha * (e ** (-self.lambtha * x))
        return pdf_value
