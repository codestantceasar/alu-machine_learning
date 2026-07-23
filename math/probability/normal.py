#!/usr/bin/env python3
"""
This module contains the Normal class that represents
a normal distribution.
"""


class Normal:
    """
    Class that represents a normal distribution.
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate the mean
            self.mean = float(sum(data) / len(data))

            # Calculate the standard deviation (population stddev)
            variance_sum = sum((x - self.mean) ** 2 for x in data)
            self.stddev = float((variance_sum / len(data)) ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.
        """
        pi = 3.1415926536
        e = 2.7182818285

        # Calculate exponent: -0.5 * ((x - mean) / stddev) ** 2
        exponent = -0.5 * (((x - self.mean) / self.stddev) ** 2)

        # Calculate denominator: stddev * sqrt(2 * pi)
        denominator = self.stddev * ((2 * pi) ** 0.5)

        # PDF value calculation
        pdf_value = (e ** exponent) / denominator
        return pdf_value

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value.
        """
        pi = 3.1415926536

        # Value to plug into error function (erf)
        w = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Maclaurin series approximation for erf(w) up to the 9th power term
        erf = (2 / (pi ** 0.5)) * (
            w - (w ** 3) / 3 + (w ** 5) / 10 - (w ** 7) / 42 + (w ** 9) / 216
        )

        # Normal CDF formula using the calculated erf value
        cdf_value = 0.5 * (1 + erf)
        return cdf_value
