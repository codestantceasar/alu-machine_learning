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
