#!/usr/bin/env python3
"""Calculates the PDF of a multivariate normal distribution"""
import numpy as np


class MultiNormal:
    """Represents a multivariate normal distribution"""

    def __init__(self, data):
        """
        Initialization of MultiNormal
        data: numpy.ndarray of shape (d, n)
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean vector of shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Calculate covariance matrix without np.cov
        X_deviations = data - self.mean
        self.cov = np.dot(X_deviations, X_deviations.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a specific data point
        x: numpy.ndarray of shape (d, 1)
        Returns: the PDF value (float)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # PDF formula components
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)

        # Calculate exponents part
        diff = x - self.mean
        exponent = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)

        # Calculate denominator part
        denominator = np.sqrt(((2 * np.pi) ** d) * det_cov)

        # Final PDF value calculation
        pdf_val = np.exp(exponent) / denominator

        # Return as a pure scalar float
        return float(pdf_val)
