#!/usr/bin/env python3
"""Defines a class MultiNormal that represents a Multivariate Normal"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Initializes class attributes mean and covariance"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate Mean (shape d x 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Calculate Covariance without using np.cov
        centered_data = data - self.mean
        self.co = np.matmul(centered_data, centered_data.T) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a given data point x"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Setup math variables
        det = np.linalg.det(self.co)
        inv = np.linalg.inv(self.co)
        
        # Calculate the exponent components: (x - mean)^T * inv * (x - mean)
        diff = x - self.mean
        exponent = -0.5 * np.matmul(np.matmul(diff.T, inv), diff)
        
        # Calculate the normalization factor denominator
        denominator = np.sqrt(((2 * np.pi) ** d) * det)
        
        # Extract scalar value from the 1x1 resulting matrix
        pdf_val = np.exp(exponent) / denominator

        return pdf_val[0, 0]
