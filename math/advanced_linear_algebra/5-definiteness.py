#!/usr/bin/env python3
"""Calculates the definiteness of a matrix using NumPy"""
import numpy as np


def definiteness(matrix):
    """Determines the definiteness of a square symmetric matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if the matrix is valid (2D, non-empty, and square)
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.size == 0:
        return None

    # A matrix must be symmetric to classify its definiteness cleanly
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Calculate the eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    # Check eigenvalue profiles
    pos = np.any(eigenvalues > 1e-10)
    neg = np.any(eigenvalues < -1e-10)
    zero = np.any(np.isclose(eigenvalues, 0, atol=1e-10))

    if pos and not neg and not zero:
        return "Positive definite"
    if pos and not neg and zero:
        return "Positive semi-definite"
    if neg and not pos and not zero:
        return "Negative definite"
    if neg and not pos and zero:
        return "Negative semi-definite"
    if pos and neg:
        return "Indefinite"

    return None
