#!/usr/bin/env python3
"""
This module contains the mat_mul function.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices and returns a new matrix.
    """
    # Matrix multiplication condition: columns of mat1 == rows of mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the resulting matrix with zeros
    # Shape of result will be (rows of mat1) x (columns of mat2)
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform dot product multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
