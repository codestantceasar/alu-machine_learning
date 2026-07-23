#!/usr/bin/env python3
"""
This module contains the matrix_transpose function.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.
    """
    # Number of columns in the original matrix determines the new row count
    cols = len(matrix[0])
    
    # Construct the transposed matrix
    return [[row[i] for row in matrix] for i in range(cols)]
