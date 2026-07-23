#!/usr/bin/env python3
"""
This module contains the add_matrices2D function.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise and returns a new matrix.
    """
    # Check if they have the same number of rows
    if len(mat1) != len(mat2):
        return None

    # Check if they have the same number of columns in the first row
    if len(mat1[0]) != len(mat2[0]):
        return None

    # Element-wise addition using nested list comprehensions
    return [[a + b for a, b in zip(row1, row2)]
            for row1, row2 in zip(mat1, mat2)]
