#!/usr/bin/env python3
"""
This module contains the cat_matrices2D function.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis (0 or 1).
    """
    # Handle vertical concatenation (row-wise)
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]

    # Handle horizontal concatenation (column-wise)
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]

    return None
