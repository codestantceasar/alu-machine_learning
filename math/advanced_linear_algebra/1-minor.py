#!/usr/bin/env python3
"""Calculates the minor matrix of a matrix using pure Python"""


def determinant(matrix):
    """Computes the determinant of a square matrix"""
    # Handle the specific 0x0 matrix case [[]]
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    n = len(matrix)

    # Base case for 1x1 matrix: return the actual scalar integer
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    # Recursive step for nxn matrix using Laplace expansion
    det = 0
    for c in range(n):
        sub_matrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        sign = 1 if c % 2 == 0 else -1
        det += sign * matrix[0][c] * determinant(sub_matrix)
    return det


def minor(matrix):
    """Calculates the minor matrix of a square matrix"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n or n == 0:
            raise ValueError("matrix must be a non-empty square matrix")

    # Base case for 1x1 matrix minor is [[1]]
    if n == 1:
        return [[1]]

    minor_matrix = []
    for r in range(n):
        minor_row = []
        for c in range(n):
            sub_matrix = [
                row[:c] + row[c + 1:] for row in (matrix[:r] + matrix[r + 1:])
            ]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix
