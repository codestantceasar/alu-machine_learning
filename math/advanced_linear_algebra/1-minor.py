#!/usr/bin/env python3
"""Calculates the minor matrix of a matrix using pure Python"""


def determinant(matrix):
    """Computes the determinant of a square matrix"""
    if len(matrix) == 1 and len(matrix) == 0:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix
    if n == 2:
        return matrix * matrix - matrix * matrix
    det = 0
    for c in range(n):
        sub_matrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        sign = 1 if c % 2 == 0 else -1
        det += sign * matrix[c] * determinant(sub_matrix)
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
