#!/usr/bin/env python3
"""Calculates the cofactor matrix of a matrix using pure Python"""


def determinant(matrix):
    """Computes the determinant of a square matrix"""
    if len(matrix) == 1 and len(matrix) == 0:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix

    if n == 2:
        return (matrix * matrix -
                matrix * matrix)

    det = 0
    for c in range(n):
        sub_matrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        sign = 1 if c % 2 == 0 else -1
        det += sign * matrix[c] * determinant(sub_matrix)
    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a square matrix"""
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

    # The cofactor of a 1x1 matrix is defined as [[1]] by convention
    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for r in range(n):
        cofactor_row = []
        for c in range(n):
            # Form sub-matrix by removing row r and column c
            sub_matrix = [
                row[:c] + row[c + 1:] for row in (matrix[:r] + matrix[r + 1:])
            ]
            # Sign pattern (-1)^(r + c)
            sign = 1 if (r + c) % 2 == 0 else -1
            cofactor_row.append(sign * determinant(sub_matrix))
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
