#!/usr/bin/env python3
"""Calculates the adjugate matrix of a matrix using pure Python"""


def determinant(matrix):
    """Computes the determinant of a square matrix"""
    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    det = 0
    for c in range(n):
        sub_matrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        sign = 1 if c % 2 == 0 else -1
        det += sign * matrix[0][c] * determinant(sub_matrix)
    return det


def adjugate(matrix):
    """Calculates the adjugate matrix of a square matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n or n == 0 or len(row) == 0:
            raise ValueError("matrix must be a non-empty square matrix")

    # The adjugate of a 1x1 matrix [[x]] is always [[1]]
    if n == 1:
        return [[1]]

    # 1. Calculate the Cofactor Matrix
    cofactor_matrix = []
    for r in range(n):
        cofactor_row = []
        for c in range(n):
            sub_matrix = [
                row[:c] + row[c + 1:] for row in (matrix[:r] + matrix[r + 1:])
            ]
            sign = 1 if (r + c) % 2 == 0 else -1
            cofactor_row.append(sign * determinant(sub_matrix))
        cofactor_matrix.append(cofactor_row)

    # 2. Transpose the Cofactor Matrix to get the Adjugate Matrix
    adjugate_matrix = []
    for c in range(n):
        adjugate_row = []
        for r in range(n):
            adjugate_row.append(cofactor_matrix[r][c])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix
