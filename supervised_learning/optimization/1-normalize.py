#!/usr/bin/env python3
"""Normalize matrix"""


def normalize(X, m, s):
    """Normalizes a matrix"""
    return (X - m) / s
