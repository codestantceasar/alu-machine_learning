#!/usr/bin/env python3
"""
This module contains the add_arrays function.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise and returns a new list.
    """
    if len(arr1) != len(arr2):
        return None

    return [a + b for a, b in zip(arr1, arr2)]
