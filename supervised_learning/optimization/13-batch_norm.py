#!/usr/bin/env python3
"""Batch normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output"""
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    return gamma * Z_norm + beta
