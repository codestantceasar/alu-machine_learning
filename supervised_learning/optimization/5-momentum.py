#!/usr/bin/env python3
"""Momentum optimization"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates variable using momentum"""
    v = beta1 * v + (1 - beta1) * grad

    var = var - alpha * v

    return var, v
