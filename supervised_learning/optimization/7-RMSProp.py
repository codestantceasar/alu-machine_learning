#!/usr/bin/env python3
"""RMSProp optimization"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates variable using RMSProp"""
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
