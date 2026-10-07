#!/usr/bin/env python3
"""Sensitivity"""

import numpy as np


def sensitivity(confusion):
    """Calculates sensitivity for each class"""
    tp = np.diag(confusion)
    fn = np.sum(confusion, axis=1) - tp

    return tp / (tp + fn)
