#!/usr/bin/env python3
"""Precision"""

import numpy as np


def precision(confusion):
    """Calculates precision for each class"""
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp

    return tp / (tp + fp)
