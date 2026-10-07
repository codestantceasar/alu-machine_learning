#!/usr/bin/env python3
"""Specificity"""

import numpy as np


def specificity(confusion):
    """Calculates specificity for each class"""
    classes = confusion.shape[0]
    total = np.sum(confusion)
    specificity = np.zeros(classes)

    for i in range(classes):
        tp = confusion[i, i]
        fn = np.sum(confusion[i, :]) - tp
        fp = np.sum(confusion[:, i]) - tp
        tn = total - tp - fn - fp

        specificity[i] = tn / (tn + fp)

    return specificity
