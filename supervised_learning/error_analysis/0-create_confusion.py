#!/usr/bin/env python3
"""Create confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    actual = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)

    for i in range(len(actual)):
        confusion[actual[i], predicted[i]] += 1

    return confusion
