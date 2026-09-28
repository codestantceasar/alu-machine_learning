#!/usr/bin/env python3
"""Moving average"""


def moving_average(data, beta):
    """Calculates weighted moving average"""
    v = 0
    moving_avg = []

    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]

        v_corrected = v / (1 - beta ** (i + 1))

        moving_avg.append(v_corrected)

    return moving_avg
