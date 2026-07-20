#!/usr/bin/env python3
"""
Performs convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale(images, kernel,
                       padding='same',
                       stride=(1, 1)):
    """
    Performs convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing convolved images.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2

    elif padding == 'valid':
        ph = 0
        pw = 0

    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output_h = int(np.ceil((h + 2 * ph - kh + 1) / sh))
    output_w = int(np.ceil((w + 2 * pw - kw + 1) / sw))

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):

            row = i * sh
            col = j * sw

            output[:, i, j] = np.sum(
                padded[:, row:row + kh, col:col + kw] * kernel,
                axis=(1, 2)
            )

    return output
