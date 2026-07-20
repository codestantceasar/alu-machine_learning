#!/usr/bin/env python3
"""
Performs convolution on images with channels.
"""

import numpy as np


def convolve_channels(images, kernel,
                      padding='same',
                      stride=(1, 1)):
    """
    Performs convolution on images with channels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel: numpy.ndarray with shape (kh, kw, c)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing convolved images.
    """

    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
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
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):

            row = i * sh
            col = j * sw

            output[:, i, j] = np.sum(
                padded[
                    :,
                    row:row + kh,
                    col:col + kw,
                    :
                ] * kernel,
                axis=(1, 2, 3)
            )

    return output
