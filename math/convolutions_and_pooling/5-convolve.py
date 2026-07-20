#!/usr/bin/env python3
"""
Performs convolution with multiple kernels.
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray of shape (m,h,w,c)
        kernels: numpy.ndarray of shape (kh,kw,c,nc)
        padding: same, valid or tuple
        stride: tuple containing strides

    Returns:
        numpy.ndarray containing convolved images
    """

    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape

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
        ((0,0),
         (ph,ph),
         (pw,pw),
         (0,0)),
        mode='constant'
    )


    output_h = ((h + 2*ph - kh)//sh) + 1
    output_w = ((w + 2*pw - kw)//sw) + 1


    output = np.zeros(
        (m, output_h, output_w, nc)
    )


    for i in range(output_h):

        for j in range(output_w):

            row = i * sh
            col = j * sw


            image_slice = padded[
                :,
                row:row+kh,
                col:col+kw,
                :
            ]


            for k in range(nc):

                output[:, i, j, k] = np.sum(
                    image_slice * kernels[:, :, :, k],
                    axis=(1,2,3)
                )


    return output
