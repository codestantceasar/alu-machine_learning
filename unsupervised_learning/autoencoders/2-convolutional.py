#!/usr/bin/env python3
"""Convolutional Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder"""

    inputs = keras.Input(shape=input_dims)

    x = inputs
    for filt in filters:
        x = keras.layers.Conv2D(
            filt,
            (3, 3),
            activation='relu',
            padding='same'
        )(x)

        x = keras.layers.MaxPooling2D(
            (2, 2),
            padding='same'
        )(x)

    encoder = keras.Model(inputs, x)

    latent_inputs = keras.Input(shape=latent_dims)

    x = latent_inputs

    for filt in filters[::-1][:-1]:
        x = keras.layers.Conv2D(
            filt,
            (3, 3),
            activation='relu',
            padding='same'
        )(x)

        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        filters[0],
        (3, 3),
        activation='relu',
        padding='valid'
    )(x)

    x = keras.layers.UpSampling2D((2, 2))(x)

    outputs = keras.layers.Conv2D(
        input_dims[2],
        (3, 3),
        activation='sigmoid',
        padding='same'
    )(x)

    decoder = keras.Model(latent_inputs, outputs)

    auto_outputs = decoder(encoder(inputs))

    auto = keras.Model(inputs, auto_outputs)

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
