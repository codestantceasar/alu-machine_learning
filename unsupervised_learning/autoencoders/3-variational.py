#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras

K = keras.backend


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder"""

    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))

    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims)(x)
    log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        """Samples from latent distribution"""
        mu, log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(mu)[0], latent_dims)
        )
        return mu + K.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(
        encoder_inputs,
        [z, mu, log_var]
    )

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))

    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        decoder_inputs,
        decoder_outputs
    )

    # Autoencoder
    encoded, mu_out, log_var_out = encoder(encoder_inputs)
    reconstructed = decoder(encoded)

    auto = keras.Model(
        encoder_inputs,
        reconstructed
    )

    # VAE loss
    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs,
        reconstructed
    )
    reconstruction_loss *= input_dims

    kl_loss = 1 + log_var_out
    kl_loss -= K.square(mu_out)
    kl_loss -= K.exp(log_var_out)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(
        reconstruction_loss + kl_loss
    )

    auto.add_loss(vae_loss)

    auto.compile(
        optimizer='adam'
    )

    return encoder, decoder, auto
