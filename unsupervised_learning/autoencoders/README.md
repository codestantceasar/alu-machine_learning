# Autoencoders

This project explores different types of autoencoders using TensorFlow Keras. Autoencoders are neural networks designed to learn efficient representations of data by compressing inputs into a latent space and reconstructing them.

## Learning Objectives

By completing this project, you should understand:

* What an autoencoder is
* What latent space represents
* What a bottleneck is
* What sparse autoencoders are
* What convolutional autoencoders are
* What generative models are
* What variational autoencoders (VAEs) are
* What Kullback-Leibler (KL) Divergence is and why it is used

## Requirements

* Ubuntu 16.04 LTS
* Python 3.5
* TensorFlow 1.12
* NumPy 1.15
* pycodestyle 2.4

## Files

| File                 | Description                                         |
| -------------------- | --------------------------------------------------- |
| `0-vanilla.py`       | Creates a vanilla autoencoder                       |
| `1-sparse.py`        | Creates a sparse autoencoder with L1 regularization |
| `2-convolutional.py` | Creates a convolutional autoencoder                 |
| `3-variational.py`   | Creates a variational autoencoder (VAE)             |
| `README.md`          | Project documentation                               |

## Autoencoder Types

### Vanilla Autoencoder

A basic encoder-decoder architecture that learns a compressed representation of the input and reconstructs it.

### Sparse Autoencoder

An autoencoder that uses regularization to encourage sparse latent representations.

### Convolutional Autoencoder

An autoencoder that uses convolutional layers for image data, preserving spatial information.

### Variational Autoencoder (VAE)

A generative model that learns a probability distribution in latent space and can generate new samples.

## Usage

Make files executable:

```bash
chmod +x *.py
```

Check style:

```bash
pycodestyle *.py
```

## Author

Constantine Akas
