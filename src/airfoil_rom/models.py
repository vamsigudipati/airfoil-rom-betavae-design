"""Model definitions lifted verbatim from the notebooks.

Stage 5: cell 23 (``Sampling``, ``build_beta_vae``, ``beta_vae_loss``), cell 24 (β warm-up),
cell 30 (``build_latent_surrogate``), cell 34 (``build_aerodynamic_head``).
Stage 6: cell 6 / cell 16 (NumPy forward pass of a stack of Dense layers, ``dense_stack`` and ``run_stack``,
and the chain ``l12_fast``).

TensorFlow is imported lazily inside the functions that need it so that the NumPy-only
helpers can be used (and tested) without it.
"""
from __future__ import annotations

import numpy as np


def _tf():
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    return tf


def _keras():
    from tensorflow import keras
    from tensorflow.keras import layers
    return keras, layers




def get_sampling_layer():
    """The ``Sampling`` layer class of Stage 5, cell 23 (reparameterisation trick)."""
    tf = _tf()
    _, layers = _keras()

    class Sampling(layers.Layer):
        def call(self, inputs):
            mu, logvar = inputs
            epsilon = tf.random.normal(shape=tf.shape(mu))
            return mu + tf.exp(0.5 * logvar) * epsilon

    return Sampling


def build_beta_vae(input_shape, latent_dim):
    """Convolutional encoder (``[mu, logvar, z]``) and decoder of Stage 5, cell 23.

    ``input_shape`` is channels-last ``(NY, NX, C)``; three stride-2 convolutions, so NY and NX
    must be divisible by 8 for the decoder to return the input shape.
    """
    keras, layers = _keras()
    Sampling = get_sampling_layer()

    encoder_input = keras.Input(shape=input_shape, name="field")

    x = layers.Conv2D(16, 4, strides=2, padding="same", activation="relu")(encoder_input)
    x = layers.Conv2D(32, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 4, strides=2, padding="same", activation="relu")(x)

    encoded_shape = tuple(int(value) for value in x.shape[1:])
    x = layers.Flatten()(x)

    mu = layers.Dense(latent_dim, name="mu")(x)
    logvar = layers.Dense(latent_dim, name="logvar")(x)
    z = Sampling(name="sampling")([mu, logvar])

    encoder = keras.Model(
        encoder_input, [mu, logvar, z], name="encoder"
    )

    decoder_input = keras.Input(shape=(latent_dim,), name="latent_vector")
    x = layers.Dense(int(np.prod(encoded_shape)), activation="relu")(decoder_input)
    x = layers.Reshape(encoded_shape)(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(16, 4, strides=2, padding="same", activation="relu")(x)
    decoder_output = layers.Conv2DTranspose(
        input_shape[-1], 4, strides=2, padding="same", name="reconstruction"
    )(x)

    decoder = keras.Model(decoder_input, decoder_output, name="decoder")
    return encoder, decoder


def beta_vae_loss(reconstruction, target, mu, logvar, beta):
    """``(total, reconstruction, kl)`` of Stage 5, cell 23: MSE plus ``beta`` times the mean KL term."""
    tf = _tf()
    reconstruction_loss = tf.reduce_mean(tf.square(reconstruction - target))
    kl_loss = -0.5 * tf.reduce_mean(
        1.0 + logvar - tf.square(mu) - tf.exp(logvar)
    )
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def beta_schedule(epoch, beta_max=1.0e-3, warmup_epochs=30):
    """Linear β warm-up of Stage 5, cell 24: ``beta_max * min(1, (epoch + 1) / warmup_epochs)`` (``epoch`` from 0)."""
    return beta_max * min(1.0, (epoch + 1) / warmup_epochs)


def build_latent_surrogate(parameter_dim, latent_dim):
    """Parameters -> latent coordinates MLP of Stage 5, cell 30 (two tanh layers of 64)."""
    keras, layers = _keras()
    return keras.Sequential(
        [
            keras.Input(shape=(parameter_dim,)),
            layers.Dense(64, activation="tanh"),
            layers.Dense(64, activation="tanh"),
            layers.Dense(latent_dim),
        ],
        name="latent_surrogate",
    )


def build_aerodynamic_head(latent_dim):
    """Latent coordinates -> (C_L, C_D) MLP of Stage 5, cell 34 (one tanh layer of 32)."""
    keras, layers = _keras()
    return keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(32, activation="tanh"),
            layers.Dense(2),
        ],
        name="aerodynamic_head",
    )


def dense_stack(model):
    """``[(W, b, activation name), ...]`` of a Keras model made of Dense layers (Stage 6, cell 16)."""
    return [(l.get_weights()[0], l.get_weights()[1], l.activation.__name__) for l in model.layers]


def run_stack(stack, h):
    """NumPy forward pass through :func:`dense_stack` output; only tanh and linear activations (Stage 6, cell 16)."""
    for w, b, act in stack:
        h = h @ w + b
        if act == "tanh":
            h = np.tanh(h)
        elif act != "linear":
            raise ValueError(act)
    return h


class NumpyChain:
    """NumPy evaluation of the Stage 5 chain, the function ``l12_fast`` of Stage 6, cell 16.

    ``(m, p, t digits, Re, alpha)`` -> parameter scaler -> latent surrogate (times its global
    scale) -> latent scaler -> aerodynamic head -> coefficient scaler -> ``(C_L, C_D)``.
    Every scaler is a fitted ``StandardScaler`` (only ``mean_`` and ``scale_`` are read).
    """

    def __init__(self, surrogate, head, scale, parameter_scaler, z_scaler, coef_scaler):
        self._sur, self._head = dense_stack(surrogate), dense_stack(head)
        self._pm, self._ps = parameter_scaler.mean_, parameter_scaler.scale_
        self._zm, self._zs = z_scaler.mean_, z_scaler.scale_
        self._cm, self._cs = coef_scaler.mean_, coef_scaler.scale_
        self._scale = scale

    def __call__(self, reynolds, alpha_deg, m=2.0, p=4.0, t=12.0):
        x = (np.array([m / 100.0, p / 10.0, t / 100.0, np.log10(reynolds), alpha_deg]) - self._pm) / self._ps
        z = run_stack(self._sur, x) * self._scale
        cl, cd = run_stack(self._head, (z - self._zm) / self._zs) * self._cs + self._cm
        return float(cl), float(cd)
