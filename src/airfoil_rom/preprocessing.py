"""Data preparation helpers lifted verbatim from the notebooks.

Stage 5 (``notebooks/05_airfrans_rom_betavae_design.ipynb``): cells 16, 24-25, 29 and 66.
Stage 6 (``notebooks/06_field_model_in_control_loop.ipynb``): cell 14 (``l12_params``).
"""
from __future__ import annotations

import numpy as np

PARAMETER_NAMES = ["m", "p", "t", "Re", "alpha_deg"]


def channel_statistics(train_fields, floor=1.0e-8):
    """Per-channel mean and standard deviation of a ``(N, C, NY, NX)`` tensor (Stage 5, cell 16).

    Both are returned with shape ``(1, C, 1, 1)`` so they broadcast over fields; the
    standard deviation is floored at ``floor`` to avoid division by zero.
    """
    channel_mean = train_fields.mean(axis=(0, 2, 3), keepdims=True)
    channel_std = train_fields.std(axis=(0, 2, 3), keepdims=True)
    channel_std = np.maximum(channel_std, floor)
    return channel_mean, channel_std


def standardise_fields(fields, channel_mean, channel_std):
    """``(fields - mean) / std`` with the statistics of :func:`channel_statistics` (Stage 5, cell 16)."""
    return (fields - channel_mean) / channel_std


def destandardise_fields(scaled, channel_mean, channel_std):
    """Inverse of :func:`standardise_fields` (Stage 5, cells 66 and 79: ``* channel_std[0] + channel_mean[0]``)."""
    return scaled * channel_std + channel_mean


def to_channels_last(fields):
    """``(N, C, NY, NX)`` -> ``(N, NY, NX, C)`` float32, the Keras image layout (Stage 5, cell 24)."""
    return np.transpose(fields, (0, 2, 3, 1)).astype(np.float32)


def to_channels_first(images):
    """``(N, NY, NX, C)`` -> ``(N, C, NY, NX)``, the database layout (Stage 5, cell 25)."""
    return np.transpose(images, (0, 3, 1, 2))


def prepare_parameters(parameters):
    """Replace the Reynolds number (column 3) by its log10; float32 copy (Stage 5, cell 29)."""
    prepared = parameters.copy().astype(np.float32)
    prepared[:, 3] = np.log10(prepared[:, 3])
    return prepared


def make_candidate_grid(lo, hi, reynolds=4.0e6, alpha_deg=5.0, n_m=15, n_p=15, n_t=16):
    """Grid of four-digit candidates inside the training box of ``(m, p, t)`` (Stage 5, cell 66).

    ``lo`` and ``hi`` are the per-parameter minima and maxima of the training parameters
    (chord fractions for m, p, t). Returns a float32 array of shape ``(n_m * n_p * n_t, 5)``
    in the notebook's units, with the Reynolds number and the angle of attack fixed.
    """
    grid_m = np.linspace(lo[0], hi[0], n_m)
    grid_p = np.linspace(lo[1], hi[1], n_p)
    grid_t = np.linspace(lo[2], hi[2], n_t)
    M_, P_, T_ = np.meshgrid(grid_m, grid_p, grid_t, indexing="ij")
    candidates = np.column_stack([
        M_.ravel(), P_.ravel(), T_.ravel(),
        np.full(M_.size, reynolds), np.full(M_.size, alpha_deg),
    ]).astype(np.float32)
    return candidates, (grid_m, grid_p, grid_t)


def digits_to_chain_inputs(reynolds, alpha_deg, m=2.0, p=4.0, t=12.0):
    """NACA digits and Re (Stage 3 convention) -> raw Stage 5 inputs ``(m/100, p/10, t/100, log10 Re, alpha)``.

    This is the array ``l12_params`` of Stage 6 (cell 14) builds before it applies the
    parameter scaler; ``alpha_deg`` may be a scalar or a 1-d array.
    """
    a = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
    return np.column_stack([
        np.full_like(a, m / 100.0), np.full_like(a, p / 10.0), np.full_like(a, t / 100.0),
        np.full_like(a, np.log10(reynolds)), a,
    ])


def chain_inputs_to_stage3(params):
    """Stage 5 parameter rows ``(m, p, t, Re, alpha)`` -> Stage 3 surrogate inputs ``(Re/1e6, alpha, m digit, p digit, t digit)``.

    The conversion inside ``provided_predict`` (Stage 5, cell 48).
    """
    P_ = np.asarray(params, dtype=np.float64)
    return np.column_stack([P_[:, 3] / 1.0e6, P_[:, 4], P_[:, 0] * 100.0, P_[:, 1] * 10.0, P_[:, 2] * 100.0])
