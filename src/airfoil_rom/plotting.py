"""Plotting helpers lifted from Stage 5 (cells 12, 21 and 27). They return the figure instead of calling ``plt.show()``."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_channels(field, x_grid, y_grid, channel_names, title="", sdf_name="sdf", n_columns=3):
    """One panel per channel of a ``(C, NY, NX)`` field with the airfoil outline (Stage 5, cell 12)."""
    X_GRID, Y_GRID = np.meshgrid(x_grid, y_grid, indexing="xy")
    sdf_index = channel_names.index(sdf_name) if sdf_name in channel_names else None
    n_channels = field.shape[0]
    n_rows = int(np.ceil(n_channels / n_columns))

    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(16, 4.2 * n_rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    for channel_index, channel_name in enumerate(channel_names):
        ax = axes[channel_index]
        image = ax.pcolormesh(
            X_GRID, Y_GRID, field[channel_index],
            shading="auto", cmap="coolwarm"
        )

        if sdf_index is not None:
            ax.contour(
                X_GRID, Y_GRID, field[sdf_index],
                levels=[0.0], colors="black", linewidths=1.2
            )

        ax.set_title(channel_name)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, shrink=0.82)

    for ax in axes[n_channels:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    return fig


def plot_field_comparison(fields, titles, x_grid, y_grid, shared_scale=True):
    """Side-by-side panels of ``(NY, NX)`` fields (Stage 5, cell 27; cell 21 with ``shared_scale=False``)."""
    X_GRID, Y_GRID = np.meshgrid(x_grid, y_grid, indexing="xy")
    fig, axes = plt.subplots(1, len(fields), figsize=(16, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    kw = {}
    if shared_scale:
        kw = {"vmin": min(values.min() for values in fields), "vmax": max(values.max() for values in fields)}

    for ax, values, title in zip(axes, fields, titles):
        image = ax.pcolormesh(
            X_GRID, Y_GRID, values, shading="auto", cmap="coolwarm", **kw
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, shrink=0.8)

    return fig
