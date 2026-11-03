import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from airfoil_rom import plotting as pl


def test_plot_channels_builds_one_panel_per_channel(tiny_fields):
    x = np.linspace(-0.5, 1.5, 32); y = np.linspace(-0.5, 0.5, 16)
    fig = pl.plot_channels(tiny_fields[0], x, y, ["a", "b", "sdf"], title="t")
    assert len([ax for ax in fig.axes if ax.get_title() in ("a", "b", "sdf")]) == 3
    plt.close(fig)


def test_plot_field_comparison_shared_scale(tiny_fields):
    x = np.linspace(-0.5, 1.5, 32); y = np.linspace(-0.5, 0.5, 16)
    fig = pl.plot_field_comparison([tiny_fields[0, 0], tiny_fields[1, 0]], ["Reference", "POD"], x, y)
    meshes = [c for ax in fig.axes for c in ax.collections]
    assert len(meshes) >= 2 and meshes[0].get_clim() == meshes[1].get_clim()
    plt.close(fig)
