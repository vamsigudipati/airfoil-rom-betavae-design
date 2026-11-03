import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "05": REPO / "notebooks" / "05_airfrans_rom_betavae_design.ipynb",
    "06": REPO / "notebooks" / "06_field_model_in_control_loop.ipynb",
}


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def notebooks():
    """Parsed notebook JSON, keyed by stage ('05', '06'). Never executed."""
    return {k: json.loads(p.read_text()) for k, p in NOTEBOOKS.items()}


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def tiny_fields(rng):
    """Four (N) x 3 (C) x 16 x 32 fields, channels-first, with a signed-distance last channel."""
    fields = rng.normal(size=(4, 3, 16, 32)).astype(np.float32)
    y, x = np.mgrid[0:16, 0:32]
    fields[:, 2] = (np.hypot(x - 10, y - 8) - 3.0) / 10.0   # a small "body" at (10, 8)
    return fields


def linear_aero(reynolds, alpha_deg, m=2.0, p=4.0, t=12.0):
    """A thin-airfoil-like law for the control tests: CL = 0.1 * (alpha + 2), CD = 0.01 + 1e-4 * alpha^2."""
    return 0.1 * (alpha_deg + 2.0), 0.01 + 1.0e-4 * alpha_deg ** 2
