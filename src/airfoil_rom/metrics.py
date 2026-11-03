"""Error measures and small analysis helpers lifted from the notebooks.

Stage 5: cells 18/25 (standardised MSE), 54 (per-channel MSE), 57 (error by region),
63 (``scores``), 72 (``nearest``), 82 (``timed``). Stage 6: cell 10 (``episode_metrics``).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

#: Distance bands of Stage 5, cell 57: (label, lower bound, upper bound) in chords of signed distance.
REGION_BANDS = [("inside the body", -9, 0.0), ("0-0.02 c", 0.0, 0.02), ("0.02-0.1 c", 0.02, 0.1), ("beyond 0.1 c", 0.1, 9)]


def standardised_mse(reconstruction, reference):
    """Mean squared error over every element (Stage 5, cell 18: ``np.mean((test_scaled - rec) ** 2)``)."""
    return float(np.mean((reconstruction - reference) ** 2))


def per_channel_mse(reconstruction, reference):
    """Mean squared error per channel of ``(N, C, NY, NX)`` tensors (Stage 5, cell 54)."""
    return ((reconstruction - reference) ** 2).mean(axis=(0, 2, 3))


def rmse(prediction, target):
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))


def coefficient_scores(pred, test_coefficients, label, **extra):
    """MAE, RMSE and R2 of C_L and C_D on the test cases, as one row (Stage 5, cell 63, ``scores``)."""
    out = {"model": label, **extra}
    for j, c in enumerate(["CL", "CD"]):
        y = test_coefficients[:, j]
        out[f"{c}_MAE"] = float(mean_absolute_error(y, pred[:, j]))
        out[f"{c}_RMSE"] = float(np.sqrt(np.mean((pred[:, j] - y) ** 2)))
        out[f"{c}_R2"] = float(r2_score(y, pred[:, j]))
    return out


def error_by_region(reconstruction, reference, sdf, channel_names, model, bands=REGION_BANDS):
    """Mean squared error per channel inside distance bands from each case's own surface (Stage 5, cell 57).

    ``sdf`` is the signed-distance channel of the *reference* fields, shape ``(N, NY, NX)``.
    Returns the rows of ``ex3_error_by_region.csv`` for one model and the mean error map ``(C, NY, NX)``.
    """
    sq = (reconstruction - reference) ** 2
    rows = []
    for band, lo, hi in bands:
        mask = (sdf >= lo) & (sdf < hi)
        rows.append({"model": model, "region": band, "share_of_points": float(mask.mean()),
                     **{c: float(sq[:, j][mask].mean()) for j, c in enumerate(channel_names)}})
    return rows, sq.mean(axis=0)


def nearest_distance(S, S_train, idx=slice(None)):
    """Euclidean distance from each row of ``S`` to its nearest row of ``S_train`` (Stage 5, cell 72, ``nearest``).

    ``idx`` restricts the columns (``slice(0, 3)`` for the three shape parameters).
    Returns ``(distances, indices)``.
    """
    d = np.sqrt(((S[:, None, idx] - S_train[None, :, idx]) ** 2).sum(axis=-1))
    return d.min(axis=1), d.argmin(axis=1)


def timed(fn, repeats=3):
    """Wall time of ``fn()`` over ``repeats`` calls after one warm-up call (Stage 5, cell 82)."""
    fn()                                                       # warm-up: Keras builds its predict function on the first call
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); out.append(time.perf_counter() - t0)
    return out


def episode_metrics(data):
    """Tracking error, flap use, drag and return of one episode (Stage 6, cell 10; copied from Stage 4)."""
    error = data["cl"] - data["cl_target"]

    return {
        "CL_RMSE": np.sqrt(np.mean(error**2)),
        "mean_abs_flap_deg": np.mean(np.abs(data["flap_deg"])),
        "max_abs_flap_deg": np.max(np.abs(data["flap_deg"])),
        "mean_CD": np.mean(data["cd"]),
        "episode_return": np.sum(data["reward"]),
    }
