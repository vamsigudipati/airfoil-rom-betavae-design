import numpy as np
import pytest

from airfoil_rom import metrics as mt


def test_standardised_and_per_channel_mse(tiny_fields):
    rec = tiny_fields + 0.5
    assert mt.standardised_mse(rec, tiny_fields) == pytest.approx(0.25, rel=1e-6)
    np.testing.assert_allclose(mt.per_channel_mse(rec, tiny_fields), [0.25, 0.25, 0.25], rtol=1e-6)
    assert mt.standardised_mse(tiny_fields, tiny_fields) == 0.0


def test_rmse_known_case():
    assert mt.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 7.0]) == pytest.approx(4.0 / np.sqrt(3))


def test_coefficient_scores_known_case():
    true = np.array([[0.5, 0.010], [1.0, 0.012], [1.5, 0.016]])
    pred = true.copy()
    pred[:, 0] += [0.1, -0.1, 0.1]           # CL errors of 0.1 everywhere; CD exact
    row = mt.coefficient_scores(pred, true, "unit", seed=1)
    assert row["model"] == "unit" and row["seed"] == 1
    assert row["CL_MAE"] == pytest.approx(0.1) and row["CL_RMSE"] == pytest.approx(0.1)
    assert row["CL_R2"] == pytest.approx(1.0 - 0.03 / 0.5)      # SS_res 0.03, SS_tot 0.5
    assert row["CD_MAE"] == 0.0 and row["CD_RMSE"] == 0.0 and row["CD_R2"] == pytest.approx(1.0)


def test_error_by_region_shares_sum_to_one(tiny_fields):
    rec = tiny_fields + 1.0
    rows, maps = mt.error_by_region(rec, tiny_fields, tiny_fields[:, 2], ["a", "b", "sdf"], "unit")
    assert [r["region"] for r in rows] == [b[0] for b in mt.REGION_BANDS]
    assert sum(r["share_of_points"] for r in rows) == pytest.approx(1.0)
    assert maps.shape == (3, 16, 32)
    for r in rows:
        if r["share_of_points"] > 0:
            assert r["a"] == pytest.approx(1.0)


def test_nearest_distance():
    train = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    query = np.array([[0.9, 0.0, 0.0], [0.0, 1.9, 5.0]])
    d, j = mt.nearest_distance(query, train)
    np.testing.assert_allclose(d, [0.1, np.hypot(0.1, 5.0)])
    assert list(j) == [1, 2]
    d2, j2 = mt.nearest_distance(query, train, slice(0, 2))     # ignore the third column
    np.testing.assert_allclose(d2, [0.1, 0.1]) and list(j2) == [1, 2]


def test_timed_calls_function_repeats_plus_warmup():
    calls = []
    out = mt.timed(lambda: calls.append(1), repeats=3)
    assert len(out) == 3 and len(calls) == 4 and all(t >= 0 for t in out)


def test_episode_metrics_known_case():
    data = {"cl": np.array([1.0, 1.0, 1.3]), "cl_target": np.array([1.0, 1.0, 1.0]),
            "flap_deg": np.array([0.0, -2.0, 4.0]), "cd": np.array([0.01, 0.02, 0.03]), "reward": np.array([-1.0, -2.0, 0.5])}
    m = mt.episode_metrics(data)
    assert m["CL_RMSE"] == pytest.approx(np.sqrt(0.09 / 3))
    assert m["mean_abs_flap_deg"] == pytest.approx(2.0) and m["max_abs_flap_deg"] == 4.0
    assert m["mean_CD"] == pytest.approx(0.02) and m["episode_return"] == pytest.approx(-2.5)
