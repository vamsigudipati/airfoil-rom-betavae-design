import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from airfoil_rom import control as ct
from conftest import linear_aero


def test_gust_profile():
    assert ct.one_minus_cosine_gust(0.5, 2.0, 1.0, 3.0) == 0.0
    assert ct.one_minus_cosine_gust(2.5, 2.0, 1.0, 3.0) == pytest.approx(3.0)      # peak at mid-gust
    assert ct.one_minus_cosine_gust(2.0, 2.0, 1.0, 3.0) == pytest.approx(0.0)
    assert ct.one_minus_cosine_gust(3.0, 2.0, 1.0, 3.0) == pytest.approx(0.0, abs=1e-12)


def test_env_passes_gymnasium_checker_and_targets_equilibrium_lift():
    from stable_baselines3.common.env_checker import check_env
    env = ct.AirfoilSurrogateGymEnv(aero_fn=linear_aero)
    check_env(env, warn=True)
    assert env.cl_target == pytest.approx(0.7)                   # 0.1 * (5 + 2)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,) and obs.dtype == np.float32
    assert obs[0] == pytest.approx(0.7 / 1.5) and obs[3] == 0.0


def test_zero_policy_episode_length_and_zero_flap():
    env = ct.AirfoilSurrogateGymEnv(aero_fn=linear_aero, random_gust=False)
    data = ct.run_episode(env, ct.ZeroPolicy(), seed=0)
    assert data["time"].shape == (120,) and data["time"][-1] == pytest.approx(6.0)
    assert np.all(data["flap_deg"] == 0.0)
    assert data["gust_angle_deg"].max() > 0 and data["gust_angle_deg"][0] == 0.0     # fixed gust starts at t = 2 s
    # with the flap at zero the equivalent angle is alpha0 plus the gust angle of the step's start time
    np.testing.assert_allclose(data["alpha_eq_deg"][1:], 5.0 + data["gust_angle_deg"][:-1], atol=1e-6)


def test_flap_actuator_first_order_response():
    env = ct.AirfoilSurrogateGymEnv(aero_fn=linear_aero, random_gust=False)
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["flap_deg"] == pytest.approx(20.0 * 0.05 / 0.10)                       # dt / tau_flap of the command
    assert not terminated and not truncated and reward < 0


def test_fly_and_suite_metrics_rows():
    row = ct.suite_metrics(ct.ZeroPolicy(), "uncontrolled", trust_range=(-4.91, 14.84), gusts=ct.SUITE[:2], aero_fn=linear_aero)
    assert row["configuration"] == "uncontrolled" and row["max_abs_flap_deg"] == 0.0
    assert "RMSE W0=+2.5,Tg=1.3" in row and "RMSE W0=-3.5,Tg=1.0" in row
    assert row["fraction_outside_trusted"] == 0.0 and row["mean_RMSE"] > 0
    tr = ct.fly(ct.ZeroPolicy(), 2.5, 1.3, aero_fn=linear_aero)
    assert set(tr) >= {"time", "cl", "cl_target", "cd", "flap_deg", "alpha_eq_deg", "reward", "action"}
    assert len(tr["time"]) == 120 and np.all(tr["action"] == 0.0)
