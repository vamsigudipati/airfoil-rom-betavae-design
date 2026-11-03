"""The flap-control environment and its helpers, copied from Stage 6 (cells 7-12 and 16).

The notebook's environment reads the aerodynamic law from a module-level function
(``predict_airfrans_aero``); here it is passed in as ``aero_fn(reynolds, alpha_deg, m, p, t)
-> (cl, cd)`` so that the same class serves the Stage 3 surrogate, the Stage 5 chain
(``Stage5ChainEnv`` of the notebook is ``AirfoilSurrogateGymEnv(aero_fn=chain)``) and the
tests. Everything else is unchanged.
"""
from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .metrics import episode_metrics


def one_minus_cosine_gust(t, t0, duration, amplitude):
    """1 - cos gust profile (Stage 6, cell 7): zero outside ``[t0, t0 + duration]``."""
    if t < t0 or t > t0 + duration:
        return 0.0

    xi = (t - t0) / duration
    return 0.5 * amplitude * (1.0 - np.cos(2.0 * np.pi * xi))


class AirfoilSurrogateGymEnv(gym.Env):
    """Flap-control environment of Stage 4 / Stage 6, cell 8, with the aerodynamic law injected."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        aero_fn,
        dt=0.05,
        horizon=120,
        U_inf=30.0,
        alpha0_deg=5.0,
        cl_target=None,
        max_flap_deg=20.0,
        flap_effectiveness=0.55,
        tau_lift=0.18,
        tau_flap=0.10,
        reynolds=4.0e6,
        naca_m=2.0,
        naca_p=4.0,
        naca_t=12.0,
        random_gust=True,
        gust_amplitude_range=(-3.0, 3.0),
        gust_duration_range=(0.8, 1.8),
        gust_start_range=(1.0, 3.0),
        effort_weight=0.015,
        smoothness_weight=0.020,
        drag_weight=0.020,
    ):
        super().__init__()

        self.aero_fn = aero_fn
        self.dt = dt
        self.horizon = horizon
        self.U_inf = U_inf
        self.alpha0_deg = alpha0_deg
        self.max_flap_deg = max_flap_deg
        self.flap_effectiveness = flap_effectiveness
        self.tau_lift = tau_lift
        self.tau_flap = tau_flap

        self.reynolds = reynolds
        self.naca_m = naca_m
        self.naca_p = naca_p
        self.naca_t = naca_t

        self.random_gust = random_gust
        self.gust_amplitude_range = gust_amplitude_range
        self.gust_duration_range = gust_duration_range
        self.gust_start_range = gust_start_range

        self.effort_weight = effort_weight
        self.smoothness_weight = smoothness_weight
        self.drag_weight = drag_weight

        # If no target is prescribed, use the zero-gust equilibrium lift.
        cl_equilibrium, _ = self.aero_fn(
            self.reynolds,
            self.alpha0_deg,
            self.naca_m,
            self.naca_p,
            self.naca_t,
        )
        self.cl_target = float(cl_equilibrium if cl_target is None else cl_target)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, -2.0, -1.5], dtype=np.float32),
            high=np.array([ 2.0,  2.0,  2.0,  1.5], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def _gust_velocity(self, t):
        return one_minus_cosine_gust(
            t,
            self.gust_start,
            self.gust_duration,
            self.gust_amplitude,
        )

    def _gust_angle_deg(self, t):
        w_g = self._gust_velocity(t)
        return np.rad2deg(np.arctan2(w_g, self.U_inf))

    def _equivalent_alpha_deg(self, gust_angle_deg):
        return (
            self.alpha0_deg
            + gust_angle_deg
            + self.flap_effectiveness * self.flap_deg
        )

    def _aero(self, gust_angle_deg):
        alpha_eq_deg = self._equivalent_alpha_deg(gust_angle_deg)

        cl_qs, cd = self.aero_fn(
            self.reynolds,
            alpha_eq_deg,
            self.naca_m,
            self.naca_p,
            self.naca_t,
        )

        return cl_qs, cd, alpha_eq_deg

    def _get_obs(self):
        gust_angle_deg = self._gust_angle_deg(self.t)

        obs = np.array([
            self.cl / 1.5,
            self.cl_target / 1.5,
            gust_angle_deg / 10.0,
            self.flap_deg / self.max_flap_deg,
        ], dtype=np.float32)

        return np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.t = 0.0
        self.prev_action = 0.0
        self.flap_deg = 0.0

        if self.random_gust:
            self.gust_amplitude = self.np_random.uniform(
                *self.gust_amplitude_range
            )
            self.gust_duration = self.np_random.uniform(
                *self.gust_duration_range
            )
            self.gust_start = self.np_random.uniform(
                *self.gust_start_range
            )
        else:
            self.gust_amplitude = 2.5
            self.gust_duration = 1.3
            self.gust_start = 2.0

        self.cl, self.cd = self.aero_fn(
            self.reynolds,
            self.alpha0_deg,
            self.naca_m,
            self.naca_p,
            self.naca_t,
        )

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_value = float(np.clip(action[0], -1.0, 1.0))

        flap_command_deg = action_value * self.max_flap_deg

        # First-order actuator dynamics.
        self.flap_deg += (
            self.dt / self.tau_flap
        ) * (flap_command_deg - self.flap_deg)

        gust_angle_deg = self._gust_angle_deg(self.t)

        cl_qs, cd, alpha_eq_deg = self._aero(gust_angle_deg)

        # First-order lift dynamics.
        self.cl += (
            self.dt / self.tau_lift
        ) * (cl_qs - self.cl)

        self.cd = cd

        tracking_error = self.cl - self.cl_target
        delta_action = action_value - self.prev_action

        reward = (
            -tracking_error**2
            -self.effort_weight * action_value**2
            -self.smoothness_weight * delta_action**2
            -self.drag_weight * cd
        )

        self.prev_action = action_value
        self.step_count += 1
        self.t = self.step_count * self.dt

        terminated = False
        truncated = self.step_count >= self.horizon

        info = {
            "time": self.t,
            "cl": self.cl,
            "cl_qs": cl_qs,
            "cl_target": self.cl_target,
            "cd": cd,
            "flap_deg": self.flap_deg,
            "flap_command_deg": flap_command_deg,
            "gust_velocity": self._gust_velocity(self.t),
            "gust_angle_deg": self._gust_angle_deg(self.t),
            "alpha_eq_deg": alpha_eq_deg,
            "tracking_error": tracking_error,
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )


class ZeroPolicy:
    """Flap held at zero (Stage 6, cell 9)."""

    def predict(self, obs, deterministic=True):
        return np.array([0.0], dtype=np.float32), None


def run_episode(env, policy, seed=0):
    """One episode with a random gust, logged step by step (Stage 6, cell 9)."""
    obs, _ = env.reset(seed=seed)

    data = {
        "time": [],
        "cl": [],
        "cl_qs": [],
        "cl_target": [],
        "cd": [],
        "flap_deg": [],
        "flap_command_deg": [],
        "gust_velocity": [],
        "gust_angle_deg": [],
        "alpha_eq_deg": [],
        "reward": [],
    }

    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        for key in data:
            if key == "reward":
                data[key].append(reward)
            else:
                data[key].append(info[key])

    return {
        key: np.asarray(value)
        for key, value in data.items()
    }


def fly(policy, amplitude, duration, start=2.0, env_cls=AirfoilSurrogateGymEnv, **env_kw):
    """One episode with a prescribed gust ``(amplitude, duration, start)`` (Stage 6, cell 12).

    ``env_kw`` must carry ``aero_fn`` (plus any other constructor argument).
    """
    env = env_cls(random_gust=False, **env_kw)
    obs, _ = env.reset(seed=0)
    env.gust_amplitude, env.gust_duration, env.gust_start = amplitude, duration, start
    obs = env._get_obs()
    keys = ["time", "cl", "cl_target", "cd", "flap_deg", "gust_angle_deg", "alpha_eq_deg"]
    out = {k: [] for k in keys + ["reward", "action"]}
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        for k in keys:
            out[k].append(info[k])
        out["reward"].append(reward); out["action"].append(float(np.clip(np.ravel(action)[0], -1, 1)))
    return {k: np.asarray(v) for k, v in out.items()}


#: The five gusts of Stage 6 (cells 11-12): the fixed gust of Stage 4 and its four unseen gusts, as (W0 m/s, Tg s).
SUITE = [(2.5, 1.3), (-3.5, 1.0), (-2.0, 1.7), (2.0, 1.1), (3.5, 1.6)]


def suite_metrics(policy, label, trust_range, gusts=SUITE, **fly_kw):
    """Scores of a policy over a suite of gusts, as one row (Stage 6, cell 12).

    ``trust_range`` is ``(TRUST_LO, TRUST_HI)``, the training range of the angle of attack,
    which the notebook reads from the coefficient table.
    """
    TRUST_LO, TRUST_HI = trust_range
    row = {"configuration": label}
    rm, fl, mfl, cds, rough, amin, amax, outside = [], [], [], [], [], [], [], []
    for W0, Tg in gusts:
        tr = fly(policy, W0, Tg, **fly_kw)
        m = episode_metrics(tr)
        rm.append(m["CL_RMSE"]); fl.append(m["max_abs_flap_deg"]); mfl.append(m["mean_abs_flap_deg"]); cds.append(m["mean_CD"])
        rough.append(np.mean(np.abs(np.diff(tr["action"]))))
        amin.append(tr["alpha_eq_deg"].min()); amax.append(tr["alpha_eq_deg"].max())
        outside.append(np.mean((tr["alpha_eq_deg"] < TRUST_LO) | (tr["alpha_eq_deg"] > TRUST_HI)))
        row[f"RMSE W0={W0:+.1f},Tg={Tg:.1f}"] = m["CL_RMSE"]
    row.update({"mean_RMSE": np.mean(rm), "max_abs_flap_deg": np.max(fl), "mean_abs_flap_deg": np.mean(mfl),
                "mean_CD": np.mean(cds), "action_roughness": np.mean(rough), "alpha_eq_min": np.min(amin),
                "alpha_eq_max": np.max(amax), "fraction_outside_trusted": np.mean(outside)})
    return row
