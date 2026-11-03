"""The package must behave exactly like the inline definitions of the notebooks.

Each test parses a notebook cell (JSON, never executed), keeps only its ``def`` / ``class``
statements, executes those in a controlled namespace, and compares the result with the
package on small synthetic inputs.
"""
import ast
import types

import numpy as np
import pytest

from airfoil_rom import control as ct, metrics as mt, models as md, preprocessing as pp
from conftest import linear_aero


def exec_defs(nb, cell_index, names, namespace):
    """Execute only the function/class definitions ``names`` of code cell ``cell_index``."""
    cell = nb["cells"][cell_index]
    assert cell["cell_type"] == "code"
    tree = ast.parse("".join(cell["source"]))
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in keep} == set(names), f"cell {cell_index} does not define {names}"
    code = compile(ast.Module(body=keep, type_ignores=[]), f"<notebook cell {cell_index}>", "exec")
    ns = dict(namespace)
    exec(code, ns)
    return ns


@pytest.fixture(scope="module")
def tf():
    return pytest.importorskip("tensorflow")


@pytest.fixture(scope="module")
def keras_ns(tf):
    from tensorflow import keras
    from tensorflow.keras import layers
    return {"np": np, "tf": tf, "keras": keras, "layers": layers}


def test_prepare_parameters(notebooks):
    ns = exec_defs(notebooks["05"], 29, ["prepare_parameters"], {"np": np})
    params = np.array([[0.02, 0.4, 0.12, 4.0e6, 5.0], [0.07, 0.7, 0.05, 2.0e6, -4.9]])
    np.testing.assert_array_equal(ns["prepare_parameters"](params), pp.prepare_parameters(params))


def test_beta_vae_architecture_and_loss(notebooks, tf, keras_ns):
    ns = exec_defs(notebooks["05"], 23, ["Sampling", "build_beta_vae", "beta_vae_loss"], keras_ns)
    batch = np.random.default_rng(0).normal(size=(2, 16, 32, 3)).astype(np.float32)

    tf.keras.utils.set_random_seed(0); enc_nb, dec_nb = ns["build_beta_vae"]((16, 32, 3), 8)
    tf.keras.utils.set_random_seed(0); enc_pk, dec_pk = md.build_beta_vae((16, 32, 3), 8)
    assert enc_nb.count_params() == enc_pk.count_params() and dec_nb.count_params() == dec_pk.count_params()
    for a, b in zip(enc_nb.get_weights() + dec_nb.get_weights(), enc_pk.get_weights() + dec_pk.get_weights()):
        np.testing.assert_array_equal(a, b)
    mu_nb, lv_nb, _ = enc_nb(batch, training=False); mu_pk, lv_pk, _ = enc_pk(batch, training=False)
    np.testing.assert_allclose(mu_nb, mu_pk, rtol=1e-6); np.testing.assert_allclose(lv_nb, lv_pk, rtol=1e-6)
    np.testing.assert_allclose(dec_nb(mu_nb, training=False), dec_pk(mu_pk, training=False), rtol=1e-6)

    rec = tf.constant(batch); target = tf.constant(batch[::-1].copy())
    for a, b in zip(ns["beta_vae_loss"](rec, target, mu_nb, lv_nb, 1e-3), md.beta_vae_loss(rec, target, mu_pk, lv_pk, 1e-3)):
        assert float(a) == pytest.approx(float(b), rel=1e-6)


def test_beta_warmup_schedule(notebooks):
    # the schedule is an expression inside the training loop of cell 24, not a function: re-read it from the source
    src = "".join(notebooks["05"]["cells"][24]["source"])
    assert "beta = BETA_MAX * min(1.0, (epoch + 1) / BETA_WARMUP_EPOCHS)" in src
    assert "BETA_MAX = 1.0e-3" in src and "BETA_WARMUP_EPOCHS = 30" in src
    for epoch in (0, 10, 29, 30, 99):
        assert md.beta_schedule(epoch) == 1.0e-3 * min(1.0, (epoch + 1) / 30)


def test_surrogate_and_head(notebooks, tf, keras_ns):
    ns = exec_defs(notebooks["05"], 30, ["build_latent_surrogate"], keras_ns)
    ns.update(exec_defs(notebooks["05"], 34, ["build_aerodynamic_head"], keras_ns))
    tf.keras.utils.set_random_seed(1); s_nb = ns["build_latent_surrogate"](5, 8); h_nb = ns["build_aerodynamic_head"](8)
    tf.keras.utils.set_random_seed(1); s_pk = md.build_latent_surrogate(5, 8); h_pk = md.build_aerodynamic_head(8)
    def shape(model):     # layer auto-names differ between builds; compare class, units and activation
        return [(l["class_name"], l["config"].get("units"), l["config"].get("activation")) for l in model.get_config()["layers"]]
    assert shape(s_nb) == shape(s_pk) and h_nb.name == h_pk.name
    assert shape(h_nb) == shape(h_pk) and s_nb.name == s_pk.name
    for a, b in zip(s_nb.get_weights() + h_nb.get_weights(), s_pk.get_weights() + h_pk.get_weights()):
        np.testing.assert_array_equal(a, b)


def test_scores_nearest_timed(notebooks):
    rng = np.random.default_rng(0)
    true = rng.normal(size=(10, 2)) * [0.5, 0.005] + [0.7, 0.01]; pred = true + rng.normal(size=(10, 2)) * [0.02, 0.0005]
    ns = exec_defs(notebooks["05"], 63, ["scores"], {"np": np, "test_coefficients": true,
                                                     "mean_absolute_error": mt.mean_absolute_error, "r2_score": mt.r2_score})
    assert ns["scores"](pred, "x", seed=3) == mt.coefficient_scores(pred, true, "x", seed=3)

    S_train = rng.normal(size=(7, 5)); S = rng.normal(size=(3, 5))
    ns = exec_defs(notebooks["05"], 72, ["nearest"], {"np": np, "S_train": S_train})
    for idx in (slice(None), slice(0, 3)):
        d_nb, j_nb = ns["nearest"](S, idx); d_pk, j_pk = mt.nearest_distance(S, S_train, idx)
        np.testing.assert_array_equal(d_nb, d_pk); np.testing.assert_array_equal(j_nb, j_pk)

    ns = exec_defs(notebooks["05"], 82, ["timed"], {"np": np, "time": __import__("time")})
    calls = []
    assert len(ns["timed"](lambda: calls.append(1), repeats=2)) == 2 and len(calls) == 3


def test_region_bands(notebooks):
    src = "".join(notebooks["05"]["cells"][57]["source"])
    assert 'bands = [("inside the body", -9, 0.0), ("0-0.02 c", 0.0, 0.02), ("0.02-0.1 c", 0.02, 0.1), ("beyond 0.1 c", 0.1, 9)]' in src
    assert mt.REGION_BANDS == [("inside the body", -9, 0.0), ("0-0.02 c", 0.0, 0.02), ("0.02-0.1 c", 0.02, 0.1), ("beyond 0.1 c", 0.1, 9)]


def test_gust_episode_metrics_and_dense_stack(notebooks):
    ns = exec_defs(notebooks["06"], 7, ["one_minus_cosine_gust"], {"np": np})
    for t in np.linspace(0, 6, 61):
        assert ns["one_minus_cosine_gust"](t, 2.0, 1.3, 2.5) == ct.one_minus_cosine_gust(t, 2.0, 1.3, 2.5)
    ns = exec_defs(notebooks["06"], 10, ["episode_metrics"], {"np": np})
    data = {k: np.random.default_rng(1).normal(size=20) for k in ("cl", "cl_target", "flap_deg", "cd", "reward")}
    assert ns["episode_metrics"](data) == mt.episode_metrics(data)
    ns = exec_defs(notebooks["06"], 16, ["dense_stack", "run_stack"], {"np": np})
    stack = [(np.eye(3), np.ones(3), "tanh"), (np.ones((3, 2)), np.zeros(2), "linear")]
    np.testing.assert_array_equal(ns["run_stack"](stack, np.arange(3.0)), md.run_stack(stack, np.arange(3.0)))


def test_numpy_chain_matches_notebook_l12_fast(notebooks, tf, keras_ns):
    """``l12_fast`` of cell 16 reads its weights from module globals; rebuild those globals from tiny models."""
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(2)
    tf.keras.utils.set_random_seed(2)
    surrogate = md.build_latent_surrogate(5, 8); head = md.build_aerodynamic_head(8)
    p_scaler = StandardScaler().fit(rng.normal(size=(20, 5)) + [0.03, 0.4, 0.12, 6.6, 5])
    z_scaler = StandardScaler().fit(rng.normal(size=(20, 8)) * 3)
    c_scaler = StandardScaler().fit(rng.normal(size=(20, 2)) * [0.5, 0.005] + [0.7, 0.01])
    ns = exec_defs(notebooks["06"], 16, ["dense_stack", "run_stack", "l12_fast"], {"np": np})
    ns.update({"_sur": ns["dense_stack"](surrogate), "_head": ns["dense_stack"](head), "_pm": p_scaler.mean_, "_ps": p_scaler.scale_,
               "_zm": z_scaler.mean_, "_zs": z_scaler.scale_, "_cm": c_scaler.mean_, "_cs": c_scaler.scale_, "_scale": 2.1})
    chain = md.NumpyChain(surrogate, head, 2.1, p_scaler, z_scaler, c_scaler)
    for q in rng.uniform([2e6, -5, 0, 0, 5], [6e6, 15, 7, 7, 20], size=(5, 5)):
        assert ns["l12_fast"](*q) == chain(*q)


def test_environment_and_suite_match_notebook(notebooks):
    """The notebook env calls a global ``predict_airfrans_aero``; the package env takes ``aero_fn``. Same trajectories."""
    gym = pytest.importorskip("gymnasium")
    from gymnasium import spaces
    base = {"np": np, "gym": gym, "spaces": spaces, "predict_airfrans_aero": linear_aero}
    ns = exec_defs(notebooks["06"], 7, ["one_minus_cosine_gust"], base)
    ns = exec_defs(notebooks["06"], 8, ["AirfoilSurrogateGymEnv"], ns)
    ns = exec_defs(notebooks["06"], 9, ["run_episode", "ZeroPolicy"], ns)
    ns = exec_defs(notebooks["06"], 10, ["episode_metrics"], ns)
    ns.update({"TRUST_LO": -4.91, "TRUST_HI": 14.84, "SUITE": ct.SUITE})
    ns = exec_defs(notebooks["06"], 12, ["fly", "suite_metrics"], ns)

    class Bang:                                   # a deterministic, non-trivial open-loop policy
        def __init__(self): self.k = 0
        def predict(self, obs, deterministic=True):
            self.k += 1; return np.array([np.sin(0.3 * self.k)], dtype=np.float32), None

    ep_nb = ns["run_episode"](ns["AirfoilSurrogateGymEnv"](), Bang(), seed=3)
    ep_pk = ct.run_episode(ct.AirfoilSurrogateGymEnv(aero_fn=linear_aero), Bang(), seed=3)
    assert ep_nb.keys() == ep_pk.keys()
    for k in ep_nb:
        np.testing.assert_array_equal(ep_nb[k], ep_pk[k])

    row_nb = ns["suite_metrics"](Bang(), "bang", gusts=ct.SUITE[:2])
    row_pk = ct.suite_metrics(Bang(), "bang", trust_range=(-4.91, 14.84), gusts=ct.SUITE[:2], aero_fn=linear_aero)
    assert row_nb == row_pk
    z_nb = ns["ZeroPolicy"]().predict(None); z_pk = ct.ZeroPolicy().predict(None)
    np.testing.assert_array_equal(z_nb[0], z_pk[0]) and z_nb[1] is None and z_pk[1] is None
