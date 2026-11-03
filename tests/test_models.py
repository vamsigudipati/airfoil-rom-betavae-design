import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from airfoil_rom import models as md

tf = pytest.importorskip("tensorflow")


@pytest.fixture(autouse=True)
def seed():
    tf.keras.utils.set_random_seed(0)


def test_beta_vae_forward_pass_2_samples():
    encoder, decoder = md.build_beta_vae(input_shape=(16, 32, 3), latent_dim=8)
    batch = np.random.default_rng(0).normal(size=(2, 16, 32, 3)).astype(np.float32)
    mu, logvar, z = encoder(batch, training=False)
    assert mu.shape == (2, 8) and logvar.shape == (2, 8) and z.shape == (2, 8)
    rec = decoder(z, training=False)
    assert rec.shape == (2, 16, 32, 3)          # three stride-2 stages down and up: 16 x 32 survives
    assert np.all(np.isfinite(rec.numpy()))
    assert [layer.name for layer in encoder.layers if layer.name in ("mu", "logvar", "sampling")] == ["mu", "logvar", "sampling"]


def test_sampling_layer_is_mu_plus_sigma_noise():
    Sampling = md.get_sampling_layer()
    mu = tf.zeros((1000, 2)); logvar = tf.fill((1000, 2), float(np.log(0.25)))   # sigma = 0.5
    z = Sampling()([mu, logvar]).numpy()
    assert abs(z.mean()) < 0.1 and abs(z.std() - 0.5) < 0.05


def test_beta_vae_loss_known_values():
    rec = tf.constant([[1.0, 2.0]]); target = tf.constant([[0.0, 0.0]])
    mu = tf.zeros((1, 2)); logvar = tf.zeros((1, 2))     # unit Gaussian posterior: KL = 0
    total, r, kl = md.beta_vae_loss(rec, target, mu, logvar, beta=0.5)
    assert float(r) == pytest.approx(2.5) and float(kl) == pytest.approx(0.0) and float(total) == pytest.approx(2.5)
    mu = tf.ones((1, 2))                                   # KL per coordinate = 0.5 * mu^2 = 0.5
    total, r, kl = md.beta_vae_loss(rec, target, mu, logvar, beta=0.5)
    assert float(kl) == pytest.approx(0.5) and float(total) == pytest.approx(2.75)


def test_beta_schedule_warmup():
    assert md.beta_schedule(0, 1e-3, 30) == pytest.approx(1e-3 / 30)
    assert md.beta_schedule(29, 1e-3, 30) == pytest.approx(1e-3)
    assert md.beta_schedule(99, 1e-3, 30) == pytest.approx(1e-3)


def test_surrogate_and_head_forward_pass_2_samples():
    surrogate = md.build_latent_surrogate(parameter_dim=5, latent_dim=8)
    head = md.build_aerodynamic_head(latent_dim=8)
    x = np.zeros((2, 5), dtype=np.float32)
    z = surrogate(x, training=False)
    assert z.shape == (2, 8)
    assert head(z, training=False).shape == (2, 2)
    assert surrogate.count_params() == 5 * 64 + 64 + 64 * 64 + 64 + 64 * 8 + 8
    assert head.count_params() == 8 * 32 + 32 + 32 * 2 + 2


def test_numpy_chain_matches_keras():
    rng = np.random.default_rng(1)
    surrogate = md.build_latent_surrogate(5, 8); head = md.build_aerodynamic_head(8)
    p_scaler = StandardScaler().fit(rng.normal(size=(20, 5)) * [0.02, 0.2, 0.05, 0.2, 5] + [0.03, 0.4, 0.12, 6.6, 5])
    z_scaler = StandardScaler().fit(rng.normal(size=(20, 8)) * 3)
    c_scaler = StandardScaler().fit(rng.normal(size=(20, 2)) * [0.5, 0.005] + [0.7, 0.01])
    chain = md.NumpyChain(surrogate, head, scale=2.1, parameter_scaler=p_scaler, z_scaler=z_scaler, coef_scaler=c_scaler)

    reynolds, alpha, m, p, t = 4.0e6, 5.0, 2.0, 4.0, 12.0
    raw = np.array([[m / 100, p / 10, t / 100, np.log10(reynolds), alpha]])
    z = surrogate.predict(p_scaler.transform(raw), verbose=0) * 2.1
    expected = c_scaler.inverse_transform(head.predict(z_scaler.transform(z), verbose=0))[0]
    np.testing.assert_allclose(chain(reynolds, alpha, m, p, t), expected, rtol=1e-5, atol=1e-6)


def test_run_stack_rejects_unknown_activation():
    with pytest.raises(ValueError):
        md.run_stack([(np.eye(2), np.zeros(2), "relu")], np.ones(2))
    np.testing.assert_allclose(md.run_stack([(np.eye(2) * 2, np.ones(2), "linear")], np.ones(2)), [3.0, 3.0])
