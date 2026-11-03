import numpy as np
import pytest

from airfoil_rom import preprocessing as pp


def test_channel_statistics_shapes_and_values(tiny_fields):
    mean, std = pp.channel_statistics(tiny_fields)
    assert mean.shape == (1, 3, 1, 1) and std.shape == (1, 3, 1, 1)
    np.testing.assert_allclose(mean.ravel(), tiny_fields.mean(axis=(0, 2, 3)), rtol=1e-6)
    assert np.all(std > 0)


def test_standardise_roundtrip_and_unit_variance(tiny_fields):
    mean, std = pp.channel_statistics(tiny_fields)
    scaled = pp.standardise_fields(tiny_fields, mean, std)
    np.testing.assert_allclose(scaled.mean(axis=(0, 2, 3)), 0.0, atol=1e-5)
    np.testing.assert_allclose(scaled.std(axis=(0, 2, 3)), 1.0, rtol=1e-5)
    back = pp.destandardise_fields(scaled, mean, std)
    np.testing.assert_allclose(back, tiny_fields, rtol=1e-5, atol=1e-6)


def test_std_floor_on_constant_channel():
    fields = np.zeros((2, 1, 4, 4), dtype=np.float32)
    _, std = pp.channel_statistics(fields)
    assert std.ravel()[0] == pytest.approx(1.0e-8)


def test_channel_layout_roundtrip(tiny_fields):
    images = pp.to_channels_last(tiny_fields)
    assert images.shape == (4, 16, 32, 3) and images.dtype == np.float32
    np.testing.assert_array_equal(pp.to_channels_first(images), tiny_fields)


def test_prepare_parameters_logs_reynolds_only():
    params = np.array([[0.02, 0.4, 0.12, 4.0e6, 5.0], [0.0, 0.0, 0.05, 1.0e6, -3.0]])
    prepared = pp.prepare_parameters(params)
    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared[:, 3], np.log10(params[:, 3]), rtol=1e-6)
    np.testing.assert_allclose(prepared[:, [0, 1, 2, 4]], params[:, [0, 1, 2, 4]], rtol=1e-6)
    assert params[0, 3] == 4.0e6           # the input is not modified


def test_candidate_grid_covers_the_box():
    lo = np.array([0.0, 0.0, 0.05, 2.0e6, -5.0])
    hi = np.array([0.07, 0.7, 0.2, 6.0e6, 15.0])
    cands, (gm, gp, gt) = pp.make_candidate_grid(lo, hi, n_m=3, n_p=4, n_t=5)
    assert cands.shape == (60, 5) and cands.dtype == np.float32
    assert cands[:, 0].min() == pytest.approx(lo[0]) and cands[:, 0].max() == pytest.approx(hi[0])
    assert cands[:, 2].min() == pytest.approx(lo[2]) and cands[:, 2].max() == pytest.approx(hi[2])
    assert np.all(cands[:, 3] == 4.0e6) and np.all(cands[:, 4] == 5.0)
    assert len(gm) == 3 and len(gp) == 4 and len(gt) == 5
    # ordering is (m, p, t) with t fastest, as in the notebook's reshape(15, 15, 16)
    np.testing.assert_allclose(cands[:5, 2], gt)


def test_digit_conversions_are_inverse():
    raw = pp.digits_to_chain_inputs(4.0e6, np.array([-1.0, 5.0]), m=2.0, p=4.0, t=12.0)
    assert raw.shape == (2, 5)
    np.testing.assert_allclose(raw[0], [0.02, 0.4, 0.12, np.log10(4.0e6), -1.0])
    # Stage 5 row (m, p, t as fractions, Re, alpha) -> Stage 3 inputs
    stage3 = pp.chain_inputs_to_stage3(np.array([[0.02, 0.4, 0.12, 4.0e6, 5.0]]))
    np.testing.assert_allclose(stage3[0], [4.0, 5.0, 2.0, 4.0, 12.0])
