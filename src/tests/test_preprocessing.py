"""Tests for preprocessing functions."""

import numpy as np
import pytest

from preprocessing import (
    compute_fuel_mask,
    convert_aspect_to_upslope_radians,
    preprocess_canopy_cnn,
    preprocess_elevation,
    preprocess_fbfm40,
    preprocess_slope_aspect_cnn,
    preprocess_slope_ca,
    preprocess_wind,
)


class TestPreprocessElevation:
    def test_output_range(self, tiny_config):
        elev = np.array([[100.0, 200.0], [300.0, 400.0]])
        result = preprocess_elevation(elev, tiny_config)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_nodata_zeroed(self, tiny_config):
        elev = np.array([[100.0, -9999.0], [200.0, 300.0]])
        result = preprocess_elevation(elev, tiny_config)
        assert result[0, 1] == 0.0

    def test_constant_elevation(self, tiny_config):
        elev = np.array([[500.0, 500.0], [500.0, 500.0]])
        result = preprocess_elevation(elev, tiny_config)
        assert result.shape == (2, 2)


class TestPreprocessSlopeAspectCNN:
    def test_output_shapes(self, tiny_config):
        slope = np.array([[100.0, 200.0], [0.0, 450.0]])
        aspect = np.array([[0.0, 900.0], [-1.0, 1800.0]])
        sin_s, sin_a, cos_a = preprocess_slope_aspect_cnn(slope, aspect, tiny_config)
        assert sin_s.shape == (2, 2)
        assert sin_a.shape == (2, 2)
        assert cos_a.shape == (2, 2)

    def test_flat_cells_zeroed(self, tiny_config):
        slope = np.array([[100.0], [200.0]])
        aspect = np.array([[-1.0], [900.0]])
        _, sin_a, cos_a = preprocess_slope_aspect_cnn(slope, aspect, tiny_config)
        assert sin_a[0, 0] == 0.0
        assert cos_a[0, 0] == 0.0


class TestPreprocessSlopeCA:
    def test_output_range(self, tiny_config):
        slope = np.array([[0.0, 450.0], [900.0, 100.0]])
        result = preprocess_slope_ca(slope, tiny_config)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_zero_slope(self, tiny_config):
        slope = np.array([[0.0]])
        result = preprocess_slope_ca(slope, tiny_config)
        assert result[0, 0] == 0.0


class TestConvertAspect:
    def test_flat_cells(self, tiny_config):
        aspect = np.array([[-1.0, 0.0]])
        result = convert_aspect_to_upslope_radians(aspect, tiny_config)
        assert result[0, 0] == 0.0

    def test_output_range(self, tiny_config):
        aspect = np.array([[0.0, 900.0, 1800.0, 2700.0]])
        result = convert_aspect_to_upslope_radians(aspect, tiny_config)
        assert result.min() >= 0.0
        assert result.max() < 2 * np.pi + 0.01


class TestPreprocessCanopyCNN:
    def test_output_shapes(self, tiny_config):
        cbd = np.array([[10.0, 20.0], [0.0, 30.0]])
        cc = np.array([[50.0, 100.0], [0.0, 75.0]])
        ch = np.array([[5.0, 10.0], [0.0, 15.0]])
        cbd_n, cc_n, ch_n = preprocess_canopy_cnn(cbd, cc, ch, tiny_config)
        assert cbd_n.shape == (2, 2)
        assert cc_n.shape == (2, 2)
        assert ch_n.shape == (2, 2)

    def test_max_normalization(self, tiny_config):
        cbd = np.array([[45.0]])  # density_max = 45
        cc = np.array([[100.0]])  # canopy_max = 100
        ch = np.array([[550.0]])  # canopy_height_max = 550
        cbd_n, cc_n, ch_n = preprocess_canopy_cnn(cbd, cc, ch, tiny_config)
        np.testing.assert_allclose(cbd_n[0, 0], 1.0)
        np.testing.assert_allclose(cc_n[0, 0], 1.0)
        np.testing.assert_allclose(ch_n[0, 0], 1.0)


class TestPreprocessFBFM40:
    def test_output_type(self):
        fbfm = np.array([[1, 91, 165], [202, 0, 99]])
        result = preprocess_fbfm40(fbfm)
        assert result.dtype == np.int32

    def test_clipping(self):
        fbfm = np.array([[-5, 300]])
        result = preprocess_fbfm40(fbfm)
        assert result[0, 0] == 0
        assert result[0, 1] == 202


class TestPreprocessWind:
    def test_output_range(self):
        wu = np.array([[[5.0, -3.0], [10.0, -10.0]]])
        wv = np.array([[[2.0, -2.0], [8.0, -8.0]]])
        wu_s, wv_s = preprocess_wind(wu, wv, 10.0)
        assert wu_s.min() >= -1.0
        assert wu_s.max() <= 1.0
        assert wv_s.min() >= -1.0
        assert wv_s.max() <= 1.0

    def test_nan_replaced(self):
        wu = np.array([[[np.nan, 5.0]]])
        wv = np.array([[[3.0, np.nan]]])
        wu_s, wv_s = preprocess_wind(wu, wv, 10.0)
        assert np.isfinite(wu_s).all()
        assert np.isfinite(wv_s).all()


class TestComputeFuelMask:
    def test_burnable_cells(self, tiny_config):
        fbfm = np.array([[1, 50], [165, 202]])
        mask = compute_fuel_mask(fbfm, tiny_config)
        assert mask[0, 0] == 1.0
        assert mask[0, 1] == 1.0
        assert mask[1, 0] == 1.0
        assert mask[1, 1] == 1.0

    def test_non_burnable_cells(self, tiny_config):
        fbfm = np.array([[91, 95], [99, 50]])
        mask = compute_fuel_mask(fbfm, tiny_config)
        assert mask[0, 0] == 0.0  # 91 = non-burnable
        assert mask[0, 1] == 0.0  # 95 = non-burnable
        assert mask[1, 0] == 0.0  # 99 = non-burnable
        assert mask[1, 1] == 1.0  # 50 = burnable
