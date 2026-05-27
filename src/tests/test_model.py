"""Tests for model architecture."""

import jax
import jax.numpy as jnp
import pytest

from model import FuelEmbedding, MSCNNParameterGenerator, WildfireCA, WildfireModel, compute_wind_speed_direction


class TestComputeWindSpeedDirection:
    def test_basic(self):
        wu = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        wv = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
        speed, direction = compute_wind_speed_direction(wu, wv)
        assert speed.shape == (2, 2)
        assert jnp.allclose(speed[0, 0], 1.0)
        assert jnp.allclose(speed[0, 1], 1.0)

    def test_zero_wind(self):
        wu = jnp.zeros((3, 3))
        wv = jnp.zeros((3, 3))
        speed, _ = compute_wind_speed_direction(wu, wv)
        assert jnp.allclose(speed, 0.0)


class TestFuelEmbedding:
    def test_output_shape(self, tiny_config):
        key = jax.random.PRNGKey(0)
        embed = FuelEmbedding(tiny_config, key)
        fuel_map = jnp.zeros((8, 10), dtype=jnp.int32)
        result = embed(fuel_map)
        assert result.shape == (4, 8, 10)  # (embed_dim, H, W)

    def test_different_indices_different_embeddings(self, tiny_config):
        key = jax.random.PRNGKey(0)
        embed = FuelEmbedding(tiny_config, key)
        fuel_map = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        result = embed(fuel_map)
        # Different fuel types should produce different embeddings
        assert not jnp.allclose(result[:, 0, 0], result[:, 0, 1])


class TestMSCNNParameterGenerator:
    def test_output_shape_kernel3(self, tiny_config):
        """kernel_size=3 → single branch (Model A degradation)."""
        cfg = {**tiny_config}
        cfg['model'] = {**tiny_config['model'], 'kernel_size': 3}
        key = jax.random.PRNGKey(0)
        cnn = MSCNNParameterGenerator(cfg, key)
        x = jax.random.normal(key, (13, 16, 20))
        params = cnn(x)
        assert params['p_base'].shape == (16, 20)
        assert params['alpha_w1'].shape == (16, 20)
        assert params['alpha_w2'].shape == (16, 20)
        assert params['alpha_s'].shape == (16, 20)
        assert params['alpha_gamma'].shape == (16, 20)
        assert params['fuel_factor'].shape == (16, 20)

    def test_output_shape_kernel7(self, tiny_config):
        """kernel_size=7 → 3 branches (k=3,5,7)."""
        cfg = {**tiny_config}
        cfg['model'] = {**tiny_config['model'], 'kernel_size': 7}
        key = jax.random.PRNGKey(0)
        cnn = MSCNNParameterGenerator(cfg, key)
        assert len(cnn.conv_branches) == 3
        x = jax.random.normal(key, (13, 16, 20))
        params = cnn(x)
        assert params['p_base'].shape == (16, 20)
        assert params['fuel_factor'].shape == (16, 20)

    def test_activation_constraints(self, tiny_config):
        """p_base in (0,1), others > 0."""
        key = jax.random.PRNGKey(42)
        cnn = MSCNNParameterGenerator(tiny_config, key)
        x = jax.random.normal(key, (13, 8, 10))
        params = cnn(x)
        assert (params['p_base'] > 0).all()
        assert (params['p_base'] < 1).all()
        assert (params['alpha_w1'] >= 0).all()
        assert (params['alpha_w2'] >= 0).all()
        assert (params['alpha_s'] >= 0).all()
        assert (params['alpha_gamma'] >= 0).all()
        assert (params['fuel_factor'] >= 0).all()

    def test_single_branch_for_kernel3(self, tiny_config):
        """kernel_size=3 creates exactly 1 branch."""
        key = jax.random.PRNGKey(0)
        cnn = MSCNNParameterGenerator(tiny_config, key)
        assert len(cnn.conv_branches) == 1


class TestWildfireCA:
    def test_compute_wind_factor_shape(self, small_grid, tiny_config):
        H, W = small_grid['H'], small_grid['W']
        alpha_w1 = jnp.ones((H, W)) * 0.5
        alpha_w2 = jnp.ones((H, W)) * 0.5
        speed = jnp.ones((H, W))
        direction = jnp.zeros((H, W))
        result = WildfireCA.compute_wind_factor(alpha_w1, alpha_w2, speed, direction, 0.0)
        assert result.shape == (H, W)
        assert jnp.isfinite(result).all()

    def test_compute_slope_factor_shape(self, small_grid):
        H, W = small_grid['H'], small_grid['W']
        alpha_s = jnp.ones((H, W)) * 0.3
        dir_slope = jnp.zeros((H, W))
        result = WildfireCA.compute_slope_factor(alpha_s, dir_slope)
        assert result.shape == (H, W)
        assert jnp.allclose(result, 1.0)  # exp(0) = 1

    def test_compute_ignition_probability_shape(self, small_grid, tiny_config):
        key = jax.random.PRNGKey(0)
        cnn = MSCNNParameterGenerator(tiny_config, key)
        x = jax.random.normal(key, (13, small_grid['H'], small_grid['W']))
        ca_params = cnn(x)

        p_ignite = WildfireCA.compute_ignition_probability(
            small_grid['p_burning'], ca_params,
            jnp.ones_like(small_grid['slope_ca']),
            jnp.zeros_like(small_grid['slope_ca']),
            small_grid['slope_ca'], small_grid['aspect_upslope'],
        )
        assert p_ignite.shape == (small_grid['H'], small_grid['W'])
        assert (p_ignite >= 0).all()
        assert (p_ignite <= 1).all()

    def test_step_deterministic_conservation(self, small_grid, tiny_config):
        """States should approximately sum to 1 per cell."""
        key = jax.random.PRNGKey(0)
        cnn = MSCNNParameterGenerator(tiny_config, key)
        x = jax.random.normal(key, (13, small_grid['H'], small_grid['W']))
        ca_params = cnn(x)

        speed, direction = compute_wind_speed_direction(
            small_grid['wind_u'][0], small_grid['wind_v'][0]
        )

        # Set initial states that sum to 1
        p_u = 1.0 - small_grid['p_burning']
        p_b = small_grid['p_burning']
        p_bd = jnp.zeros_like(p_u)

        new_pu, new_pb, new_pbd = WildfireCA.step_deterministic(
            p_u, p_b, p_bd, ca_params,
            speed, direction,
            small_grid['slope_ca'], small_grid['aspect_upslope'],
            small_grid['fuel_mask'], 3,
        )

        total = new_pu + new_pb + new_pbd
        # Should be approximately 1 everywhere (clipping may cause small deviations)
        assert jnp.allclose(total, 1.0, atol=0.1)

    def test_fuel_mask_blocks_ignition(self, small_grid, tiny_config):
        """Cells with fuel_mask=0 should have zero ignition."""
        key = jax.random.PRNGKey(0)
        cnn = MSCNNParameterGenerator(tiny_config, key)
        x = jax.random.normal(key, (13, small_grid['H'], small_grid['W']))
        ca_params = cnn(x)

        speed, direction = compute_wind_speed_direction(
            small_grid['wind_u'][0], small_grid['wind_v'][0]
        )

        # Zero fuel mask everywhere
        fuel_mask = jnp.zeros((small_grid['H'], small_grid['W']))
        p_u = jnp.ones((small_grid['H'], small_grid['W']))
        p_b = jnp.zeros((small_grid['H'], small_grid['W'])).at[4, 5].set(1.0)
        p_bd = jnp.zeros_like(p_u)

        new_pu, new_pb, new_pbd = WildfireCA.step_deterministic(
            p_u, p_b, p_bd, ca_params,
            speed, direction,
            small_grid['slope_ca'], small_grid['aspect_upslope'],
            fuel_mask, 3,
        )
        # No new burning should occur (fuel mask blocks ignition)
        newly_burning = p_u - new_pu
        assert jnp.allclose(newly_burning, 0.0, atol=1e-6)


class TestWildfireModel:
    def test_generate_params(self, tiny_config, small_grid):
        key = jax.random.PRNGKey(0)
        model = WildfireModel(tiny_config, key)
        params = model.generate_params(
            small_grid['static_cnn_input'],
            small_grid['wind_u'][0],
            small_grid['wind_v'][0],
            small_grid['fuel_type_map'],
        )
        assert 'p_base' in params
        assert 'fuel_factor' in params
        assert params['p_base'].shape == (small_grid['H'], small_grid['W'])
        assert params['fuel_factor'].shape == (small_grid['H'], small_grid['W'])

    def test_is_equinox_module(self, tiny_config):
        import equinox as eqx
        key = jax.random.PRNGKey(0)
        model = WildfireModel(tiny_config, key)
        assert isinstance(model, eqx.Module)

    def test_has_fuel_embedding(self, tiny_config):
        key = jax.random.PRNGKey(0)
        model = WildfireModel(tiny_config, key)
        assert hasattr(model, 'fuel_embedding')
        assert isinstance(model.fuel_embedding, FuelEmbedding)
