"""Tests for simulation loops."""

import jax
import jax.numpy as jnp
import pytest

from model import WildfireModel
from simulation import run_simulation, run_simulation_no_grad


class TestRunSimulation:
    def test_output_shape(self, tiny_config, small_grid):
        key = jax.random.PRNGKey(0)
        model = WildfireModel(tiny_config, key)

        # 3 days of wind → 3 days output
        daily_pfire, final_state = run_simulation(
            model,
            small_grid['p_unburned'],
            small_grid['p_burning'],
            small_grid['p_burned'],
            small_grid['static_cnn_input'],
            small_grid['slope_ca'],
            small_grid['aspect_upslope'],
            small_grid['fuel_type_map'],
            small_grid['wind_u'],
            small_grid['wind_v'],
            small_grid['fuel_mask'],
            tiny_config,
        )
        assert daily_pfire.shape == (3, small_grid['H'], small_grid['W'])
        # Final state should be a tuple of 3 arrays
        assert len(final_state) == 3
        for s in final_state:
            assert s.shape == (small_grid['H'], small_grid['W'])

    def test_pfire_range(self, tiny_config, small_grid):
        key = jax.random.PRNGKey(0)
        model = WildfireModel(tiny_config, key)

        daily_pfire, _ = run_simulation(
            model,
            small_grid['p_unburned'],
            small_grid['p_burning'],
            small_grid['p_burned'],
            small_grid['static_cnn_input'],
            small_grid['slope_ca'],
            small_grid['aspect_upslope'],
            small_grid['fuel_type_map'],
            small_grid['wind_u'],
            small_grid['wind_v'],
            small_grid['fuel_mask'],
            tiny_config,
        )
        assert (daily_pfire >= 0).all()
        assert (daily_pfire <= 1).all()

    def test_fuel_mask_prevents_spread(self, tiny_config, small_grid):
        """With zero fuel mask, fire should not spread."""
        key = jax.random.PRNGKey(0)
        model = WildfireModel(tiny_config, key)

        zero_mask = jnp.zeros((small_grid['H'], small_grid['W']))
        p_u = 1.0 - small_grid['p_burning']

        daily_pfire, _ = run_simulation(
            model, p_u, small_grid['p_burning'], small_grid['p_burned'],
            small_grid['static_cnn_input'], small_grid['slope_ca'],
            small_grid['aspect_upslope'], small_grid['fuel_type_map'],
            small_grid['wind_u'], small_grid['wind_v'],
            zero_mask, tiny_config,
        )
        # Initial fire cells remain, but no new spread
        initial_pfire = 1.0 - p_u
        # Fire should only decrease or stay same (burning → burned transition)
        # but not increase in area
        for d in range(daily_pfire.shape[0]):
            assert daily_pfire[d].sum() <= initial_pfire.sum() + 1.0

    def test_no_grad_matches(self, tiny_config, small_grid):
        """run_simulation_no_grad should produce same output."""
        key = jax.random.PRNGKey(0)
        model = WildfireModel(tiny_config, key)

        args = (
            model, small_grid['p_unburned'], small_grid['p_burning'],
            small_grid['p_burned'], small_grid['static_cnn_input'],
            small_grid['slope_ca'], small_grid['aspect_upslope'],
            small_grid['fuel_type_map'],
            small_grid['wind_u'], small_grid['wind_v'],
            small_grid['fuel_mask'], tiny_config,
        )

        pfire1, _ = run_simulation(*args)
        pfire2 = run_simulation_no_grad(*args)
        assert jnp.allclose(pfire1, pfire2, atol=1e-5)


class TestSimulationWithRemat:
    def test_remat_produces_same_output(self, tiny_config, small_grid):
        cfg_remat = {**tiny_config}
        cfg_remat['training'] = {**tiny_config['training'], 'use_remat': True}

        key = jax.random.PRNGKey(0)
        model = WildfireModel(tiny_config, key)

        args = (
            model, small_grid['p_unburned'], small_grid['p_burning'],
            small_grid['p_burned'], small_grid['static_cnn_input'],
            small_grid['slope_ca'], small_grid['aspect_upslope'],
            small_grid['fuel_type_map'],
            small_grid['wind_u'], small_grid['wind_v'],
            small_grid['fuel_mask'],
        )

        pfire_no_remat, _ = run_simulation(*args, tiny_config)
        pfire_remat, _ = run_simulation(*args, cfg_remat)
        assert jnp.allclose(pfire_no_remat, pfire_remat, atol=1e-5)
