"""Tests for training loop."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model import WildfireModel
from training import (
    _event_to_jax,
    clip_gradients,
    create_optimizer,
    make_initial_state,
)
from simulation import run_one_day
from loss import compute_day_loss


class TestCreateOptimizer:
    def test_adam(self, tiny_config):
        opt = create_optimizer(tiny_config)
        assert opt is not None

    def test_adamw(self, tiny_config):
        cfg = {**tiny_config}
        cfg['training'] = {**tiny_config['training'], 'optimizer': 'adamw'}
        opt = create_optimizer(cfg)
        assert opt is not None

    def test_unknown_raises(self, tiny_config):
        cfg = {**tiny_config}
        cfg['training'] = {**tiny_config['training'], 'optimizer': 'sgd'}
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(cfg)


class TestMakeInitialState:
    def test_shapes(self):
        fire = jnp.zeros((10, 10)).at[3:5, 3:5].set(1.0)
        pu, pb, pbd = make_initial_state(fire)
        assert pu.shape == (10, 10)
        assert pb.shape == (10, 10)
        assert pbd.shape == (10, 10)

    def test_fire_cells_burning(self):
        fire = jnp.zeros((5, 5)).at[2, 2].set(1.0)
        pu, pb, pbd = make_initial_state(fire)
        assert pu[2, 2] == 0.0
        assert pb[2, 2] == 1.0
        assert pbd[2, 2] == 0.0

    def test_unburned_cells(self):
        fire = jnp.zeros((5, 5))
        pu, pb, pbd = make_initial_state(fire)
        assert jnp.allclose(pu, 1.0)
        assert jnp.allclose(pb, 0.0)
        assert jnp.allclose(pbd, 0.0)


class TestClipGradients:
    def test_clips_values(self):
        grads = {'a': jnp.array([5.0, -5.0, 0.5])}
        clipped = clip_gradients(grads, -1.0, 1.0)
        assert jnp.allclose(clipped['a'], jnp.array([1.0, -1.0, 0.5]))

    def test_handles_nan(self):
        grads = {'a': jnp.array([float('nan'), 1.0])}
        clipped = clip_gradients(grads, -1.0, 1.0)
        assert jnp.allclose(clipped['a'], jnp.array([0.0, 1.0]))


class TestPerDayTraining:
    def test_single_day_loss_and_grad(self, tiny_config, small_grid):
        """Verify single-day forward + backward produces finite loss and grads."""
        import equinox as eqx

        key = jax.random.PRNGKey(0)
        model = WildfireModel(tiny_config, key)

        H, W = small_grid['H'], small_grid['W']
        fire_seq = jnp.zeros((4, H, W))
        fire_seq = fire_seq.at[0, 3:5, 4:6].set(1.0)
        for d in range(1, 4):
            fire_seq = fire_seq.at[d, 2:6, 3:7].set(1.0)

        p_u, p_b, p_bd = make_initial_state(fire_seq[0])

        def loss_fn(m):
            p_fire, final_state = run_one_day(
                m, p_u, p_b, p_bd,
                small_grid['static_cnn_input'],
                small_grid['slope_ca'], small_grid['aspect_upslope'],
                small_grid['fuel_type_map'],
                small_grid['wind_u'][0], small_grid['wind_v'][0],
                small_grid['fuel_mask'], tiny_config,
            )
            loss = compute_day_loss(p_fire, fire_seq[1], tiny_config)
            return loss, final_state

        (loss, _), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        assert jnp.isfinite(loss)

        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if g is not None)
        assert has_nonzero, "All gradients are zero — no gradient flow"
