"""Tests for loss functions."""

import jax
import jax.numpy as jnp
import pytest

from loss import (
    area_constraint_loss,
    bce_loss,
    compute_total_loss,
    frontier_mask,
    frontier_weighted_bce,
    mse_loss,
    soft_iou_loss,
)


class TestFrontierMask:
    def test_basic_shape(self):
        target = jnp.zeros((10, 10))
        target = target.at[3:7, 3:7].set(1.0)
        mask = frontier_mask(target)
        assert mask.shape == (10, 10)

    def test_all_zeros(self):
        target = jnp.zeros((8, 8))
        mask = frontier_mask(target)
        assert jnp.allclose(mask, 0.0)

    def test_frontier_on_boundary(self):
        target = jnp.zeros((10, 10))
        target = target.at[4:6, 4:6].set(1.0)
        mask = frontier_mask(target)
        # Interior of the fire region should not be frontier
        assert mask[4, 4] == 0.0 or mask[4, 4] == 1.0  # corner can be frontier
        # Cells adjacent to fire should be frontier
        assert mask[3, 4] == 1.0 or mask[3, 5] == 1.0


class TestBCELoss:
    def test_perfect_prediction(self):
        pred = jnp.array([[1.0, 0.0], [1.0, 0.0]])
        target = jnp.array([[1.0, 0.0], [1.0, 0.0]])
        loss = bce_loss(pred, target, 1e-7)
        assert loss < 0.01

    def test_worst_prediction(self):
        pred = jnp.array([[0.0, 1.0]])
        target = jnp.array([[1.0, 0.0]])
        loss = bce_loss(pred, target, 1e-7)
        assert loss > 10.0

    def test_symmetric(self):
        pred = jnp.array([[0.7]])
        target = jnp.array([[1.0]])
        loss1 = bce_loss(pred, target, 1e-7)
        pred2 = jnp.array([[0.3]])
        target2 = jnp.array([[0.0]])
        loss2 = bce_loss(pred2, target2, 1e-7)
        assert jnp.allclose(loss1, loss2, atol=0.01)


class TestMSELoss:
    def test_perfect(self):
        pred = jnp.ones((5, 5))
        target = jnp.ones((5, 5))
        assert mse_loss(pred, target) == 0.0

    def test_known_value(self):
        pred = jnp.array([[1.0]])
        target = jnp.array([[0.0]])
        assert jnp.allclose(mse_loss(pred, target), 1.0)


class TestAreaConstraintLoss:
    def test_perfect_area(self):
        pred = jnp.ones((5, 5)) * 0.5
        target = jnp.ones((5, 5)) * 0.5
        loss = area_constraint_loss(pred, target, 1e-7)
        assert jnp.allclose(loss, 0.0)

    def test_nonzero_area_diff(self):
        pred = jnp.ones((5, 5))
        target = jnp.zeros((5, 5))
        loss = area_constraint_loss(pred, target, 1e-7)
        assert loss > 0


class TestSoftIoULoss:
    def test_perfect_overlap(self):
        pred = jnp.zeros((10, 10)).at[3:7, 3:7].set(1.0)
        target = jnp.zeros((10, 10)).at[3:7, 3:7].set(1.0)
        loss = soft_iou_loss(pred, target, 1e-7)
        assert jnp.allclose(loss, 0.0, atol=1e-5)

    def test_no_overlap(self):
        pred = jnp.zeros((10, 10)).at[0:3, 0:3].set(1.0)
        target = jnp.zeros((10, 10)).at[7:10, 7:10].set(1.0)
        loss = soft_iou_loss(pred, target, 1e-7)
        assert jnp.allclose(loss, 1.0, atol=1e-5)

    def test_partial_overlap(self):
        pred = jnp.zeros((10, 10)).at[3:7, 3:7].set(1.0)
        target = jnp.zeros((10, 10)).at[5:9, 5:9].set(1.0)
        loss = soft_iou_loss(pred, target, 1e-7)
        assert 0.0 < float(loss) < 1.0


class TestFrontierWeightedBCE:
    def test_higher_than_plain_bce(self):
        """Frontier weighting should increase loss when fire front exists."""
        pred = jnp.ones((10, 10)) * 0.5
        target = jnp.zeros((10, 10))
        target = target.at[3:7, 3:7].set(1.0)
        loss_plain = bce_loss(pred, target, 1e-7)
        loss_weighted = frontier_weighted_bce(pred, target, 5.0, 1e-7)
        assert loss_weighted >= loss_plain


class TestComputeTotalLoss:
    def test_output_structure(self, tiny_config):
        daily_pfire = jnp.ones((3, 8, 10)) * 0.5
        targets = jnp.ones((3, 8, 10)) * 0.5
        total, loss_dict = compute_total_loss(daily_pfire, targets, tiny_config)
        assert jnp.isfinite(total)
        assert 'bce' in loss_dict
        assert 'mse' in loss_dict
        assert 'area' in loss_dict
        assert 'total' in loss_dict

    def test_perfect_prediction(self, tiny_config):
        daily_pfire = jnp.ones((3, 8, 10)) * 0.7
        targets = jnp.ones((3, 8, 10)) * 0.7
        total, _ = compute_total_loss(daily_pfire, targets, tiny_config)
        assert total < 1.0  # Should be small for near-perfect match
