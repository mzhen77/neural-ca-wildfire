"""Loss functions for CNN-parameterized CA wildfire model."""

import jax
import jax.numpy as jnp


def frontier_mask(target):
    """Compute fire frontier mask via morphological dilation.

    Fire perimeter = dilated fire region minus fire region.
    Returns: (H, W) binary mask of frontier pixels.
    """
    target_4d = target[None, None, :, :]
    kernel = jnp.ones((1, 1, 3, 3))
    dilated = jax.lax.conv_general_dilated(
        target_4d, kernel,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    )[0, 0]
    dilated = (dilated > 0.5).astype(jnp.float32)
    frontier = dilated - (target > 0.5).astype(jnp.float32)
    return jnp.clip(frontier, 0.0, 1.0)


def bce_loss(pred, target, epsilon):
    """Binary cross-entropy loss."""
    pred_clipped = jnp.clip(pred, epsilon, 1.0 - epsilon)
    bce = -(target * jnp.log(pred_clipped) + (1.0 - target) * jnp.log(1.0 - pred_clipped))
    return jnp.mean(bce)


def frontier_weighted_bce(pred, target, frontier_weight, epsilon):
    """BCE with frontier pixels upweighted."""
    pred_clipped = jnp.clip(pred, epsilon, 1.0 - epsilon)
    bce = -(target * jnp.log(pred_clipped) + (1.0 - target) * jnp.log(1.0 - pred_clipped))
    front = frontier_mask(target)
    weights = jnp.where(front > 0.5, frontier_weight, 1.0)
    return jnp.mean(bce * weights)


def mse_loss(pred, target):
    """Mean squared error loss."""
    return jnp.mean((pred - target) ** 2)


def pooled_mse_loss(pred, target, pool_window):
    """Average-pooled MSE loss for coarse-grained spatial signal.

    Pools pred and target with average pooling before computing MSE.
    Provides smoother gradients for large-scale fire shape.
    """
    w = pool_window
    pool_kernel = jnp.ones((1, 1, w, w)) / (w * w)

    p_4d = pred[None, None, :, :]
    t_4d = target[None, None, :, :]

    p_pooled = jax.lax.conv_general_dilated(
        p_4d, pool_kernel,
        window_strides=(w, w),
        padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    )[0, 0]
    t_pooled = jax.lax.conv_general_dilated(
        t_4d, pool_kernel,
        window_strides=(w, w),
        padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    )[0, 0]

    return jnp.mean((p_pooled - t_pooled) ** 2)


def soft_iou_loss(pred, target, epsilon):
    """Soft IoU (Jaccard) loss: 1 - IoU.

    Directly measures shape overlap. A spherical prediction vs an elongated
    target will have low IoU → high loss, providing shape-aware gradients.

    IoU = sum(pred * target) / (sum(pred) + sum(target) - sum(pred * target) + eps)
    """
    intersection = jnp.sum(pred * target)
    union = jnp.sum(pred) + jnp.sum(target) - intersection
    iou = intersection / (union + epsilon)
    return 1.0 - iou


def area_constraint_loss(pred, target, epsilon):
    """Area constraint: penalize difference in total burned area.

    L_area = |sum(pred) - sum(target)| / (sum(target) + eps)
    """
    pred_area = jnp.sum(pred)
    true_area = jnp.sum(target)
    return jnp.abs(pred_area - true_area) / (true_area + epsilon)


def compute_day_loss(pred, target, config):
    """Compute combined loss for a single day.

    Args:
        pred: (H, W) predicted P_fire
        target: (H, W) target fire state
        config: config dict

    Returns:
        total_loss scalar
    """
    bce_weight = config['training']['bce_weight']
    mse_weight = config['training']['mse_weight']
    area_weight = config['training']['area_weight']
    frontier_wt = config['training']['frontier_weight']
    epsilon = config['training']['bce_epsilon']
    pool_window = config['training'].get('mse_pool_window', 0)

    iou_weight = config['training'].get('iou_weight', 0.0)

    l_bce = frontier_weighted_bce(pred, target, frontier_wt, epsilon)
    l_mse = mse_loss(pred, target)
    l_area = area_constraint_loss(pred, target, epsilon)

    total = bce_weight * l_bce + mse_weight * l_mse + area_weight * l_area

    if iou_weight > 0:
        l_iou = soft_iou_loss(pred, target, epsilon)
        total = total + iou_weight * l_iou

    if pool_window > 0:
        l_pool = pooled_mse_loss(pred, target, pool_window)
        total = total + mse_weight * l_pool

    return total


def compute_total_loss(daily_pfire, targets, config):
    """Compute combined loss across all days.

    Args:
        daily_pfire: (D, H, W) predicted P_fire at end of each day
        targets: (D, H, W) target fire states for days 1..D
        config: config dict

    Returns:
        (total_loss, loss_dict)
    """
    bce_weight = config['training']['bce_weight']
    mse_weight = config['training']['mse_weight']
    area_weight = config['training']['area_weight']
    frontier_wt = config['training']['frontier_weight']
    epsilon = config['training']['bce_epsilon']

    iou_weight = config['training'].get('iou_weight', 0.0)
    n_days = daily_pfire.shape[0]

    pool_window = config['training'].get('mse_pool_window', 0)

    def day_loss(d):
        pred = daily_pfire[d]
        tgt = targets[d]
        l_bce = frontier_weighted_bce(pred, tgt, frontier_wt, epsilon)
        l_mse = mse_loss(pred, tgt)
        l_area = area_constraint_loss(pred, tgt, epsilon)
        l_iou = soft_iou_loss(pred, tgt, epsilon) if iou_weight > 0 else 0.0
        l_pool = pooled_mse_loss(pred, tgt, pool_window) if pool_window > 0 else 0.0
        return l_bce, l_mse, l_area, l_iou, l_pool

    bce_vals, mse_vals, area_vals, iou_vals, pool_vals = jax.vmap(day_loss)(jnp.arange(n_days))

    mean_bce = jnp.mean(bce_vals)
    mean_mse = jnp.mean(mse_vals)
    mean_area = jnp.mean(area_vals)

    total = bce_weight * mean_bce + mse_weight * mean_mse + area_weight * mean_area

    if iou_weight > 0:
        mean_iou = jnp.mean(iou_vals)
        total = total + iou_weight * mean_iou

    if pool_window > 0:
        mean_pool = jnp.mean(pool_vals)
        total = total + mse_weight * mean_pool

    return total, {
        'bce': mean_bce,
        'mse': mean_mse,
        'area': mean_area,
        'total': total,
    }
