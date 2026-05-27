"""Training loop for CNN-parameterized CA wildfire model.

Matches prob_jax_real/training_modes.py single_day mode:
- Forward 1 day → loss → backward → per-element grad clip → optimizer update
- N_days updates per epoch (not gradient accumulation)
- Per-element gradient clipping [-1, 1] matching prob_jax_real
"""

import logging
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from loss import compute_day_loss, compute_total_loss
from simulation import run_one_day, run_simulation

logger = logging.getLogger(__name__)


def clip_gradients(grads, lo, hi):
    """Per-element gradient clipping with NaN handling. Matches prob_jax_real."""
    def clip_grad(g):
        if g is None:
            return g
        g = jnp.where(jnp.isnan(g), 0.0, g)
        return jnp.clip(g, lo, hi)
    return jax.tree_util.tree_map(clip_grad, grads)


def create_optimizer(config):
    """Create optax optimizer with constant LR. Matches prob_jax_real."""
    optimizer_name = config['training']['optimizer'].strip()
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    grad_clip_norm = config['training']['grad_clip_norm']

    if optimizer_name == "adamw":
        opt = optax.adamw(lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        opt = optax.adam(lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        opt,
    )


def make_initial_state(fire_state_0):
    """Create initial 3-state probabilities from day-0 fire map."""
    p_unburned = jnp.where(fire_state_0 > 0.5, 0.0, 1.0)
    p_burning = jnp.where(fire_state_0 > 0.5, 1.0, 0.0)
    p_burned = jnp.zeros_like(fire_state_0)
    return p_unburned, p_burning, p_burned


def _event_to_jax(event):
    """Convert FireEvent numpy arrays to jax arrays."""
    return {
        'static_cnn_input': jnp.array(event.static_cnn_input),
        'slope_ca': jnp.array(event.slope_ca),
        'aspect_upslope': jnp.array(event.aspect_upslope),
        'fuel_type_map': jnp.array(event.fuel_type_map),
        'wind_u': jnp.array(event.wind_u),
        'wind_v': jnp.array(event.wind_v),
        'fire_seq': jnp.array(event.fire_seq),
        'fuel_mask': jnp.array(event.fuel_mask),
    }


def validate(model, val_events, config):
    """Run validation on all events without gradients.

    Uses eqx.tree_inference to disable dropout during validation.
    Returns (avg_loss, per_event_ious) where per_event_ious
    is a dict {event_name: iou} based on new area at last day.
    """
    inference_model = eqx.tree_inference(model, value=True)
    total_loss = 0.0
    n_events = len(val_events)
    event_ious = {}
    Pth = config['training']['Pth']

    for event in val_events:
        fire_seq = jnp.array(event.fire_seq)
        initial_fire = fire_seq[0]
        targets = fire_seq[1:]

        p_u, p_b, p_bd = make_initial_state(initial_fire)

        static_cnn = jnp.array(event.static_cnn_input)
        slope_ca = jnp.array(event.slope_ca)
        aspect_up = jnp.array(event.aspect_upslope)
        fuel_type_map = jnp.array(event.fuel_type_map)
        wind_u = jnp.array(event.wind_u)
        wind_v = jnp.array(event.wind_v)
        fuel_mask = jnp.array(event.fuel_mask)

        daily_pfire, _ = jax.lax.stop_gradient(
            run_simulation(
                inference_model, p_u, p_b, p_bd,
                static_cnn, slope_ca, aspect_up,
                fuel_type_map, wind_u, wind_v,
                fuel_mask, config,
            )
        )

        loss, _ = compute_total_loss(daily_pfire, targets, config)
        total_loss += float(loss)

        # IoU on new area at last day (excluding initial territory)
        initial_mask = (np.array(initial_fire) > Pth).astype(float)
        pred_last = (np.clip(np.array(daily_pfire[-1]), 0.0, 1.0) > Pth).astype(float)
        target_last = (np.array(fire_seq[-1]) > Pth).astype(float)
        pred_new = pred_last * (1 - initial_mask)
        target_new = target_last * (1 - initial_mask)
        intersection = float((pred_new * target_new).sum())
        union = float(pred_new.sum() + target_new.sum()) - intersection
        iou = intersection / union if union > 0 else 0.0
        event_ious[event.event_name] = iou

    avg_loss = total_loss / max(n_events, 1)
    return avg_loss, event_ious


def train(model, train_events, val_events, config, out_dir):
    """Single-day training with equal event weighting.

    Per-day updates: forward 1 day → loss → backward → clip → update.
    Events are shuffled each epoch and loss is scaled by 1/event_days
    so every event contributes equally regardless of day count.
    """
    epochs = config['training']['epochs']
    print_interval = config['training']['print_interval']
    patience = config['training']['patience']
    start_save_epoch = config['training']['start_save_epoch']
    grad_clip = config['training']['grad_clip']
    grad_clip_lo = grad_clip['per_element'][0]
    grad_clip_hi = grad_clip['per_element'][1]

    optimizer = create_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    patience_counter = 0

    # Epoch number width for checkpoint filenames (zero-padded)
    epoch_width = len(str(epochs))

    # Create checkpoint directory (optional)
    checkpoint_dir = None
    if 'checkpoint_dir' in config['output']:
        checkpoint_dir = Path(out_dir) / config['output']['checkpoint_dir']
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_events = len(train_events)
    train_data = [_event_to_jax(e) for e in train_events]
    event_names = [e.event_name for e in train_events]
    event_days = [ed['wind_u'].shape[0] for ed in train_data]

    # Count total days for logging
    total_days = sum(event_days)

    # Loss scale per event: 1/n_days so each event contributes equally
    event_loss_scales = [1.0 / nd for nd in event_days]

    @eqx.filter_jit
    def make_step(model, opt_state, p_u, p_b, p_bd,
                  static_cnn, slope_ca, aspect_up,
                  fuel_type_map, wind_u_day, wind_v_day,
                  fuel_mask, target_day, loss_scale, dropout_key):
        """Forward 1 day → loss → grads → clip → update."""
        def loss_fn(m):
            p_fire, final_state = run_one_day(
                m, p_u, p_b, p_bd,
                static_cnn, slope_ca, aspect_up,
                fuel_type_map, wind_u_day, wind_v_day,
                fuel_mask, config,
                key=dropout_key,
            )
            loss = compute_day_loss(p_fire, target_day, config)
            return loss * loss_scale, (final_state, loss)

        (scaled_loss, ((final_pu, final_pb, final_pbd), raw_loss)), grads = (
            eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        )

        # Per-element gradient clipping matching prob_jax_real
        grads = clip_gradients(grads, grad_clip_lo, grad_clip_hi)

        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)

        return (new_model, new_opt_state, raw_loss,
                jax.lax.stop_gradient(final_pu),
                jax.lax.stop_gradient(final_pb),
                jax.lax.stop_gradient(final_pbd))

    logger.info(f"Single-day mode: {total_days} days, {total_days} updates/epoch")
    logger.info(f"Event days: {dict(zip(event_names, event_days))}")
    logger.info(f"Event loss scales: {dict(zip(event_names, [f'{s:.3f}' for s in event_loss_scales]))}")
    logger.info(f"Constant LR: {config['training']['lr']}")
    logger.info(f"Per-element grad clip: [{grad_clip_lo}, {grad_clip_hi}]")
    logger.info(f"Early stopping patience: {patience} validation checks")
    logger.info(f"Checkpoint saving starts at epoch {start_save_epoch}")
    logger.info(f"Train events: {event_names}")
    logger.info(f"Val events: {[e.event_name for e in val_events]}")

    # Optional per-epoch loss+iou CSV
    loss_csv_path = None
    loss_csv_file = None
    save_all_loss = config['training'].get('save_all_loss')
    if save_all_loss:
        loss_csv_path = Path(out_dir) / save_all_loss
        loss_csv_file = open(loss_csv_path, 'w')
        loss_csv_file.write('epoch,loss,iou\n')
        logger.info(f"Saving per-epoch loss+iou to {loss_csv_path}")

    rng = np.random.default_rng(config['model']['seed'])
    dropout_key = jax.random.PRNGKey(config['model']['seed'] + 1)
    best_avg_iou = -1.0       # Best average IoU across events
    best_loss_iou = 0.0       # IoU when best loss was achieved
    best_iou_loss = float("inf")  # Loss when best IoU was achieved
    val_ious = {}             # Per-event IoU at last validation
    curr_val = float('nan')   # Last computed validation loss

    for epoch in range(epochs):
        t0 = time.perf_counter()
        nan_detected = False

        # Shuffle event order each epoch
        event_order = rng.permutation(n_events)

        # Split dropout key for this epoch
        dropout_key, epoch_key = jax.random.split(dropout_key)

        # Track per-event losses
        event_losses = {name: [] for name in event_names}

        # Pre-split keys for all days in this epoch
        day_keys = jax.random.split(epoch_key, total_days)
        day_key_idx = 0

        for idx in event_order:
            ed = train_data[idx]
            fire_seq = ed['fire_seq']
            n_days = event_days[idx]
            loss_scale = event_loss_scales[idx]

            p_u, p_b, p_bd = make_initial_state(fire_seq[0])
            p_u = jax.lax.stop_gradient(p_u)
            p_b = jax.lax.stop_gradient(p_b)
            p_bd = jax.lax.stop_gradient(p_bd)

            for day in range(n_days):
                target_day = fire_seq[day + 1]
                wind_u_day = ed['wind_u'][day]
                wind_v_day = ed['wind_v'][day]

                (model, opt_state, loss,
                 p_u, p_b, p_bd) = make_step(
                    model, opt_state, p_u, p_b, p_bd,
                    ed['static_cnn_input'], ed['slope_ca'], ed['aspect_upslope'],
                    ed['fuel_type_map'],
                    wind_u_day, wind_v_day,
                    ed['fuel_mask'], target_day, loss_scale,
                    day_keys[day_key_idx],
                )
                day_key_idx += 1

                loss_val = float(loss)
                if not np.isfinite(loss_val):
                    nan_detected = True
                    logger.warning(
                        f"NaN/Inf at epoch {epoch}, event {event_names[idx]}, "
                        f"day {day}. Stopping epoch.")
                    break

                event_losses[event_names[idx]].append(loss_val)

            if nan_detected:
                break

        if nan_detected:
            continue

        # Per-event average loss
        event_avg = {name: np.mean(losses) for name, losses in event_losses.items()}
        avg_loss = np.mean(list(event_avg.values()))  # Equal weight per event
        dt = time.perf_counter() - t0

        # Compute IoU (on val events if available, else train events)
        eval_events = val_events if val_events else train_events
        curr_val, val_ious = validate(model, eval_events, config)
        avg_iou = np.mean(list(val_ious.values())) if val_ious else 0.0

        # Write to loss+iou CSV
        if loss_csv_file:
            loss_csv_file.write(f'{epoch},{avg_loss:.6f},{avg_iou:.6f}\n')
            loss_csv_file.flush()

        # Save best loss model
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            best_loss_iou = avg_iou
            best_loss_path = Path(out_dir) / config['output']['best_loss_file']
            eqx.tree_serialise_leaves(str(best_loss_path), model)
            # Save checkpoint after start_save_epoch
            if checkpoint_dir and epoch >= start_save_epoch:
                loss_str = f"{best_train_loss:.3f}".replace('.', '_')
                ckpt_name = f"{epoch:0{epoch_width}d}__{loss_str}.eqx"
                eqx.tree_serialise_leaves(str(checkpoint_dir / ckpt_name), model)

        # Save best IoU model
        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_iou_loss = avg_loss
            best_iou_path = Path(out_dir) / config['output']['best_iou_file']
            eqx.tree_serialise_leaves(str(best_iou_path), model)

        # Early stopping based on validation loss
        if val_events:
            if np.isfinite(curr_val):
                if curr_val < best_val_loss:
                    best_val_loss = curr_val
                    patience_counter = 0
                else:
                    patience_counter += 1

        if epoch % print_interval == 0 or epoch == epochs - 1:
            parts = [f"{n} iou={v:.3f}" for n, v in val_ious.items()]
            per_event_str = ", ".join(parts)
            logger.info(
                f"Epoch {epoch}: avg={avg_loss:.4f}, best={best_train_loss:.4f}, "
                f"best_iou={best_avg_iou:.4f}, {per_event_str} ({dt:.2f}s)")

        if val_events and patience_counter >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(val not improved for {patience} validation checks)")
            break

    if loss_csv_file:
        loss_csv_file.close()

    logger.info("-" * 70)
    logger.info(f"best_loss.eqx  — loss: {best_train_loss:.6f}, iou: {best_loss_iou:.4f}")
    logger.info(f"best_iou.eqx   — loss: {best_iou_loss:.6f}, iou: {best_avg_iou:.4f}")

    return model
