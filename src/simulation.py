"""Simulation loops for CNN-parameterized CA wildfire model.

Follows prob_jax_real_nn/simulation.py pattern:
- Nested scan: outer=days, inner=micro-steps
- CNN generates params once per day (wind changes daily)
- 3-state CA physics from prob_jax_real for micro-steps
"""

import jax
import jax.numpy as jnp

from model import WildfireCA, WildfireModel, compute_wind_speed_direction


def run_simulation(model, p_unburned_0, p_burning_0, p_burned_0,
                   static_cnn_input, slope_ca, aspect_upslope,
                   fuel_type_map, wind_u, wind_v,
                   fuel_mask, config, *, key=None):
    """Run full multi-day simulation with CNN-parameterized CA.

    Args:
        model: WildfireModel instance
        p_unburned_0, p_burning_0, p_burned_0: (H, W) initial states
        static_cnn_input: (7, H, W) static CNN channels
        slope_ca: (H, W) normalized slope for CA physics
        aspect_upslope: (H, W) upslope direction in radians
        fuel_type_map: (H, W) int32 FBFM40 fuel type indices
        wind_u: (D, H, W) daily wind u (scaled)
        wind_v: (D, H, W) daily wind v (scaled)
        fuel_mask: (H, W) binary fuel mask
        config: config dict
        key: PRNG key for dropout (None = inference mode via eqx.tree_inference)

    Returns:
        (daily_pfire, final_state)
    """
    steps_per_day = config['training']['steps_per_day']
    use_remat = config['training']['use_remat']
    burn_duration = model.burn_duration
    n_days = wind_u.shape[0]

    # Pre-split keys for all days if training with dropout
    if key is not None:
        day_keys = jax.random.split(key, n_days)
    else:
        day_keys = jnp.zeros((n_days, 2), dtype=jnp.uint32)  # dummy

    def daily_step(carry, day_inputs):
        p_u, p_b, p_bd = carry
        day_wind_u, day_wind_v, day_key = day_inputs

        # Generate CNN params for this day (with dropout during training)
        cnn_key = day_key if key is not None else None
        ca_params = model.generate_params(
            static_cnn_input, day_wind_u, day_wind_v, fuel_type_map,
            key=cnn_key,
        )

        # Compute wind speed and direction for CA physics
        wind_speed, wind_dir = compute_wind_speed_direction(day_wind_u, day_wind_v)

        def micro_step(state, _step_idx):
            pu, pb, pbd = state
            new_pu, new_pb, new_pbd = WildfireCA.step_deterministic(
                pu, pb, pbd,
                ca_params,
                wind_speed, wind_dir, slope_ca, aspect_upslope,
                fuel_mask, burn_duration,
            )
            return (new_pu, new_pb, new_pbd), None

        scan_fn = jax.checkpoint(micro_step) if use_remat else micro_step
        (final_pu, final_pb, final_pbd), _ = jax.lax.scan(
            scan_fn, (p_u, p_b, p_bd), jnp.arange(steps_per_day)
        )

        p_fire = 1.0 - final_pu
        return (final_pu, final_pb, final_pbd), p_fire

    final_state, daily_pfire = jax.lax.scan(
        daily_step,
        (p_unburned_0, p_burning_0, p_burned_0),
        (wind_u, wind_v, day_keys),
    )

    return daily_pfire, final_state  # (D, H, W), (p_u, p_b, p_bd)


def run_one_day(model, p_u, p_b, p_bd,
                static_cnn_input, slope_ca, aspect_upslope,
                fuel_type_map, wind_u_day, wind_v_day,
                fuel_mask, config, *, key=None):
    """Run one day of micro-steps (CNN params + CA physics).

    Used by single_day training mode: forward 1 day → loss → backward → update.

    Args:
        model: WildfireModel
        p_u, p_b, p_bd: (H, W) initial 3-state for this day
        static_cnn_input: (7, H, W) static CNN channels
        slope_ca, aspect_upslope: (H, W) CA physics inputs
        fuel_type_map: (H, W) int32 FBFM40 fuel type indices
        wind_u_day, wind_v_day: (H, W) wind for this day
        fuel_mask: (H, W)
        config: config dict
        key: PRNG key for dropout (None = inference)

    Returns:
        (p_fire, final_state)
    """
    steps_per_day = config['training']['steps_per_day']
    use_remat = config['training']['use_remat']
    burn_duration = model.burn_duration

    ca_params = model.generate_params(
        static_cnn_input, wind_u_day, wind_v_day, fuel_type_map,
        key=key,
    )
    wind_speed, wind_dir = compute_wind_speed_direction(wind_u_day, wind_v_day)

    def micro_step(state, _):
        pu, pb, pbd = state
        new_pu, new_pb, new_pbd = WildfireCA.step_deterministic(
            pu, pb, pbd,
            ca_params,
            wind_speed, wind_dir, slope_ca, aspect_upslope,
            fuel_mask, burn_duration,
        )
        return (new_pu, new_pb, new_pbd), None

    scan_fn = jax.checkpoint(micro_step) if use_remat else micro_step
    (final_pu, final_pb, final_pbd), _ = jax.lax.scan(
        scan_fn, (p_u, p_b, p_bd), jnp.arange(steps_per_day)
    )

    p_fire = 1.0 - final_pu
    return p_fire, (final_pu, final_pb, final_pbd)


def run_simulation_no_grad(model, p_unburned_0, p_burning_0, p_burned_0,
                           static_cnn_input, slope_ca, aspect_upslope,
                           fuel_type_map, wind_u, wind_v,
                           fuel_mask, config):
    """Run simulation without gradient tracking for validation/inference.

    Model should already have inference=True set via eqx.tree_inference.
    No key needed — dropout is disabled.
    Returns only daily_pfire (D, H, W).
    """
    daily_pfire, _ = jax.lax.stop_gradient(
        run_simulation(
            model, p_unburned_0, p_burning_0, p_burned_0,
            static_cnn_input, slope_ca, aspect_upslope,
            fuel_type_map, wind_u, wind_v,
            fuel_mask, config,
        )
    )
    return daily_pfire
