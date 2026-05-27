"""Multi-event data loading for CNN-parameterized CA wildfire model."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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

logger = logging.getLogger(__name__)


@dataclass
class FireEvent:
    """Preprocessed fire event data."""

    event_name: str
    static_cnn_input: np.ndarray   # (7, H, W) CNN static channels
    slope_ca: np.ndarray           # (H, W) normalized slope [0, 1] for CA physics
    aspect_upslope: np.ndarray     # (H, W) upslope direction in radians for CA physics
    fuel_type_map: np.ndarray      # (H, W) FBFM40 integer indices for fuel embedding
    wind_u: np.ndarray             # (D, H, W) scaled wind u
    wind_v: np.ndarray             # (D, H, W) scaled wind v
    fire_seq: np.ndarray           # (D+1, H, W) fire progression
    fuel_mask: np.ndarray          # (H, W) binary mask from FBFM40
    n_days: int
    grid_shape: tuple


def load_fire_event(event_name, config, event_info):
    """Load and preprocess a single fire event.

    Args:
        event_name: Name of the fire event directory.
        config: Full config dict.
        event_info: Dict with {days, y1, y2, x1, x2} for cropping.
    """
    base_dir = config['data']['base_dir']
    static_files = config['data']['static_files']
    dynamic_files = config['data']['dynamic_files']
    target_file = config['data']['target_file']

    event_dir = f"{base_dir}/{event_name}"

    # Load raw static data
    elev = np.load(f"{event_dir}/{static_files['elevation']}")
    slope = np.load(f"{event_dir}/{static_files['slope']}")
    aspect = np.load(f"{event_dir}/{static_files['aspect']}")
    cbd = np.load(f"{event_dir}/{static_files['canopy_bulk_density']}")
    cc = np.load(f"{event_dir}/{static_files['canopy_cover']}")
    ch = np.load(f"{event_dir}/{static_files['canopy_height']}")
    fbfm = np.load(f"{event_dir}/{static_files['fuel_model']}")

    # Load dynamic data
    wind_u = np.load(f"{event_dir}/{dynamic_files['wind_u']}")
    wind_v = np.load(f"{event_dir}/{dynamic_files['wind_v']}")

    # Load target and normalize to [0, 1]
    fire = np.load(f"{event_dir}/{target_file}").astype(np.float32)
    if fire.max() > 1.0:
        fire = fire / 255.0
    fire = np.clip(fire, 0.0, 1.0)

    # Apply spatial crop
    y1, y2 = event_info['y1'], event_info['y2']
    x1, x2 = event_info['x1'], event_info['x2']

    elev = elev[y1:y2, x1:x2]
    slope = slope[y1:y2, x1:x2]
    aspect = aspect[y1:y2, x1:x2]
    cbd = cbd[y1:y2, x1:x2]
    cc = cc[y1:y2, x1:x2]
    ch = ch[y1:y2, x1:x2]
    fbfm = fbfm[y1:y2, x1:x2]
    wind_u = wind_u[:, y1:y2, x1:x2]
    wind_v = wind_v[:, y1:y2, x1:x2]
    fire = fire[:, y1:y2, x1:x2]

    # --- CNN input channels (7 static) ---
    elev_norm = preprocess_elevation(elev, config)
    sin_slope, sin_aspect, cos_aspect = preprocess_slope_aspect_cnn(slope, aspect, config)
    cbd_cnn, cc_cnn, ch_cnn = preprocess_canopy_cnn(cbd, cc, ch, config)

    static_cnn_input = np.stack([
        elev_norm, sin_slope, sin_aspect, cos_aspect,
        cbd_cnn, cc_cnn, ch_cnn,
    ], axis=0).astype(np.float32)

    # --- CA physics inputs ---
    slope_ca = preprocess_slope_ca(slope, config)
    aspect_upslope = convert_aspect_to_upslope_radians(aspect, config)

    # --- FBFM40 fuel type map (for learnable embedding in model) ---
    fuel_type_map = preprocess_fbfm40(fbfm)

    # Preprocess wind (fixed divisor matching prob_jax_real)
    wind_speed_max = config['preprocessing']['wind_speed_max']
    wind_u_scaled, wind_v_scaled = preprocess_wind(
        wind_u, wind_v, wind_speed_max
    )

    # Compute fuel mask from FBFM40 (non-burnable = codes 91-99)
    fuel_mask = compute_fuel_mask(fbfm, config)

    # Parse day range: [start, end] or int (legacy: int N → [0, N])
    days_cfg = event_info['days']
    if isinstance(days_cfg, (list, tuple)):
        day_start, day_end = int(days_cfg[0]), int(days_cfg[1])
    else:
        day_start, day_end = 0, int(days_cfg)

    # Cap day_end by available data
    n_fire_frames = fire.shape[0]
    n_wind_days = wind_u_scaled.shape[0]
    max_end = min(n_fire_frames - 1, n_wind_days)
    day_end = min(day_end, max_end)

    # Slice wind and fire for [day_start, day_end).
    # Wind for day d drives fire spread during day d.
    # Initial condition: for day_start > 0, use ground truth from day (day_start - 1);
    # for day_start == 0, use fire frame 0 as initial state.
    n_usable_days = day_end - day_start
    fire_init_idx = day_start - 1 if day_start > 0 else 0
    fire_end_idx = fire_init_idx + n_usable_days + 1  # initial + n_usable_days targets
    wind_u_scaled = wind_u_scaled[day_start:day_end]
    wind_v_scaled = wind_v_scaled[day_start:day_end]
    fire = fire[fire_init_idx:fire_end_idx]  # fire[0] = initial state

    grid_shape = (fire.shape[1], fire.shape[2])

    burnable_pct = fuel_mask.sum() / fuel_mask.size * 100
    logger.info(
        f"  {event_name}: grid={grid_shape}, days={n_usable_days}, "
        f"static_cnn={static_cnn_input.shape}, fire={fire.shape}, "
        f"fuel_mask={burnable_pct:.1f}% burnable"
    )

    return FireEvent(
        event_name=event_name,
        static_cnn_input=static_cnn_input,
        slope_ca=slope_ca.astype(np.float32),
        aspect_upslope=aspect_upslope.astype(np.float32),
        fuel_type_map=fuel_type_map,
        wind_u=wind_u_scaled,
        wind_v=wind_v_scaled,
        fire_seq=fire,
        fuel_mask=fuel_mask,
        n_days=n_usable_days,
        grid_shape=grid_shape,
    )


def load_all_events(config):
    """Load all train and test fire events.

    Returns: (train_events, test_events)
    """
    train_events_cfg = config['data']['train_events']
    test_events_cfg = config['data'].get('test_events') or {}

    logger.info("Loading training events...")
    train_events = [
        load_fire_event(name, config, event_info=info)
        for name, info in train_events_cfg.items()
    ]

    test_events = []
    if test_events_cfg:
        logger.info("Loading test events...")
        test_events = [
            load_fire_event(name, config, event_info=info)
            for name, info in test_events_cfg.items()
        ]
    else:
        logger.info("No test events configured, skipping validation data.")

    return train_events, test_events
