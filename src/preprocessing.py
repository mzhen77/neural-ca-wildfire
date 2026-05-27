"""Feature engineering and preprocessing for CNN-parameterized CA wildfire model.

Produces two sets of outputs:
1. CNN input channels: [elev, sin_slope, sin_asp, cos_asp, cbd_norm, cc_norm, ch_norm, fbfm_embed(4), wind(2)]
   FBFM40 embedding is computed at runtime by the model (learnable).
   Static CNN channels = 7, total with embedding + wind = 13.
2. CA physics inputs: slope [0,1], aspect_upslope (radians)
   Fuel contribution is now learned by the CNN (fuel_factor output), not hard-coded from canopy.
"""

import numpy as np


def preprocess_elevation(elev, config):
    """Per-event min-max normalize elevation to [0, 1].

    Masks nodata values, then min-max normalizes.
    """
    nodata = config['preprocessing']['nodata_value']
    elev = elev.astype(np.float32)
    mask = elev <= nodata + 1
    elev[mask] = 0.0
    e_min = elev[~mask].min() if (~mask).any() else 0.0
    e_max = elev[~mask].max() if (~mask).any() else 1.0
    denom = e_max - e_min
    if denom < 1e-8:
        denom = 1.0
    elev_norm = (elev - e_min) / denom
    elev_norm[mask] = 0.0
    return np.clip(elev_norm, 0.0, 1.0)


def preprocess_slope_aspect_cnn(slope_raw, aspect_raw, config):
    """Preprocess slope/aspect for CNN input channels.

    Raw data is already in degrees (slope: 0-90, aspect: 0-359 compass).

    Returns: (sin_slope, sin_aspect, cos_aspect) each (H, W).
    Flat cells (LANDFIRE aspect=-1) get sin_aspect=0, cos_aspect=0.
    """
    slope_deg = slope_raw.astype(np.float32)
    aspect_deg = aspect_raw.astype(np.float32)

    slope_rad = np.deg2rad(np.clip(slope_deg, 0.0, 90.0))
    sin_slope = np.sin(slope_rad)

    flat_mask = aspect_raw < 0
    aspect_rad = np.deg2rad(aspect_deg)
    sin_aspect = np.sin(aspect_rad)
    cos_aspect = np.cos(aspect_rad)
    sin_aspect[flat_mask] = 0.0
    cos_aspect[flat_mask] = 0.0

    return sin_slope, sin_aspect, cos_aspect


def preprocess_slope_ca(slope_raw, config):
    """Preprocess slope for CA physics: normalized to [0, 1].

    slope_raw is in degrees (0-90). Matches prob_jax_real: slope/90.
    """
    slope_deg = slope_raw.astype(np.float32)
    slope_norm = np.clip(slope_deg / 90.0, 0.0, 1.0)
    return slope_norm


def convert_aspect_to_upslope_radians(aspect_raw, config):
    """Convert LANDFIRE compass-degree aspect to upslope direction in math radians.

    LANDFIRE: 0=North, 90=East, clockwise, -1=flat.
    Math: 0=East, counter-clockwise.
    Upslope = opposite of downslope direction.

    Matches prob_jax_real/data_loader.py convert_aspect_to_upslope_radians().

    Returns: (H, W) upslope direction in radians [0, 2π]. Flat cells → 0.
    """
    aspect_deg = aspect_raw.astype(np.float32)
    flat_mask = aspect_raw < 0

    # LANDFIRE downslope compass → upslope compass → math radians
    # Matches prob_jax_real exactly:
    upslope_compass = (aspect_deg + 180.0) % 360.0
    upslope_math_deg = (90.0 - upslope_compass) % 360.0
    upslope_rad = np.deg2rad(upslope_math_deg)

    upslope_rad[flat_mask] = 0.0
    return upslope_rad.astype(np.float32)


def preprocess_canopy_cnn(cbd, cc, ch, config):
    """Preprocess canopy variables for CNN input: divide by fixed max.

    CBD / density_max, CC / canopy_max, CH / canopy_height_max.
    Returns: (cbd_norm, cc_norm, ch_norm) each (H, W).
    """
    nodata = config['preprocessing']['nodata_value']
    canopy_max = config['preprocessing']['canopy_max']
    density_max = config['preprocessing']['density_max']
    canopy_height_max = config['preprocessing']['canopy_height_max']

    cbd_f = cbd.astype(np.float32)
    cc_f = cc.astype(np.float32)
    ch_f = ch.astype(np.float32)

    cbd_f[cbd_f <= nodata + 1] = 0.0
    cc_f[cc_f <= nodata + 1] = 0.0
    ch_f[ch_f <= nodata + 1] = 0.0

    cbd_norm = cbd_f / density_max
    cc_norm = cc_f / canopy_max
    ch_norm = ch_f / canopy_height_max

    return cbd_norm, cc_norm, ch_norm


def preprocess_fbfm40(fbfm):
    """Convert raw FBFM40 integers to integer indices for embedding.

    Values are used directly as indices (max 202 fits in fbfm_num_classes=203).
    Returns: (H, W) int32 array.
    """
    fbfm_int = fbfm.astype(np.int32)
    fbfm_int = np.clip(fbfm_int, 0, 202)
    return fbfm_int


def preprocess_wind(wind_u, wind_v, wind_speed_max):
    """Scale wind by wind_speed_max (fixed divisor, matching prob_jax_real).

    NaN values replaced with 0.0 (calm wind).
    Returns: (wind_u_scaled, wind_v_scaled) each (D, H, W).
    """
    wind_u = wind_u.astype(np.float32)
    wind_v = wind_v.astype(np.float32)
    wind_u = np.nan_to_num(wind_u, nan=0.0)
    wind_v = np.nan_to_num(wind_v, nan=0.0)
    wind_u = wind_u / wind_speed_max
    wind_v = wind_v / wind_speed_max
    return wind_u, wind_v


def compute_fuel_mask(fbfm, config):
    """Binary fuel mask from FBFM40: 1=burnable, 0=non-burnable.

    Non-burnable: FBFM40 codes in [lo, hi] range from config (default 91-99).
    These codes represent urban, water, snow/ice, agriculture, bare ground.
    """
    lo, hi = config['preprocessing']['fbfm_non_burnable']
    non_burnable = (fbfm >= lo) & (fbfm <= hi)
    return (~non_burnable).astype(np.float32)
