"""Shared test fixtures for prob_nn tests."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Add prob_nn to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tiny_config():
    """Minimal config dict for unit testing with small dimensions."""
    return {
        "data": {
            "base_dir": "data/data_real",
            "train_events": {
                "Bear_2020": {"days": [0, 3], "y1": 166, "y2": 178, "x1": 166, "x2": 178},
            },
            "test_events": {
                "Buck_2017": {"days": [0, 3], "y1": 166, "y2": 178, "x1": 166, "x2": 178},
            },
            "static_files": {
                "elevation": "ELEV2020.npy",
                "slope": "SLPD2020.npy",
                "aspect": "ASP2020.npy",
                "canopy_bulk_density": "230CBD.npy",
                "canopy_cover": "230CC.npy",
                "canopy_height": "230CH.npy",
                "fuel_model": "230FBFM40.npy",
            },
            "dynamic_files": {
                "wind_u": "u_component_of_wind_10m.npy",
                "wind_v": "v_component_of_wind_10m.npy",
            },
            "target_file": "fire.npy",
        },
        "preprocessing": {
            "nodata_value": -9999,
            "canopy_max": 100,
            "density_max": 45,
            "canopy_height_max": 550,
            "wind_speed_max": 10.0,
            "fbfm_num_classes": 203,
            "fbfm_embedding_dim": 4,
            "fbfm_non_burnable": [91, 99],
        },
        "model": {
            "kernel_size": 3,
            "branch_out_channels": 4,
            "n_ca_params": 6,
            "cnn_in_channels": 13,
            "burn_duration": 3,
            "dropout_rate": 0.0,
            "seed": 42,
        },
        "training": {
            "epochs": 2,
            "lr": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "grad_clip_norm": 1.0,
            "steps_per_day": 3,
            "bce_weight": 0.5,
            "mse_weight": 1.0,
            "area_weight": 0.6,
            "frontier_weight": 5.0,
            "mse_pool_window": 0,
            "Pth": 0.5,
            "bce_epsilon": 1e-7,
            "use_remat": False,
            "print_interval": 1,
            "patience": 50,
            "start_save_epoch": 0,
            "grad_clip": {
                "per_element": [-1.0, 1.0],
            },
        },
        "output": {
            "out_dir": "/tmp/prob_nn_test",
            "model_train_file": "model_train.eqx",
            "checkpoint_dir": "checkpoint",
            "config_file": "config.yaml",
            "log_file": "training.log",
        },
    }


@pytest.fixture
def small_grid():
    """Small 8x10 grid arrays for testing."""
    H, W = 8, 10
    key = jax.random.PRNGKey(0)
    return {
        "H": H,
        "W": W,
        "p_unburned": jnp.ones((H, W)),
        "p_burning": jnp.zeros((H, W)).at[3:5, 4:6].set(1.0),
        "p_burned": jnp.zeros((H, W)),
        "fire_state_0": jnp.zeros((H, W)).at[3:5, 4:6].set(1.0),
        "static_cnn_input": jax.random.normal(key, (7, H, W)),
        "slope_ca": jax.random.uniform(key, (H, W)),
        "aspect_upslope": jax.random.uniform(key, (H, W)) * 2 * jnp.pi,
        "fuel_type_map": jnp.zeros((H, W), dtype=jnp.int32),  # All fuel type 0
        "wind_u": jax.random.normal(key, (3, H, W)) * 0.3,
        "wind_v": jax.random.normal(key, (3, H, W)) * 0.3,
        "fuel_mask": jnp.ones((H, W)),
        "key": key,
    }
