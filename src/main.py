#!/usr/bin/env python3
"""Entry point for CNN-parameterized CA wildfire model training."""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax

from data_loader import load_all_events
from model import WildfireModel
from training import train
from utils import configure_logging, copy_config, load_config

logger = logging.getLogger(__name__)


def main(config_path):
    config = load_config(config_path)

    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    out_dir = Path(f"{config['output']['out_dir']}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to output dir at start
    copy_config(Path(config_path), out_dir / config["output"]["config_file"])

    # Setup logging
    log_file = str(out_dir / config["output"]["log_file"])
    configure_logging(log_file)

    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"JAX devices: {jax.devices()}")

    # Load data
    logger.info("Loading data...")
    train_events, test_events = load_all_events(config)

    # Create model
    logger.info("Creating model...")
    key = jax.random.PRNGKey(config["model"]["seed"])
    model = WildfireModel(config, key)

    # Load pretrained weights if specified
    pretrained_path = config.get("pretrained_model_path")
    if pretrained_path:
        pretrained_path = Path(pretrained_path)
        if pretrained_path.exists():
            model = eqx.tree_deserialise_leaves(str(pretrained_path), model)
            logger.info(f"  Loaded pretrained model from {pretrained_path}")
        else:
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")

    param_count = sum(
        x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)) if hasattr(x, "size")
    )
    logger.info(f"  Model parameters: {param_count:,}")
    logger.info(f"  Kernel size: {config['model']['kernel_size']}")
    logger.info(f"  Branch out channels: {config['model']['branch_out_channels']}")
    logger.info(f"  Burn duration: {config['model']['burn_duration']}")

    # Train
    logger.info("=" * 70)
    logger.info("TRAINING")
    logger.info("=" * 70)

    t_start = time.perf_counter()
    model = train(model, train_events, test_events, config, out_dir)
    total_time = time.perf_counter() - t_start

    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    logger.info(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "src/config.yaml"

    main(config_path)
