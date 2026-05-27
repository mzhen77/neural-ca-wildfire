"""Utility functions for CNN-parameterized probabilistic CA wildfire model."""

import logging
import os
import shutil
from pathlib import Path

import yaml


def load_config(config_path):
    """Load YAML configuration file into a dictionary."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def configure_logging(log_file):
    """Configure logging to console and file."""
    handlers = [logging.StreamHandler()]

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def copy_config(src_path, dst_path):
    """Copy configuration file to output directory."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if src_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
