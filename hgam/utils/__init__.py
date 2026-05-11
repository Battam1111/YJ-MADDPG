"""HGAM utility surface: paths, config loading, checkpoints, action ops."""

from .paths import (
    HGAM_PKG_ROOT,
    PROJECT_ROOT,
    CONFIG_DIR,
    LEGACY_CONFIG_DIR,
    DATA_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    CHECKPOINTS_DIR,
    ARCHIVE_DIR,
    SIGNAL_POINT_DATA,
    ensure_runtime_dirs,
)
from .config import load_default_config, load_yaml_dir
from ._legacy_utils import action_normalize, latest_logdir, get_load_path

__all__ = [
    "HGAM_PKG_ROOT",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "LEGACY_CONFIG_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "LOGS_DIR",
    "CHECKPOINTS_DIR",
    "ARCHIVE_DIR",
    "SIGNAL_POINT_DATA",
    "ensure_runtime_dirs",
    "load_default_config",
    "load_yaml_dir",
    "action_normalize",
    "latest_logdir",
    "get_load_path",
]
