"""Filesystem path constants for the HGAM project.

Computed from the location of this file, so the package can be moved or
installed elsewhere without any string edits in code.
"""
from pathlib import Path

# This file lives at hgam/utils/paths.py — parent.parent.parent gives project root
HGAM_PKG_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = HGAM_PKG_ROOT.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
LEGACY_CONFIG_DIR = CONFIG_DIR / "_legacy"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
ARCHIVE_DIR = PROJECT_ROOT / "_archive"

SIGNAL_POINT_DATA = DATA_DIR / "signal_points.npy"


def ensure_runtime_dirs() -> None:
    """Create runtime output directories if they do not exist. Idempotent."""
    for d in (DATA_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
