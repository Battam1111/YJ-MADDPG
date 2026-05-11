"""YAML configuration loader.

Merges all YAML files under configs/env/ and configs/method/ into a single
flat dict. This replaces the legacy pattern of iterating os.listdir on a
hardcoded HGAT-MADDPG_ver2/config directory.

Files starting with `_` (underscore) are skipped — reserved for legacy
compatibility scaffolding.
"""
from pathlib import Path
from typing import Any, Dict, Union

from yaml import safe_load

from .paths import CONFIG_DIR


PathLike = Union[str, Path]


def load_yaml_dir(directory: PathLike) -> Dict[str, Any]:
    """Load every .yaml file in `directory` and merge their top-level keys.

    Files whose names start with `_` are skipped (legacy scaffolding).
    Returns an empty dict if the directory does not exist.
    """
    directory = Path(directory)
    merged: Dict[str, Any] = {}
    if not directory.exists():
        return merged
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        if f.suffix not in (".yaml", ".yml"):
            continue
        if f.name.startswith("_"):
            continue
        with f.open("r", encoding="utf-8") as fp:
            doc = safe_load(fp) or {}
        merged.update(doc)
    return merged


def load_default_config(
    env_subdir: str = "env",
    method_subdir: str = "method",
) -> Dict[str, Any]:
    """Merge env-level and method-level configs into a single param dict.

    The same flat dict that the legacy code expected (one big bag of
    SHOUTING_CASE attributes) is preserved as the return value.
    """
    config: Dict[str, Any] = {}
    config.update(load_yaml_dir(CONFIG_DIR / env_subdir))
    config.update(load_yaml_dir(CONFIG_DIR / method_subdir))
    return config
