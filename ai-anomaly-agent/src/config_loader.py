"""Small helper to load config/config.yaml once and reuse it everywhere."""
from __future__ import annotations

import functools
import os
import yaml

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml"
)


@functools.lru_cache(maxsize=1)
def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
