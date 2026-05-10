from __future__ import annotations

from pathlib import Path

import yaml

from two_stage_screen.models import ScreenConfig


def load_screen_config(path: Path) -> ScreenConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must parse to a mapping at the root.")
    return ScreenConfig.model_validate(raw)
