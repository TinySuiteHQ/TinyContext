"""Native storage locations for installed TinyContext packages."""

from __future__ import annotations

from pathlib import Path

import platformdirs


APP_NAME = "tinycontext"


def native_config_path() -> Path:
    return (
        Path(platformdirs.user_config_dir(APP_NAME, appauthor=False))
        / "context_config.json"
    )


def native_data_dir() -> Path:
    return Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))


def native_memory_db_path() -> Path:
    return native_data_dir() / "memories.db"
