"""Public, transport-independent TinyContext configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinycontext.paths import native_data_dir, native_memory_db_path


def default_config() -> dict[str, Any]:
    return {
        "memory_db_path": str(native_memory_db_path()),
        "recall_top_k": 10,
        "recall_max_tokens": 2000,
        "encoding_name": "o200k_base",
    }


def _resolve_db_path(value: Any, *, base_dir: Path | None) -> str:
    raw = str(value or "").strip()
    path = Path(raw).expanduser() if raw else native_memory_db_path()
    if not path.is_absolute():
        path = (base_dir or native_data_dir()) / path
    return os.path.normpath(str(path))


def normalize_config(
    raw: Mapping[str, Any] | None = None,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    values = {
        key: value
        for key, value in dict(raw or {}).items()
        if not str(key).startswith("_comment")
    }
    unknown = set(values) - set(default_config())
    if unknown:
        raise ValueError(
            "unknown TinyContext config field(s): " + ", ".join(sorted(unknown))
        )

    config = default_config()
    config.update(values)
    config["memory_db_path"] = _resolve_db_path(
        config["memory_db_path"],
        base_dir=Path(base_dir).expanduser().resolve() if base_dir else None,
    )

    for key in ("recall_top_k", "recall_max_tokens"):
        try:
            value = int(config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if value < 1:
            raise ValueError(f"{key} must be at least 1")
        config[key] = value

    encoding_name = str(config["encoding_name"] or "").strip()
    if not encoding_name:
        raise ValueError("encoding_name must not be empty")
    config["encoding_name"] = encoding_name
    return config


@dataclass(frozen=True)
class TinyContextConfig(Mapping[str, Any]):
    """Validated immutable wrapper around TinyContext's flat configuration."""

    _values: dict[str, Any]

    def __init__(self, **values: Any) -> None:
        object.__setattr__(self, "_values", normalize_config(values))

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None = None,
        *,
        base_dir: str | Path | None = None,
    ) -> TinyContextConfig:
        instance = cls.__new__(cls)
        object.__setattr__(
            instance,
            "_values",
            normalize_config(values, base_dir=base_dir),
        )
        return instance

    @classmethod
    def from_json(cls, path: str | Path) -> TinyContextConfig:
        config_path = Path(path).expanduser().resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"context config must be a JSON object: {config_path}")
        return cls.from_mapping(raw, base_dir=config_path.parent)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def with_overrides(
        self,
        values: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
    ) -> TinyContextConfig:
        merged = self.to_dict()
        merged.update(values)
        return TinyContextConfig.from_mapping(merged, base_dir=base_dir)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, key: str) -> Any:
        try:
            return object.__getattribute__(self, "_values")[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


ConfigInput = TinyContextConfig | Mapping[str, Any]


def resolve_config(
    config: ConfigInput | None = None,
    *,
    path: str | Path | None = None,
) -> TinyContextConfig:
    """Resolve an explicit file and per-call overrides without reading the environment."""
    base = TinyContextConfig.from_json(path) if path is not None else TinyContextConfig()
    if config is None:
        return base
    if isinstance(config, TinyContextConfig):
        return config if path is None else base.with_overrides(config.to_dict())
    return base.with_overrides(config)


def save_config(config: ConfigInput, path: str | Path) -> TinyContextConfig:
    config_path = Path(path).expanduser().resolve()
    resolved = TinyContextConfig.from_mapping(config, base_dir=config_path.parent)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(resolved.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(config_path)
    return resolved
