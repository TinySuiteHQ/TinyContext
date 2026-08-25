"""Public, transport-independent TinyContext configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinycontext.paths import native_data_dir, native_memory_db_path, native_models_dir


def default_config() -> dict[str, Any]:
    return {
        "memory_db_path": str(native_memory_db_path()),
        "recall_top_k": 10,
        "recall_max_tokens": 2000,
        "profile_max_tokens": 500,
        "encoding_name": "o200k_base",
        "models_dir": str(native_models_dir()),
        "embedding_model": "balanced",
        "embedding_backend": "onnx",
        "embedding_openai_env_file": ".env",
        "embedding_batch_size": 32,
        "recall_rrf_cutoff": 0.0,
        "recall_dense_weight": 0.5,
        "recall_rrf_k": 60,
        "dense_query_prefix": "",
        "dense_document_prefix": "",
        "dedup_similarity_threshold": 0.95,
        "dedup_review_similarity_threshold": 0.80,
        "recall_access_weight": 0.0,
        "save_length_notice_tokens": 800,
    }


def _resolve_path(
    value: Any,
    *,
    default: Path,
    base_dir: Path | None,
) -> str:
    raw = str(value or "").strip()
    path = Path(raw).expanduser() if raw else default
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
    resolved_base_dir = Path(base_dir).expanduser().resolve() if base_dir else None
    config["memory_db_path"] = _resolve_path(
        config["memory_db_path"],
        default=native_memory_db_path(),
        base_dir=resolved_base_dir,
    )
    config["models_dir"] = _resolve_path(
        config["models_dir"],
        default=native_models_dir(),
        base_dir=resolved_base_dir,
    )

    for key in (
        "recall_top_k",
        "recall_max_tokens",
        "profile_max_tokens",
        "embedding_batch_size",
        "save_length_notice_tokens",
    ):
        try:
            value = int(config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if value < 1:
            raise ValueError(f"{key} must be at least 1")
        config[key] = value

    try:
        dense_weight = float(config["recall_dense_weight"])
    except (TypeError, ValueError) as exc:
        raise ValueError("recall_dense_weight must be a number") from exc
    if dense_weight <= 0.0 or dense_weight > 1.0:
        raise ValueError("recall_dense_weight must be greater than 0 and at most 1")
    config["recall_dense_weight"] = dense_weight

    try:
        rrf_cutoff = float(config["recall_rrf_cutoff"])
    except (TypeError, ValueError) as exc:
        raise ValueError("recall_rrf_cutoff must be a number") from exc
    if rrf_cutoff < 0.0 or rrf_cutoff > 1.0:
        raise ValueError("recall_rrf_cutoff must be between 0 and 1")
    config["recall_rrf_cutoff"] = rrf_cutoff

    try:
        rrf_k = int(config["recall_rrf_k"])
    except (TypeError, ValueError) as exc:
        raise ValueError("recall_rrf_k must be an integer") from exc
    if rrf_k < 0:
        raise ValueError("recall_rrf_k must be at least 0")
    config["recall_rrf_k"] = rrf_k

    try:
        dedup_threshold = float(config["dedup_similarity_threshold"])
    except (TypeError, ValueError) as exc:
        raise ValueError("dedup_similarity_threshold must be a number") from exc
    if dedup_threshold <= 0.0 or dedup_threshold > 1.0:
        raise ValueError("dedup_similarity_threshold must be greater than 0 and at most 1")
    config["dedup_similarity_threshold"] = dedup_threshold

    try:
        dedup_review_threshold = float(config["dedup_review_similarity_threshold"])
    except (TypeError, ValueError) as exc:
        raise ValueError("dedup_review_similarity_threshold must be a number") from exc
    if dedup_review_threshold <= 0.0 or dedup_review_threshold > 1.0:
        raise ValueError(
            "dedup_review_similarity_threshold must be greater than 0 and at most 1"
        )
    config["dedup_review_similarity_threshold"] = dedup_review_threshold

    try:
        access_weight = float(config["recall_access_weight"])
    except (TypeError, ValueError) as exc:
        raise ValueError("recall_access_weight must be a number") from exc
    if access_weight < 0.0 or access_weight > 1.0:
        raise ValueError("recall_access_weight must be between 0 and 1")
    config["recall_access_weight"] = access_weight

    for key in ("encoding_name", "embedding_model", "embedding_openai_env_file"):
        value = str(config[key] or "").strip()
        if not value:
            raise ValueError(f"{key} must not be empty")
        config[key] = value
    from tinycontext.services.embedding_service import normalize_embedding_backend

    config["embedding_backend"] = normalize_embedding_backend(str(config["embedding_backend"]))
    for key in ("dense_query_prefix", "dense_document_prefix"):
        config[key] = str(config[key] or "")
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
