"""Filesystem and environment configuration for TinyContext server processes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tinycontext.config import TinyContextConfig, normalize_config, save_config
from tinycontext.paths import native_config_path


def resolve_context_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    env_path = os.environ.get("TINYCONTEXT_CONFIG_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return native_config_path()


def _environment_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    db_path = os.environ.get("TINYCONTEXT_MEMORY_DB_PATH", "").strip()
    if db_path:
        overrides["memory_db_path"] = str(Path(db_path).expanduser().resolve())

    for env_name, config_name in (
        ("TINYCONTEXT_RECALL_TOP_K", "recall_top_k"),
        ("TINYCONTEXT_RECALL_MAX_TOKENS", "recall_max_tokens"),
        ("TINYCONTEXT_ENCODING_NAME", "encoding_name"),
        ("TINYCONTEXT_MODELS_DIR", "models_dir"),
        ("TINYCONTEXT_EMBEDDING_MODEL", "embedding_model"),
        ("TINYCONTEXT_EMBEDDING_BACKEND", "embedding_backend"),
        ("TINYCONTEXT_EMBEDDING_OPENAI_ENV_FILE", "embedding_openai_env_file"),
        ("TINYCONTEXT_EMBEDDING_BATCH_SIZE", "embedding_batch_size"),
        ("TINYCONTEXT_RECALL_RRF_CUTOFF", "recall_rrf_cutoff"),
        ("TINYCONTEXT_RECALL_DENSE_WEIGHT", "recall_dense_weight"),
        ("TINYCONTEXT_RECALL_RRF_K", "recall_rrf_k"),
        ("TINYCONTEXT_DENSE_QUERY_PREFIX", "dense_query_prefix"),
        ("TINYCONTEXT_DENSE_DOCUMENT_PREFIX", "dense_document_prefix"),
    ):
        raw_value = os.environ.get(env_name, "")
        value = (
            raw_value
            if config_name in {"dense_query_prefix", "dense_document_prefix"}
            else raw_value.strip()
        )
        if value:
            overrides[config_name] = value
    return overrides


def load_context_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_context_config_path(path)
    config = (
        TinyContextConfig.from_json(config_path)
        if config_path.exists()
        else TinyContextConfig()
    )
    overrides = _environment_overrides()
    return (
        config.with_overrides(overrides, base_dir=config_path.parent).to_dict()
        if overrides
        else config.to_dict()
    )


def save_context_config(
    raw: Mapping[str, Any],
    path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = resolve_context_config_path(path)
    existing = (
        TinyContextConfig.from_json(config_path).to_dict()
        if config_path.exists()
        else {}
    )
    merged = dict(existing)
    merged.update(raw)
    return save_config(
        TinyContextConfig.from_mapping(merged, base_dir=config_path.parent),
        config_path,
    ).to_dict()


def resolve_memory_db_path(config: Mapping[str, Any] | None = None) -> Path:
    resolved = load_context_config() if config is None else normalize_config(config)
    return Path(str(resolved["memory_db_path"])).expanduser().resolve()


def context_tokenizer_name(config: Mapping[str, Any] | None = None) -> str:
    resolved = load_context_config() if config is None else normalize_config(config)
    return str(resolved["encoding_name"])
