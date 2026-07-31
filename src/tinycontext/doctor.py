"""Check TinyContext configuration, storage, and SQLite readiness."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

from tinycontext.core import describe_embedding_drift
from tinycontext.services.context_config_service import (
    load_context_config,
    resolve_context_config_path,
)
from tinycontext.services.embedding_service import (
    onnx_bundle_ready,
    resolve_local_embedding_model_spec,
)
from tinycontext.services.memory_store_service import sqlite_vec_version


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _check_writable_directory(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"{path} could not be created: {exc}"
    if os.access(path, os.W_OK):
        return True, f"{path} is writable"
    return False, f"{path} is not writable"


def _check_sqlite() -> tuple[bool, str]:
    try:
        with sqlite3.connect(":memory:") as connection:
            connection.execute("SELECT 1").fetchone()
    except sqlite3.Error as exc:
        return False, f"SQLite self-check failed: {exc}"
    return True, f"SQLite {sqlite3.sqlite_version} is available"


def _check_sqlite_vec() -> tuple[bool, str]:
    try:
        version = sqlite_vec_version()
    except (ImportError, OSError, sqlite3.Error) as exc:
        return False, f"sqlite-vec self-check failed: {exc}"
    return True, f"sqlite-vec {version} is available"


def run() -> int:
    try:
        config_path = resolve_context_config_path()
        config = load_context_config()
    except (OSError, ValueError) as exc:
        _log(f"config: INVALID - {exc}")
        return 1

    _log(
        f"config: {config_path} "
        f"({'found' if config_path.exists() else 'not found, using built-in defaults'})"
    )
    db_path = Path(str(config["memory_db_path"]))
    model_spec = resolve_local_embedding_model_spec(
        str(config.get("embedding_model", "fast")),
        models_dir=str(config.get("models_dir") or ""),
    )
    _log(f"database: {db_path}")
    _log(f"embedding model: {model_spec.requested_model} ({model_spec.repo_id})")
    _log(f"model directory: {model_spec.local_dir}")
    if onnx_bundle_ready(
        model_spec.requested_model,
        models_dir=model_spec.local_dir.parent,
    ):
        _log("model bundle: ready")
    else:
        _log("model bundle: not downloaded; first server start or API use will fetch it")
    drift_warning = describe_embedding_drift(config)
    if drift_warning:
        _log(f"embedding drift: WARNING - {drift_warning}")
    else:
        _log("embedding drift: none")
    checks = [
        ("data directory", *_check_writable_directory(db_path.parent)),
        ("sqlite", *_check_sqlite()),
        ("sqlite-vec", *_check_sqlite_vec()),
    ]
    all_ok = True
    for name, ok, message in checks:
        _log(f"{name}: {'ok' if ok else 'FAILED'} - {message}")
        all_ok = all_ok and ok
    _log("all checks passed" if all_ok else "some checks failed")
    return 0 if all_ok else 1
