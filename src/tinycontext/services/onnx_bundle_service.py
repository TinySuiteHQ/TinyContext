"""Download and persist local ONNX embedding bundles."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import snapshot_download

from tinycontext.services.embedding_service import (
    clear_onnx_runtime_cache,
    onnx_bundle_ready,
    resolve_local_embedding_model_spec,
)


_LOCK_NAME = ".download.lock"
_STALE_LOCK_SECONDS = 1800.0
_LOCK_WAIT_SECONDS = 3600.0
_POLL_SECONDS = 0.25


@contextmanager
def _exclusive_bundle_lock(bundle_dir: Path):
    bundle_dir.mkdir(parents=True, exist_ok=True)
    lock_path = bundle_dir / _LOCK_NAME
    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    while True:
        try:
            descriptor = os.open(
                str(lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for ONNX bundle lock {lock_path}"
                ) from None
            try:
                if time.time() - lock_path.stat().st_mtime > _STALE_LOCK_SECONDS:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            time.sleep(_POLL_SECONDS)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _copy_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(source)
        if relative_path.parts and relative_path.parts[0] == ".cache":
            continue
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def ensure_onnx_bundle_sync(
    embedding_model: str = "fast",
    *,
    models_dir: str | Path | None = None,
) -> None:
    if onnx_bundle_ready(embedding_model, models_dir=models_dir):
        return

    spec = resolve_local_embedding_model_spec(
        embedding_model,
        models_dir=models_dir,
    )
    with _exclusive_bundle_lock(spec.local_dir):
        if onnx_bundle_ready(embedding_model, models_dir=models_dir):
            return
        print(
            "[tinycontext] downloading ONNX embedding bundle "
            f"model={spec.requested_model!r} repo={spec.repo_id!r}; "
            "see the Hugging Face model card for its license.",
            file=sys.stderr,
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="tinycontext-onnx-download-") as raw:
            temporary_dir = Path(raw)
            snapshot_download(
                repo_id=spec.repo_id,
                local_dir=str(temporary_dir),
                local_dir_use_symlinks=False,
                allow_patterns=list(spec.allow_patterns),
            )
            _copy_tree(temporary_dir, spec.local_dir)
        if not onnx_bundle_ready(embedding_model, models_dir=models_dir):
            raise RuntimeError(
                f"ONNX bundle for {spec.requested_model!r} is incomplete after "
                f"download from {spec.repo_id!r} under {spec.local_dir}"
            )
    clear_onnx_runtime_cache()
