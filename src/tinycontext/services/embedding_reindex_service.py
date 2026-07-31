"""Background re-embedding when the configured embedding model changes.

When ``embedding_model`` no longer matches what's stored for a memory row,
the naive fix is to re-embed everything inline on the next recall -- which
blocks that call for as long as the whole backlog takes. Instead, a daemon
thread walks the backlog in small batches while callers get an immediate,
informative notice instead of a stall.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from tinycontext.services.embedding_service import embed_texts, embedding_model_key
from tinycontext.services.memory_store_service import (
    embedding_model_mismatch_count,
    fetch_candidates,
    update_memory_embeddings,
)

_REINDEX_BATCH_SIZE = 16


@dataclass
class _Job:
    model_key: str
    total: int
    started_at: float = field(default_factory=time.monotonic)
    completed: int = 0
    finished: bool = False
    error: str | None = None

    @property
    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    @property
    def eta_seconds(self) -> float | None:
        if self.completed <= 0 or self.finished:
            return None
        rate = self.completed / self.elapsed if self.elapsed > 0 else 0.0
        if rate <= 0:
            return None
        return (self.total - self.completed) / rate


_lock = threading.Lock()
_jobs: dict[str, _Job] = {}
_threads: dict[str, threading.Thread] = {}


def _job_key(db_path: Path) -> str:
    return str(Path(db_path))


def _run(
    key: str,
    db_path: Path,
    *,
    embedding_model: str,
    models_dir: Path,
    embedding_batch_size: int,
    document_prefix: str,
    model_key: str,
) -> None:
    try:
        stale = [
            row
            for row in fetch_candidates(db_path)
            if row.embedding_model != model_key
        ]
        with _lock:
            job = _jobs.get(key)
            if job is not None:
                job.total = len(stale)

        batch_size = max(1, min(_REINDEX_BATCH_SIZE, embedding_batch_size))
        for start in range(0, len(stale), batch_size):
            batch = stale[start : start + batch_size]
            vectors = embed_texts(
                [document_prefix + row.content for row in batch],
                embedding_model=embedding_model,
                models_dir=models_dir,
                batch_size=embedding_batch_size,
            )
            update_memory_embeddings(
                db_path,
                [
                    (row.id, vector)
                    for row, vector in zip(batch, vectors, strict=True)
                ],
                embedding_model=model_key,
            )
            with _lock:
                job = _jobs.get(key)
                if job is not None:
                    job.completed += len(batch)
    except Exception as exc:  # pragma: no cover - defensive background path
        with _lock:
            job = _jobs.get(key)
            if job is not None:
                job.error = str(exc)
    finally:
        with _lock:
            job = _jobs.get(key)
            if job is not None:
                job.finished = True


def ensure_background_reindex(
    db_path: Path,
    *,
    embedding_model: str,
    models_dir: Path,
    embedding_batch_size: int,
    document_prefix: str,
) -> None:
    """Start a background re-embed job if the store has drifted and none is running."""
    model_key = embedding_model_key(
        embedding_model,
        models_dir=models_dir,
        document_prefix=document_prefix,
    )
    key = _job_key(db_path)
    with _lock:
        thread = _threads.get(key)
        if thread is not None and thread.is_alive():
            return
        job = _jobs.get(key)
        if job is not None and not job.finished:
            return

    mismatched = embedding_model_mismatch_count(db_path, model_key)
    if mismatched <= 0:
        with _lock:
            _jobs.pop(key, None)
            _threads.pop(key, None)
        return

    with _lock:
        _jobs[key] = _Job(model_key=model_key, total=mismatched)
        thread = threading.Thread(
            target=_run,
            args=(key, db_path),
            kwargs=dict(
                embedding_model=embedding_model,
                models_dir=models_dir,
                embedding_batch_size=embedding_batch_size,
                document_prefix=document_prefix,
                model_key=model_key,
            ),
            daemon=True,
            name="tinycontext-reindex",
        )
        _threads[key] = thread
        thread.start()


def reindex_notice(db_path: Path) -> str | None:
    """A caller-facing status message, or None when nothing is in flight."""
    key = _job_key(db_path)
    with _lock:
        job = _jobs.get(key)
        if job is None or job.finished:
            return None
        completed, total, eta = job.completed, job.total, job.eta_seconds

    eta_text = f"~{max(1, int(eta))}s" if eta is not None else "estimating"
    return (
        "Embedding model changed; background re-embedding is in progress "
        f"({completed}/{total} memories done, ETA {eta_text}). Recall results "
        "may be incomplete until this finishes. Wait and retry, or proceed "
        "with the partial results below."
    )


def is_reindex_active(db_path: Path) -> bool:
    return reindex_notice(db_path) is not None
