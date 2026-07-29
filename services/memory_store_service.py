from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  session_id TEXT,
  content TEXT NOT NULL,
  tags TEXT,
  metadata TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""

_BUSY_TIMEOUT_MS = 5000


@dataclass(frozen=True)
class MemoryRow:
    id: str
    session_id: str | None
    content: str
    tags: list[str]
    metadata: dict[str, Any]
    created_at: str


class _ConnectionPool:
    """Thread-safe, single-connection-per-path pool with WAL mode.

    SQLite supports concurrent readers with WAL, but only one writer at a time.
    This pool serialises all access to a given database file through a lock,
    preventing "database is locked" errors from concurrent async handlers.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._connections: dict[Path, sqlite3.Connection] = {}
        self._locks: dict[Path, threading.Lock] = {}

    def _get_or_create(self, db_path: Path) -> tuple[sqlite3.Connection, threading.Lock]:
        with self._guard:
            if db_path not in self._connections:
                db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(
                    db_path,
                    check_same_thread=False,
                    timeout=_BUSY_TIMEOUT_MS / 1000,
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
                conn.executescript(_SCHEMA_SQL)
                self._connections[db_path] = conn
                self._locks[db_path] = threading.Lock()
            return self._connections[db_path], self._locks[db_path]

    def execute(
        self,
        db_path: Path,
        fn: Any,
    ) -> Any:
        """Run *fn(conn)* while holding the per-path lock."""
        conn, lock = self._get_or_create(db_path)
        with lock:
            return fn(conn)

    def close(self, db_path: Path) -> None:
        """Close and remove the cached connection for *db_path*."""
        with self._guard:
            conn = self._connections.pop(db_path, None)
            self._locks.pop(db_path, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


_pool = _ConnectionPool()


def close_connection(db_path: Path) -> None:
    """Close the pooled connection for *db_path*.

    Use in tests or shutdown hooks to release the database file so it can
    be deleted on Windows.
    """
    _pool.close(db_path)


def init_db(db_path: Path) -> None:
    """Ensure the database and schema exist.

    Safe to call multiple times; the pool creates the schema on first access.
    """
    _pool._get_or_create(db_path)


def insert_memories(db_path: Path, rows: list[MemoryRow]) -> None:
    def _insert(conn: sqlite3.Connection) -> None:
        conn.executemany(
            """
            INSERT INTO memories (id, session_id, content, tags, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.id,
                    row.session_id,
                    row.content,
                    json.dumps(row.tags),
                    json.dumps(row.metadata),
                    row.created_at,
                )
                for row in rows
            ],
        )
        conn.commit()

    _pool.execute(db_path, _insert)


def _row_to_memory(row: sqlite3.Row) -> MemoryRow:
    tags_raw = row["tags"]
    metadata_raw = row["metadata"]
    tags = json.loads(tags_raw) if tags_raw else []
    metadata = json.loads(metadata_raw) if metadata_raw else {}
    if not isinstance(tags, list):
        tags = []
    if not isinstance(metadata, dict):
        metadata = {}
    return MemoryRow(
        id=str(row["id"]),
        session_id=row["session_id"],
        content=str(row["content"]),
        tags=[str(tag) for tag in tags],
        metadata=metadata,
        created_at=str(row["created_at"]),
    )


def fetch_candidates(
    db_path: Path,
    *,
    session_id: str | None = None,
    limit: int | None = None,
) -> list[MemoryRow]:
    def _fetch(conn: sqlite3.Connection) -> list[MemoryRow]:
        query = "SELECT id, session_id, content, tags, metadata, created_at FROM memories"
        params: list[Any] = []
        if session_id is not None:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        rows = conn.execute(query, params).fetchall()
        return [_row_to_memory(row) for row in rows]

    return _pool.execute(db_path, _fetch)


def session_exists(db_path: Path, session_id: str) -> bool:
    def _exists(conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT 1 FROM memories WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
        return row is not None

    return _pool.execute(db_path, _exists)
