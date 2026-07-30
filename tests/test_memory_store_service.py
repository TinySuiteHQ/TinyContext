from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tinycontext.models import MemoryRow
from tinycontext.services.memory_store_service import (
    close_connection,
    embedding_storage_stats,
    fetch_candidates,
    fetch_dense_scores,
    init_db,
    insert_memories,
    session_exists,
)


class MemoryStoreServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "memories.db"

    def tearDown(self) -> None:
        close_connection(self.db_path)
        self._tmpdir.cleanup()

    def test_insert_and_fetch_candidates(self) -> None:
        rows = [
            MemoryRow(
                id="m1",
                session_id="s1",
                content="User prefers dark mode",
                tags=["preference"],
                metadata={"source": "chat"},
                created_at="2026-06-30T10:00:00Z",
            ),
            MemoryRow(
                id="m2",
                session_id="s1",
                content="Project uses FastAPI",
                tags=[],
                metadata={},
                created_at="2026-06-30T10:01:00Z",
            ),
        ]
        insert_memories(self.db_path, rows)
        all_rows = fetch_candidates(self.db_path)
        self.assertEqual(len(all_rows), 2)
        self.assertEqual(all_rows[0].id, "m2")

    def test_fetch_filters_by_session(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="m1",
                    session_id="s1",
                    content="one",
                    tags=[],
                    metadata={},
                    created_at="2026-06-30T10:00:00Z",
                ),
                MemoryRow(
                    id="m2",
                    session_id="s2",
                    content="two",
                    tags=[],
                    metadata={},
                    created_at="2026-06-30T10:01:00Z",
                ),
            ],
        )
        rows = fetch_candidates(self.db_path, session_id="s1")
        self.assertEqual([row.id for row in rows], ["m1"])

    def test_session_exists(self) -> None:
        self.assertFalse(session_exists(self.db_path, "missing"))
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="m1",
                    session_id="s1",
                    content="one",
                    tags=[],
                    metadata={},
                    created_at="2026-06-30T10:00:00Z",
                )
            ],
        )
        self.assertTrue(session_exists(self.db_path, "s1"))

    def test_vectors_are_stored_as_blobs_and_ranked_inside_sqlite(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="near",
                    session_id=None,
                    content="near",
                    tags=[],
                    metadata={},
                    created_at="2026-06-30T10:00:00Z",
                    embedding=[1.0, 0.0],
                    embedding_model="test-model",
                    embedding_dimensions=2,
                ),
                MemoryRow(
                    id="far",
                    session_id=None,
                    content="far",
                    tags=[],
                    metadata={},
                    created_at="2026-06-30T10:01:00Z",
                    embedding=[0.0, 1.0],
                    embedding_model="test-model",
                    embedding_dimensions=2,
                ),
            ],
        )
        scores = fetch_dense_scores(
            self.db_path,
            [1.0, 0.0],
            embedding_model="test-model",
        )
        self.assertEqual(list(scores), ["near", "far"])
        self.assertAlmostEqual(scores["near"], 1.0)
        self.assertAlmostEqual(scores["far"], 0.0)
        close_connection(self.db_path)
        connection = sqlite3.connect(self.db_path)
        try:
            storage_type = connection.execute(
                "SELECT typeof(embedding) FROM memories WHERE id = 'near'"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(storage_type, "blob")

    def test_existing_database_is_migrated_without_losing_rows(self) -> None:
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE memories (
                  id TEXT PRIMARY KEY,
                  session_id TEXT,
                  content TEXT NOT NULL,
                  tags TEXT,
                  metadata TEXT,
                  created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO memories
                VALUES ('legacy', NULL, 'kept', '[]', '{}', '2026-01-01T00:00:00Z')
                """
            )
            connection.commit()
        finally:
            connection.close()
        init_db(self.db_path)
        self.assertEqual(fetch_candidates(self.db_path)[0].id, "legacy")
        self.assertEqual(
            embedding_storage_stats(self.db_path),
            {"total": 1, "embedded": 0},
        )
