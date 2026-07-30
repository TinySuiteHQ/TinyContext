from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tinycontext.models import MemoryRow
from tinycontext.services.memory_store_service import (
    close_connection,
    fetch_candidates,
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
