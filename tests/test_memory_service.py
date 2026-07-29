from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.memory_service import (
    EmptyMemoryError,
    MemoryInput,
    SessionNotFoundError,
    recall_memories,
    save_memories,
)
from services.memory_store_service import close_connection


class MemoryServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.config = {
            "memory_db_path": str(Path(self._tmpdir.name) / "memories.db"),
            "recall_top_k": 10,
            "recall_max_tokens": 2000,
            "encoding_name": "o200k_base",
        }

    def tearDown(self) -> None:
        close_connection(Path(self.config["memory_db_path"]))
        self._tmpdir.cleanup()

    def test_save_memories_persists_rows(self) -> None:
        payload = save_memories(
            [MemoryInput(content="User likes tea", tags=["preference"])],
            session_id="session-a",
            config=self.config,
        )
        self.assertEqual(len(payload["saved"]), 1)
        self.assertEqual(payload["saved"][0]["session_id"], "session-a")
        self.assertGreater(payload["saved"][0]["content_tokens"], 0)

    def test_save_rejects_empty_content(self) -> None:
        with self.assertRaises(EmptyMemoryError):
            save_memories(
                [MemoryInput(content="   ")],
                config=self.config,
            )

    def test_recall_returns_ranked_memories(self) -> None:
        save_memories(
            [
                MemoryInput(content="User prefers Python for backend work"),
                MemoryInput(content="User enjoys hiking on weekends"),
            ],
            session_id="session-a",
            config=self.config,
        )
        payload = recall_memories(
            "Python backend",
            session_id="session-a",
            config=self.config,
        )
        self.assertEqual(payload["query"], "Python backend")
        self.assertGreaterEqual(len(payload["memories"]), 1)
        self.assertIn("Python", payload["memories"][0]["content"])

    def test_recall_unknown_session_raises(self) -> None:
        with self.assertRaises(SessionNotFoundError):
            recall_memories(
                "anything",
                session_id="missing",
                config=self.config,
            )
