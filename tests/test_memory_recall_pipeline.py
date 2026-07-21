from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipelines.memory_recall import memory_recall_run
from services.memory_service import MemoryInput, save_memories
from services.memory_store_service import close_connection


class MemoryRecallPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.config = {
            "memory_db_path": str(Path(self._tmpdir.name) / "memories.db"),
            "recall_top_k": 2,
            "recall_max_tokens": 8,
            "encoding_name": "o200k_base",
        }

    def tearDown(self) -> None:
        close_connection(Path(self.config["memory_db_path"]))
        self._tmpdir.cleanup()

    def test_pipeline_trims_to_token_budget(self) -> None:
        save_memories(
            [
                MemoryInput(content="alpha beta gamma delta"),
                MemoryInput(content="epsilon zeta eta theta"),
            ],
            config=self.config,
        )
        payload = memory_recall_run(
            "alpha",
            session_id=None,
            max_tokens=4,
            top_k=2,
            config=self.config,
        )
        self.assertTrue(payload["truncated"])
        self.assertLessEqual(payload["total_tokens"], 8)

    def test_pipeline_returns_empty_when_no_memories(self) -> None:
        payload = memory_recall_run(
            "anything",
            session_id=None,
            max_tokens=100,
            top_k=5,
            config=self.config,
        )
        self.assertEqual(payload["memories"], [])
        self.assertEqual(payload["total_tokens"], 0)
