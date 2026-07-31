from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tinycontext import MemoryInput, save_memories
from tinycontext.pipelines.memory_recall import memory_recall_run
from tinycontext.services.memory_store_service import close_connection
from tinycontext.services.memory_store_service import (
    embedding_storage_stats,
    insert_memories,
)
from tinycontext.models import MemoryRow
from tests.embedding_fakes import start_fake_embeddings


class MemoryRecallPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        start_fake_embeddings(self)
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
            db_path=Path(self.config["memory_db_path"]),
            encoding_name=self.config["encoding_name"],
        )
        self.assertTrue(payload["truncated"])
        self.assertLessEqual(payload["total_tokens"], 8)

    def test_pipeline_returns_empty_when_no_memories(self) -> None:
        payload = memory_recall_run(
            "anything",
            session_id=None,
            max_tokens=100,
            top_k=5,
            db_path=Path(self.config["memory_db_path"]),
            encoding_name=self.config["encoding_name"],
        )
        self.assertEqual(payload["memories"], [])
        self.assertEqual(payload["total_tokens"], 0)

    def test_pipeline_backfills_embeddings_for_legacy_rows(self) -> None:
        insert_memories(
            Path(self.config["memory_db_path"]),
            [
                MemoryRow(
                    id="legacy",
                    session_id=None,
                    content="legacy sqlite memory",
                    created_at="2026-01-01T00:00:00Z",
                )
            ],
        )
        memory_recall_run(
            "database",
            session_id=None,
            max_tokens=100,
            top_k=5,
            db_path=Path(self.config["memory_db_path"]),
            encoding_name=self.config["encoding_name"],
        )
        self.assertEqual(
            embedding_storage_stats(Path(self.config["memory_db_path"])),
            {"total": 1, "embedded": 1},
        )

    def test_pipeline_applies_normalized_rrf_cutoff_before_top_k(self) -> None:
        save_memories(
            [
                MemoryInput(content="Python backend FastAPI preference"),
                MemoryInput(content="Hiking outside on weekends"),
                MemoryInput(content="Quarterly finance spreadsheet"),
            ],
            config=self.config,
        )
        payload = memory_recall_run(
            "Python backend",
            session_id=None,
            max_tokens=100,
            top_k=3,
            db_path=Path(self.config["memory_db_path"]),
            encoding_name=self.config["encoding_name"],
            rrf_similarity_cutoff=0.99,
        )
        self.assertEqual(len(payload["memories"]), 1)
        self.assertIn("Python", payload["memories"][0]["content"])
        self.assertEqual(payload["memories"][0]["scores"]["rrf"], 1.0)
