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
    record_recall_hits,
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
        self.assertEqual(payload["matched_count"], 2)
        self.assertLess(len(payload["memories"]), payload["matched_count"])

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
        self.assertEqual(payload["matched_count"], 0)

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

    def test_access_weight_narrows_gap_between_tied_candidates(self) -> None:
        # Different wording, same single-group token overlap ("likes" vs
        # "preference" both live in the same fake-embedder semantic group),
        # so both rows backfill to an identical embedding vector -- and
        # neither shares any term with the query, so BM25 ties at 0 too.
        # RRF is rank- not value-based, so a raw tie still resolves to a
        # rank-1/rank-2 split; "newer" (later created_at) sorts first and
        # wins that split at access_weight=0. "older" only has more
        # recall_count going for it. With default dense_weight=0.5 (so
        # sparse_weight=0.5 too) and access_weight pushed to its 1.0 max,
        # the rank-1-vs-rank-2 advantage access buys "older" exactly cancels
        # "newer"'s combined dense+sparse advantage -- provably an exact tie.
        db_path = Path(self.config["memory_db_path"])
        insert_memories(
            db_path,
            [
                MemoryRow(
                    id="older-but-recalled",
                    session_id=None,
                    content="User has a preference for it",
                    created_at="2026-01-01T00:00:00Z",
                ),
                MemoryRow(
                    id="newer-but-unrecalled",
                    session_id=None,
                    content="User likes it",
                    created_at="2026-01-01T00:00:01Z",
                ),
            ],
        )
        record_recall_hits(db_path, ["older-but-recalled"], "2026-01-01T00:00:02Z")

        baseline = memory_recall_run(
            "something unrelated xyz",
            session_id=None,
            max_tokens=100,
            top_k=2,
            db_path=db_path,
            encoding_name=self.config["encoding_name"],
            access_weight=0.0,
        )
        self.assertEqual(baseline["memories"][0]["content"], "User likes it")
        self.assertGreater(
            baseline["memories"][0]["scores"]["rrf"],
            baseline["memories"][1]["scores"]["rrf"],
        )

        boosted = memory_recall_run(
            "something unrelated xyz",
            session_id=None,
            max_tokens=100,
            top_k=2,
            db_path=db_path,
            encoding_name=self.config["encoding_name"],
            access_weight=1.0,
        )
        self.assertAlmostEqual(
            boosted["memories"][0]["scores"]["rrf"],
            boosted["memories"][1]["scores"]["rrf"],
        )
