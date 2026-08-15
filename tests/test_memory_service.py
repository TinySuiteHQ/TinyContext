from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext import MemoryInput, delete_memory, recall_memories, save_memories
from tinycontext.core import describe_embedding_drift
from tinycontext.errors import (
    AmbiguousMemoryReferenceError,
    EmptyMemoryError,
    MemoryNotFoundError,
    SessionNotFoundError,
)
from tinycontext.models import MemoryRow
from tinycontext.services.memory_store_service import close_connection, insert_memories
from tests.embedding_fakes import fake_embed_texts, start_fake_embeddings


class MemoryServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        start_fake_embeddings(self)
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
            [MemoryInput(content="User likes tea")],
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
        self.assertEqual(payload["memories"][0]["rank"], 1)
        self.assertEqual(payload["memories"][0]["relevance"], "high")
        self.assertEqual(payload["memories"][0]["scores"]["rrf"], 1.0)
        self.assertIn("dense", payload["memories"][0]["scores"])
        self.assertIn("bm25", payload["memories"][0]["scores"])
        self.assertIn("created_at", payload["memories"][0])
        self.assertIn("current_time", payload)

    def test_recall_unknown_session_raises(self) -> None:
        with self.assertRaises(SessionNotFoundError):
            recall_memories(
                "anything",
                session_id="missing",
                config=self.config,
            )

    def test_describe_embedding_drift_is_none_before_any_save(self) -> None:
        self.assertIsNone(describe_embedding_drift(self.config))

    def test_describe_embedding_drift_is_none_when_model_unchanged(self) -> None:
        save_memories([MemoryInput(content="User likes tea")], config=self.config)
        self.assertIsNone(describe_embedding_drift(self.config))

    def test_describe_embedding_drift_flags_a_changed_model(self) -> None:
        save_memories([MemoryInput(content="User likes tea")], config=self.config)
        changed_config = dict(self.config, embedding_model="balanced")
        warning = describe_embedding_drift(changed_config)
        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertIn("1 of 1", warning)
        self.assertIn("balanced", warning)

    def test_recall_surfaces_notice_while_background_reindex_runs(self) -> None:
        save_memories([MemoryInput(content="User likes tea")], config=self.config)
        release = threading.Event()
        started = threading.Event()

        def slow_embed(inputs, **_kwargs):
            started.set()
            release.wait(timeout=5)
            return fake_embed_texts(inputs)

        changed_config = dict(self.config, embedding_model="balanced")
        with patch(
            "tinycontext.services.embedding_reindex_service.embed_texts",
            side_effect=slow_embed,
        ):
            payload = recall_memories("tea", config=changed_config)
            self.assertTrue(started.wait(timeout=5))
            self.assertIn("notice", payload)
            self.assertIn("in progress", payload["notice"])
            release.set()

    def test_delete_memory_removes_it_from_recall(self) -> None:
        payload = save_memories(
            [MemoryInput(content="User likes tea")],
            config=self.config,
        )
        memory_id = payload["saved"][0]["id"]
        ref = payload["saved"][0]["ref"]
        result = delete_memory(memory_id, config=self.config)
        self.assertEqual(result, {"id": memory_id, "ref": ref, "deleted": True})
        recalled = recall_memories("tea", config=self.config)
        self.assertEqual(recalled["memories"], [])

    def test_delete_missing_memory_raises(self) -> None:
        with self.assertRaises(MemoryNotFoundError):
            delete_memory("missing-id", config=self.config)

    def test_delete_rejects_empty_id(self) -> None:
        with self.assertRaises(EmptyMemoryError):
            delete_memory("   ", config=self.config)

    def test_delete_memory_by_short_ref(self) -> None:
        payload = save_memories(
            [MemoryInput(content="delete me by ref")],
            config=self.config,
        )
        record = payload["saved"][0]
        result = delete_memory(record["ref"], config=self.config)
        self.assertEqual(result["id"], record["id"])
        self.assertTrue(result["deleted"])
        recalled = recall_memories("delete me by ref", config=self.config)
        self.assertEqual(recalled["memories"], [])

    def test_delete_memory_unknown_ref_raises_not_found(self) -> None:
        with self.assertRaises(MemoryNotFoundError):
            delete_memory("aaaaaaaaaaaa", config=self.config)

    def test_delete_memory_ambiguous_ref_raises(self) -> None:
        db_path = Path(self.config["memory_db_path"])
        insert_memories(
            db_path,
            [
                MemoryRow(
                    id="aaaaaaaaaaaa0001",
                    session_id=None,
                    content="one",
                    created_at="2026-01-01T00:00:00Z",
                ),
                MemoryRow(
                    id="aaaaaaaaaaaa0002",
                    session_id=None,
                    content="two",
                    created_at="2026-01-01T00:00:00Z",
                ),
            ],
        )
        with self.assertRaises(AmbiguousMemoryReferenceError):
            delete_memory("aaaaaaaaaaaa", config=self.config)

    def test_delete_memory_still_accepts_full_uuid(self) -> None:
        payload = save_memories(
            [MemoryInput(content="delete me by full id")],
            config=self.config,
        )
        memory_id = payload["saved"][0]["id"]
        result = delete_memory(memory_id, config=self.config)
        self.assertEqual(result["id"], memory_id)
        self.assertTrue(result["deleted"])

    def test_dense_similarity_recalls_a_semantic_match(self) -> None:
        save_memories(
            [
                MemoryInput(content="Dog nutrition and exercise notes"),
                MemoryInput(content="Quarterly finance spreadsheet"),
            ],
            config=self.config,
        )
        payload = recall_memories("canine health", config=self.config)
        self.assertIn("Dog", payload["memories"][0]["content"])

    def test_relevance_reflects_absolute_similarity_not_just_rank(self) -> None:
        # RRF fuses ranks, not scores: the sole candidate in a pool always
        # ranks #1 in both signals and gets rrf_similarity == 1.0, even when
        # it has nothing to do with the query. relevance must not be "high"
        # (or even "medium") purely on that rank-based signal.
        save_memories(
            [MemoryInput(content="User enjoys hiking on weekends")],
            config=self.config,
        )
        payload = recall_memories("Python backend framework", config=self.config)
        self.assertEqual(len(payload["memories"]), 1)
        memory = payload["memories"][0]
        self.assertEqual(memory["scores"]["rrf"], 1.0)
        self.assertEqual(memory["scores"]["dense"], 0.0)
        self.assertEqual(memory["relevance"], "low")
