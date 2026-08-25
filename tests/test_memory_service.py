from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext import (
    MemoryInput,
    delete_memory,
    get_memory,
    list_memories,
    recall_memories,
    save_memories,
    update_memory,
)
from tinycontext.core import describe_embedding_drift
from tinycontext.errors import (
    AmbiguousMemoryReferenceError,
    EmptyMemoryError,
    InvalidMemoryKindError,
    MemoryAlreadySupersededError,
    MemoryNotFoundError,
    RecallBudgetError,
    SessionNotFoundError,
)
from tinycontext.models import MemoryRow
from tinycontext.services.memory_store_service import (
    close_connection,
    fetch_candidates,
    insert_memories,
)
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

    def test_recent_recall_defaults_to_five_and_is_newest_first(self) -> None:
        # Seeded directly (not via save_memories) so this recall-ordering
        # test doesn't depend on save-time dedup treating any of these
        # near-identical placeholder strings as duplicates of each other.
        db_path = Path(self.config["memory_db_path"])
        insert_memories(
            db_path,
            [
                MemoryRow(
                    id=f"recent-{index:02d}",
                    session_id=None,
                    content=f"memory {index}",
                    created_at=f"2026-01-01T00:00:{index:02d}Z",
                )
                for index in range(6)
            ],
        )
        payload = recall_memories(config=self.config)
        self.assertEqual(payload["mode"], "recent")
        self.assertEqual(len(payload["memories"]), 5)
        self.assertEqual(
            [memory["content"] for memory in payload["memories"]],
            [f"memory {index}" for index in range(5, 0, -1)],
        )
        self.assertEqual(
            [memory["rank"] for memory in payload["memories"]],
            [1, 2, 3, 4, 5],
        )
        self.assertNotIn("relevance", payload["memories"][0])
        self.assertNotIn("scores", payload["memories"][0])

    def test_recent_recall_supports_session_and_custom_count(self) -> None:
        save_memories(
            [MemoryInput(content="project one")], session_id="one", config=self.config
        )
        save_memories(
            [MemoryInput(content="project two")], session_id="two", config=self.config
        )
        payload = recall_memories(
            session_id="one", top_k=1, config=self.config
        )
        self.assertEqual(
            [memory["content"] for memory in payload["memories"]],
            ["project one"],
        )

    def test_recent_recall_empty_store_and_missing_session(self) -> None:
        empty = recall_memories(config=self.config)
        self.assertEqual(empty["memories"], [])
        self.assertFalse(empty["truncated"])
        with self.assertRaises(SessionNotFoundError):
            recall_memories(session_id="missing", config=self.config)

    def test_recent_recall_validates_top_k(self) -> None:
        with self.assertRaises(RecallBudgetError):
            recall_memories(top_k=0, config=self.config)

    def test_recent_recall_respects_budget_and_keeps_oversized_newest_memory(self) -> None:
        save_memories(
            [
                MemoryInput(content="older memory"),
                MemoryInput(content="newest memory with several words"),
            ],
            config=self.config,
        )
        payload = recall_memories(
            top_k=2,
            config=dict(self.config, recall_max_tokens=1),
        )
        self.assertEqual(
            [memory["content"] for memory in payload["memories"]],
            ["newest memory with several words"],
        )
        self.assertTrue(payload["truncated"])
        self.assertGreater(payload["total_tokens"], 1)

    def test_recent_recall_does_not_compute_embeddings_or_reindex(self) -> None:
        save_memories([MemoryInput(content="read without embeddings")], config=self.config)
        with (
            patch("tinycontext.core.embed_texts", side_effect=AssertionError("embedded")),
            patch(
                "tinycontext.core._kick_off_background_reindex",
                side_effect=AssertionError("reindexed"),
            ),
        ):
            payload = recall_memories(config=self.config)
        self.assertEqual(payload["memories"][0]["content"], "read without embeddings")

    def test_save_memories_skips_exact_duplicate(self) -> None:
        save_memories([MemoryInput(content="User likes tea")], config=self.config)
        payload = save_memories([MemoryInput(content="User likes tea")], config=self.config)
        self.assertEqual(payload["saved"], [])
        self.assertEqual(len(payload["skipped_duplicates"]), 1)
        self.assertIn("similarity", payload["skipped_duplicates"][0])
        recalled = recall_memories(config=self.config)
        self.assertEqual(len(recalled["memories"]), 1)

    def test_save_memories_dedups_within_same_batch(self) -> None:
        payload = save_memories(
            [
                MemoryInput(content="User likes tea"),
                MemoryInput(content="User likes tea"),
            ],
            config=self.config,
        )
        self.assertEqual(len(payload["saved"]), 1)
        self.assertEqual(len(payload["skipped_duplicates"]), 1)

    def test_save_memories_dedup_is_scoped_to_session(self) -> None:
        save_memories(
            [MemoryInput(content="User likes tea")],
            session_id="session-a",
            config=self.config,
        )
        payload = save_memories(
            [MemoryInput(content="User likes tea")],
            session_id="session-b",
            config=self.config,
        )
        self.assertEqual(len(payload["saved"]), 1)
        self.assertNotIn("skipped_duplicates", payload)

    def test_save_memories_does_not_skip_distinct_content(self) -> None:
        payload = save_memories(
            [
                MemoryInput(content="User prefers Python for backend work"),
                MemoryInput(content="User enjoys hiking on weekends"),
            ],
            config=self.config,
        )
        self.assertEqual(len(payload["saved"]), 2)
        self.assertNotIn("skipped_duplicates", payload)

    def test_save_memories_partial_similarity_not_skipped_by_default(self) -> None:
        # "sqlite storage" only partially overlaps "sqlite storage python
        # backend" under the fake embedder (shared group + an empty group),
        # giving cosine similarity ~0.707 -- below the default 0.95 floor.
        save_memories(
            [MemoryInput(content="sqlite storage python backend")],
            config=self.config,
        )
        payload = save_memories([MemoryInput(content="sqlite storage")], config=self.config)
        self.assertEqual(len(payload["saved"]), 1)
        self.assertNotIn("skipped_duplicates", payload)

    def test_save_memories_lower_threshold_skips_related_content(self) -> None:
        save_memories(
            [MemoryInput(content="sqlite storage python backend")],
            config=self.config,
        )
        payload = save_memories(
            [MemoryInput(content="sqlite storage")],
            config=dict(self.config, dedup_similarity_threshold=0.5),
        )
        self.assertEqual(payload["saved"], [])
        self.assertEqual(len(payload["skipped_duplicates"]), 1)

    def test_update_memory_supersedes_old_and_hides_it_from_recall(self) -> None:
        saved = save_memories(
            [MemoryInput(content="uses MySQL")],
            session_id="session-a",
            config=self.config,
        )
        old_id = saved["saved"][0]["id"]
        payload = update_memory(old_id, "uses Postgres now", config=self.config)
        self.assertNotEqual(payload["id"], old_id)
        self.assertEqual(payload["session_id"], "session-a")
        self.assertEqual(payload["supersedes"]["id"], old_id)

        recent = recall_memories(config=self.config)
        self.assertEqual(
            [memory["content"] for memory in recent["memories"]],
            ["uses Postgres now"],
        )

    def test_update_memory_by_short_ref(self) -> None:
        saved = save_memories([MemoryInput(content="uses MySQL")], config=self.config)
        ref = saved["saved"][0]["ref"]
        payload = update_memory(ref, "uses Postgres now", config=self.config)
        self.assertEqual(payload["supersedes"]["ref"], ref)

    def test_update_memory_missing_id_raises(self) -> None:
        with self.assertRaises(MemoryNotFoundError):
            update_memory("missing-id", "new content", config=self.config)

    def test_update_memory_rejects_empty_content(self) -> None:
        saved = save_memories([MemoryInput(content="uses MySQL")], config=self.config)
        with self.assertRaises(EmptyMemoryError):
            update_memory(saved["saved"][0]["id"], "   ", config=self.config)

    def test_update_memory_twice_on_original_ref_raises(self) -> None:
        saved = save_memories([MemoryInput(content="uses MySQL")], config=self.config)
        old_id = saved["saved"][0]["id"]
        update_memory(old_id, "uses Postgres now", config=self.config)
        with self.assertRaises(MemoryAlreadySupersededError):
            update_memory(old_id, "uses SQLite now", config=self.config)

    def test_delete_memory_successor_restores_predecessor_visibility(self) -> None:
        saved = save_memories([MemoryInput(content="uses MySQL")], config=self.config)
        old_id = saved["saved"][0]["id"]
        updated = update_memory(old_id, "uses Postgres now", config=self.config)
        delete_memory(updated["id"], config=self.config)
        recent = recall_memories(config=self.config)
        self.assertEqual(
            [memory["content"] for memory in recent["memories"]],
            ["uses MySQL"],
        )

    def test_recall_memories_bumps_recall_count_only_on_returned_memories(self) -> None:
        save_memories(
            [
                MemoryInput(content="User prefers Python for backend work"),
                MemoryInput(content="User enjoys hiking on weekends"),
            ],
            config=self.config,
        )
        recall_memories("Python backend", top_k=1, config=self.config)
        db_path = Path(self.config["memory_db_path"])
        rows = {row.content: row for row in fetch_candidates(db_path)}
        self.assertEqual(rows["User prefers Python for backend work"].recall_count, 1)
        self.assertIsNotNone(
            rows["User prefers Python for backend work"].last_recalled_at
        )
        self.assertEqual(rows["User enjoys hiking on weekends"].recall_count, 0)
        self.assertIsNone(rows["User enjoys hiking on weekends"].last_recalled_at)

    def test_recent_recall_does_not_bump_recall_count(self) -> None:
        save_memories([MemoryInput(content="User likes tea")], config=self.config)
        recall_memories(config=self.config)
        db_path = Path(self.config["memory_db_path"])
        self.assertEqual(fetch_candidates(db_path)[0].recall_count, 0)

    def test_save_memories_rejects_invalid_kind(self) -> None:
        with self.assertRaises(InvalidMemoryKindError):
            save_memories(
                [MemoryInput(content="User likes tea", kind="bogus")],
                config=self.config,
            )

    def test_profile_memory_is_global_regardless_of_session_id(self) -> None:
        payload = save_memories(
            [MemoryInput(content="Call the user Marcell", kind="profile")],
            session_id="session-a",
            config=self.config,
        )
        self.assertIsNone(payload["saved"][0]["session_id"])
        self.assertEqual(payload["saved"][0]["kind"], "profile")

    def test_profile_memory_excluded_from_ranked_and_recent_lists(self) -> None:
        save_memories(
            [
                MemoryInput(content="Call the user Marcell", kind="profile"),
                MemoryInput(content="User prefers Python for backend work"),
            ],
            config=self.config,
        )
        ranked = recall_memories("Python backend", config=self.config)
        self.assertEqual(len(ranked["memories"]), 1)
        self.assertIn("Python", ranked["memories"][0]["content"])

        recent = recall_memories(config=self.config)
        self.assertEqual(
            [memory["content"] for memory in recent["memories"]],
            ["User prefers Python for backend work"],
        )

    def test_profile_block_attached_to_every_recall_response(self) -> None:
        save_memories(
            [MemoryInput(content="Call the user Marcell", kind="profile")],
            config=self.config,
        )
        save_memories(
            [MemoryInput(content="Project uses SQLite")],
            config=self.config,
        )
        ranked = recall_memories("SQLite", config=self.config)
        self.assertEqual(len(ranked["profile"]), 1)
        self.assertEqual(ranked["profile"][0]["content"], "Call the user Marcell")

        recent = recall_memories(config=self.config)
        self.assertEqual(len(recent["profile"]), 1)
        self.assertEqual(recent["profile"][0]["content"], "Call the user Marcell")

    def test_profile_block_empty_when_no_profile_memories(self) -> None:
        save_memories([MemoryInput(content="Project uses SQLite")], config=self.config)
        payload = recall_memories("SQLite", config=self.config)
        self.assertEqual(payload["profile"], [])

    def test_profile_block_respects_its_own_token_budget(self) -> None:
        save_memories(
            [
                MemoryInput(content="older profile fact", kind="profile"),
                MemoryInput(
                    content="newest profile fact with several words",
                    kind="profile",
                ),
            ],
            config=self.config,
        )
        payload = recall_memories(
            config=dict(self.config, profile_max_tokens=1),
        )
        self.assertEqual(len(payload["profile"]), 1)
        self.assertEqual(
            payload["profile"][0]["content"], "newest profile fact with several words"
        )

    def test_profile_dedup_scoped_separately_from_episodic(self) -> None:
        save_memories(
            [MemoryInput(content="User likes tea")],
            config=self.config,
        )
        payload = save_memories(
            [MemoryInput(content="User likes tea", kind="profile")],
            config=self.config,
        )
        self.assertEqual(len(payload["saved"]), 1)
        self.assertNotIn("skipped_duplicates", payload)

    def test_update_memory_preserves_profile_kind(self) -> None:
        saved = save_memories(
            [MemoryInput(content="Call the user Marc", kind="profile")],
            config=self.config,
        )
        old_id = saved["saved"][0]["id"]
        update_memory(old_id, "Call the user Marcell", config=self.config)
        payload = recall_memories(config=self.config)
        self.assertEqual(
            [memory["content"] for memory in payload["profile"]],
            ["Call the user Marcell"],
        )

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

    def test_list_memories_is_newest_first_and_not_semantic(self) -> None:
        save_memories(
            [MemoryInput(content="first saved"), MemoryInput(content="second saved")],
            config=self.config,
        )
        payload = list_memories(config=self.config)
        self.assertEqual(
            [memory["content"] for memory in payload["memories"]],
            ["second saved", "first saved"],
        )
        self.assertEqual(payload["total_count"], 2)
        self.assertEqual(payload["returned_count"], 2)
        self.assertFalse(payload["has_more"])

    def test_list_memories_paginates_with_limit_and_offset(self) -> None:
        save_memories(
            [MemoryInput(content=f"item {index}") for index in range(5)],
            config=self.config,
        )
        first_page = list_memories(limit=2, offset=0, config=self.config)
        second_page = list_memories(limit=2, offset=2, config=self.config)
        self.assertEqual(
            [memory["content"] for memory in first_page["memories"]],
            ["item 4", "item 3"],
        )
        self.assertTrue(first_page["has_more"])
        self.assertEqual(
            [memory["content"] for memory in second_page["memories"]],
            ["item 2", "item 1"],
        )
        self.assertTrue(second_page["has_more"])
        self.assertEqual(first_page["total_count"], 5)
        self.assertEqual([m["rank"] for m in second_page["memories"]], [3, 4])

    def test_list_memories_filters_by_date_range(self) -> None:
        db_path = Path(self.config["memory_db_path"])
        insert_memories(
            db_path,
            [
                MemoryRow(
                    id="aaaaaaaaaaaa0001",
                    session_id=None,
                    content="old memory",
                    created_at="2026-08-01T00:00:00Z",
                ),
                MemoryRow(
                    id="aaaaaaaaaaaa0002",
                    session_id=None,
                    content="recent memory",
                    created_at="2026-08-20T00:00:00Z",
                ),
            ],
        )
        payload = list_memories(
            since="2026-08-10T00:00:00Z", config=self.config
        )
        self.assertEqual(
            [memory["content"] for memory in payload["memories"]], ["recent memory"]
        )
        self.assertEqual(payload["total_count"], 1)

    def test_list_memories_truncates_long_content_preview(self) -> None:
        long_content = "x" * 500
        save_memories([MemoryInput(content=long_content)], config=self.config)
        payload = list_memories(config=self.config)
        memory = payload["memories"][0]
        self.assertTrue(memory["preview_truncated"])
        self.assertLess(len(memory["content"]), len(long_content))
        self.assertGreater(memory["content_tokens"], 0)

    def test_list_memories_kind_filter(self) -> None:
        save_memories(
            [MemoryInput(content="profile fact", kind="profile")], config=self.config
        )
        save_memories([MemoryInput(content="episodic fact")], config=self.config)
        payload = list_memories(kind="profile", config=self.config)
        self.assertEqual(
            [memory["content"] for memory in payload["memories"]], ["profile fact"]
        )

    def test_list_memories_rejects_invalid_kind(self) -> None:
        with self.assertRaises(InvalidMemoryKindError):
            list_memories(kind="bogus", config=self.config)

    def test_list_memories_validates_limit_and_offset(self) -> None:
        with self.assertRaises(RecallBudgetError):
            list_memories(limit=0, config=self.config)
        with self.assertRaises(RecallBudgetError):
            list_memories(offset=-1, config=self.config)

    def test_get_memory_returns_full_content_by_ref(self) -> None:
        saved = save_memories([MemoryInput(content="full detail")], config=self.config)
        ref = saved["saved"][0]["ref"]
        payload = get_memory(ref, config=self.config)
        self.assertEqual(payload["content"], "full detail")
        self.assertEqual(payload["id"], saved["saved"][0]["id"])
        self.assertEqual(payload["recall_count"], 0)

    def test_get_memory_missing_raises(self) -> None:
        with self.assertRaises(MemoryNotFoundError):
            get_memory("missing-id", config=self.config)
