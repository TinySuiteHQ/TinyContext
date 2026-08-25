from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tinycontext.models import MemoryRow
from tinycontext.services import memory_store_service
from tinycontext.services.memory_store_service import (
    clear_superseded_by,
    close_connection,
    count_memories,
    delete_memory,
    embedding_storage_stats,
    fetch_candidates,
    fetch_memory_by_id,
    fetch_recent_memories,
    fetch_dense_scores,
    init_db,
    insert_memories,
    record_recall_hits,
    session_exists,
    supersede_memory,
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
                created_at="2026-06-30T10:00:00Z",
            ),
            MemoryRow(
                id="m2",
                session_id="s1",
                content="Project uses FastAPI",
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
                        created_at="2026-06-30T10:00:00Z",
                ),
                MemoryRow(
                    id="m2",
                    session_id="s2",
                    content="two",
                        created_at="2026-06-30T10:01:00Z",
                ),
            ],
        )
        rows = fetch_candidates(self.db_path, session_id="s1")
        self.assertEqual([row.id for row in rows], ["m1"])

    def test_fetch_filters_by_kind(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="m1",
                    session_id="s1",
                    content="episodic fact",
                    created_at="2026-06-30T10:00:00Z",
                    kind="episodic",
                ),
                MemoryRow(
                    id="m2",
                    session_id=None,
                    content="call the user Marcell",
                    created_at="2026-06-30T10:01:00Z",
                    kind="profile",
                ),
            ],
        )
        self.assertEqual(
            [row.id for row in fetch_candidates(self.db_path, kind="episodic")],
            ["m1"],
        )
        self.assertEqual(
            [row.id for row in fetch_candidates(self.db_path, kind="profile")],
            ["m2"],
        )
        self.assertEqual(
            {row.id for row in fetch_candidates(self.db_path)}, {"m1", "m2"}
        )
        self.assertEqual(
            [row.id for row in fetch_recent_memories(self.db_path, kind="profile")],
            ["m2"],
        )
        row = fetch_memory_by_id(self.db_path, "m2")
        assert row is not None
        self.assertEqual(row.kind, "profile")

    def test_fetch_recent_orders_ties_by_insertion_and_applies_limit(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="first",
                    session_id="s1",
                    content="first",
                    created_at="2026-06-30T10:00:00Z",
                ),
                MemoryRow(
                    id="second",
                    session_id="s1",
                    content="second",
                    created_at="2026-06-30T10:00:00Z",
                ),
                MemoryRow(
                    id="newest",
                    session_id="s2",
                    content="newest",
                    created_at="2026-06-30T10:01:00Z",
                ),
            ],
        )
        rows = fetch_recent_memories(self.db_path, session_id="s1", limit=2)
        self.assertEqual([row.id for row in rows], ["second", "first"])

    def test_fetch_recent_supports_offset_for_pagination(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id=str(index),
                    session_id="s1",
                    content=str(index),
                    created_at=f"2026-06-30T10:0{index}:00Z",
                )
                for index in range(4)
            ],
        )
        first_page = fetch_recent_memories(self.db_path, limit=2, offset=0)
        second_page = fetch_recent_memories(self.db_path, limit=2, offset=2)
        self.assertEqual([row.id for row in first_page], ["3", "2"])
        self.assertEqual([row.id for row in second_page], ["1", "0"])

    def test_fetch_recent_filters_by_since_and_until(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="old",
                    session_id=None,
                    content="old",
                    created_at="2026-08-01T00:00:00Z",
                ),
                MemoryRow(
                    id="mid",
                    session_id=None,
                    content="mid",
                    created_at="2026-08-10T00:00:00Z",
                ),
                MemoryRow(
                    id="new",
                    session_id=None,
                    content="new",
                    created_at="2026-08-20T00:00:00Z",
                ),
            ],
        )
        self.assertEqual(
            [row.id for row in fetch_recent_memories(self.db_path, since="2026-08-05T00:00:00Z")],
            ["new", "mid"],
        )
        self.assertEqual(
            [row.id for row in fetch_recent_memories(self.db_path, until="2026-08-05T00:00:00Z")],
            ["old"],
        )
        self.assertEqual(
            [
                row.id
                for row in fetch_recent_memories(
                    self.db_path,
                    since="2026-08-05T00:00:00Z",
                    until="2026-08-15T00:00:00Z",
                )
            ],
            ["mid"],
        )

    def test_count_memories_matches_filters(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="m1",
                    session_id="s1",
                    content="one",
                    created_at="2026-06-30T10:00:00Z",
                    kind="episodic",
                ),
                MemoryRow(
                    id="m2",
                    session_id=None,
                    content="two",
                    created_at="2026-06-30T10:01:00Z",
                    kind="profile",
                ),
            ],
        )
        self.assertEqual(count_memories(self.db_path), 2)
        self.assertEqual(count_memories(self.db_path, kind="episodic"), 1)
        self.assertEqual(count_memories(self.db_path, session_id="s1"), 1)
        self.assertEqual(count_memories(self.db_path, since="2026-07-01T00:00:00Z"), 0)

    def test_session_exists(self) -> None:
        self.assertFalse(session_exists(self.db_path, "missing"))
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="m1",
                    session_id="s1",
                    content="one",
                        created_at="2026-06-30T10:00:00Z",
                )
            ],
        )
        self.assertTrue(session_exists(self.db_path, "s1"))

    def test_delete_memory_removes_row_and_reports_existence(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="m1",
                    session_id="s1",
                    content="one",
                    created_at="2026-06-30T10:00:00Z",
                )
            ],
        )
        self.assertTrue(delete_memory(self.db_path, "m1"))
        self.assertEqual(fetch_candidates(self.db_path), [])
        self.assertFalse(delete_memory(self.db_path, "m1"))

    def test_fetch_memory_by_id_returns_row_or_none(self) -> None:
        self.assertIsNone(fetch_memory_by_id(self.db_path, "missing"))
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="m1",
                    session_id="s1",
                    content="one",
                    created_at="2026-06-30T10:00:00Z",
                )
            ],
        )
        row = fetch_memory_by_id(self.db_path, "m1")
        assert row is not None
        self.assertEqual(row.content, "one")
        self.assertEqual(row.recall_count, 0)
        self.assertIsNone(row.superseded_by)

    def test_supersede_memory_hides_row_from_fetches(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="old",
                    session_id=None,
                    content="uses MySQL",
                    created_at="2026-06-30T10:00:00Z",
                ),
                MemoryRow(
                    id="new",
                    session_id=None,
                    content="uses Postgres",
                    created_at="2026-06-30T10:01:00Z",
                ),
            ],
        )
        supersede_memory(self.db_path, "old", "new", "2026-06-30T10:01:00Z")
        self.assertEqual([row.id for row in fetch_candidates(self.db_path)], ["new"])
        self.assertEqual(
            [row.id for row in fetch_recent_memories(self.db_path)], ["new"]
        )
        old_row = fetch_memory_by_id(self.db_path, "old")
        assert old_row is not None
        self.assertEqual(old_row.superseded_by, "new")
        self.assertEqual(old_row.superseded_at, "2026-06-30T10:01:00Z")

    def test_clear_superseded_by_restores_visibility(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="old",
                    session_id=None,
                    content="uses MySQL",
                    created_at="2026-06-30T10:00:00Z",
                ),
                MemoryRow(
                    id="new",
                    session_id=None,
                    content="uses Postgres",
                    created_at="2026-06-30T10:01:00Z",
                ),
            ],
        )
        supersede_memory(self.db_path, "old", "new", "2026-06-30T10:01:00Z")
        clear_superseded_by(self.db_path, "new")
        old_row = fetch_memory_by_id(self.db_path, "old")
        assert old_row is not None
        self.assertIsNone(old_row.superseded_by)
        self.assertEqual(
            {row.id for row in fetch_candidates(self.db_path)}, {"old", "new"}
        )

    def test_record_recall_hits_increments_count_and_timestamp(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="m1",
                    session_id=None,
                    content="one",
                    created_at="2026-06-30T10:00:00Z",
                )
            ],
        )
        record_recall_hits(self.db_path, ["m1"], "2026-06-30T11:00:00Z")
        record_recall_hits(self.db_path, ["m1"], "2026-06-30T12:00:00Z")
        row = fetch_memory_by_id(self.db_path, "m1")
        assert row is not None
        self.assertEqual(row.recall_count, 2)
        self.assertEqual(row.last_recalled_at, "2026-06-30T12:00:00Z")

    def test_vectors_are_stored_as_blobs_and_ranked_inside_sqlite(self) -> None:
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="near",
                    session_id=None,
                    content="near",
                        created_at="2026-06-30T10:00:00Z",
                    embedding=[1.0, 0.0],
                    embedding_model="test-model",
                    embedding_dimensions=2,
                ),
                MemoryRow(
                    id="far",
                    session_id=None,
                    content="far",
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
        profile_scores = fetch_dense_scores(
            self.db_path,
            [1.0, 0.0],
            embedding_model="test-model",
            kind="profile",
        )
        self.assertEqual(profile_scores, {})
        close_connection(self.db_path)
        connection = sqlite3.connect(self.db_path)
        try:
            storage_type = connection.execute(
                "SELECT typeof(embedding) FROM memories WHERE id = 'near'"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(storage_type, "blob")

    def test_cosine_ranking_falls_back_when_extensions_are_unavailable(self) -> None:
        connection = sqlite3.connect(":memory:")

        class ExtensionlessConnection:
            def create_function(self, *args: object, **kwargs: object) -> None:
                connection.create_function(*args, **kwargs)

        try:
            loaded = memory_store_service._load_sqlite_vec(ExtensionlessConnection())
            near = memory_store_service.sqlite_vec.serialize_float32([1.0, 0.0])
            far = memory_store_service.sqlite_vec.serialize_float32([0.0, 1.0])
            distance = connection.execute(
                "SELECT vec_distance_cosine(?, ?)",
                (near, far),
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertFalse(loaded)
        self.assertAlmostEqual(distance, 1.0)

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
