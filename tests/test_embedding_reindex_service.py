from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext.models import MemoryRow
from tinycontext.services.embedding_reindex_service import (
    ensure_background_reindex,
    reindex_notice,
)
from tinycontext.services.memory_store_service import (
    close_connection,
    fetch_candidates,
    insert_memories,
)


class EmbeddingReindexServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "memories.db"
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id="m1",
                    session_id=None,
                    content="one",
                    created_at="2026-01-01T00:00:00Z",
                    embedding=[1.0, 0.0],
                    embedding_model="old-model",
                    embedding_dimensions=2,
                ),
                MemoryRow(
                    id="m2",
                    session_id=None,
                    content="two",
                    created_at="2026-01-01T00:00:01Z",
                    embedding=[0.0, 1.0],
                    embedding_model="old-model",
                    embedding_dimensions=2,
                ),
            ],
        )

    def tearDown(self) -> None:
        close_connection(self.db_path)
        self._tmpdir.cleanup()

    def test_background_reindex_updates_stale_rows_and_clears_notice(self) -> None:
        release = threading.Event()
        started = threading.Event()

        def slow_embed(inputs, **_kwargs):
            started.set()
            release.wait(timeout=5)
            return [[0.5, 0.5] for _ in inputs]

        with patch(
            "tinycontext.services.embedding_reindex_service.embed_texts",
            side_effect=slow_embed,
        ):
            ensure_background_reindex(
                self.db_path,
                embedding_model="fast",
                models_dir=Path("unused"),
                embedding_batch_size=16,
                document_prefix="",
            )
            self.assertTrue(started.wait(timeout=5))
            notice = reindex_notice(self.db_path)
            self.assertIsNotNone(notice)
            assert notice is not None
            self.assertIn("in progress", notice)
            self.assertIn("2", notice)

            release.set()
            for _ in range(50):
                if reindex_notice(self.db_path) is None:
                    break
                time.sleep(0.05)
            self.assertIsNone(reindex_notice(self.db_path))

        rows = {row.id: row for row in fetch_candidates(self.db_path)}
        self.assertEqual(rows["m1"].embedding_model, rows["m2"].embedding_model)
        self.assertNotEqual(rows["m1"].embedding_model, "old-model")

    def test_ensure_background_reindex_is_a_noop_when_already_current(self) -> None:
        with patch(
            "tinycontext.services.embedding_reindex_service.embed_texts",
            side_effect=lambda inputs, **_kwargs: [[0.1, 0.2] for _ in inputs],
        ) as embed:
            ensure_background_reindex(
                self.db_path,
                embedding_model="fast",
                models_dir=Path("unused"),
                embedding_batch_size=16,
                document_prefix="",
            )
            for _ in range(50):
                if reindex_notice(self.db_path) is None:
                    break
                time.sleep(0.05)
            embed.assert_called()

        with patch(
            "tinycontext.services.embedding_reindex_service.embed_texts",
        ) as embed_again:
            ensure_background_reindex(
                self.db_path,
                embedding_model="fast",
                models_dir=Path("unused"),
                embedding_batch_size=16,
                document_prefix="",
            )
            time.sleep(0.1)
            embed_again.assert_not_called()


if __name__ == "__main__":
    unittest.main()
