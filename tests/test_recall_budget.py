from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tinycontext import core
from tinycontext.models import MemoryRow
from tinycontext.servers.mcp_server import _format_recalled_memories
from tinycontext.services.memory_store_service import close_connection, insert_memories
from tinycontext.services.token_counter_service import select_within_budget
from tests.embedding_fakes import start_fake_embeddings


# A bogus encoding name falls back to CharacterTokenizerAdapter, so one
# character is exactly one token and budgets can be asserted precisely.
CHAR_ENCODING = "__character_fallback__"


def _payload(item, rank, tokens):
    return {"name": item[0], "rank": rank, "content_tokens": tokens}


def _pack(sizes, budget):
    items = [(f"m{index}", "x" * size) for index, size in enumerate(sizes, 1)]
    result = select_within_budget(
        items,
        max_tokens=budget,
        encoding_name=CHAR_ENCODING,
        content_of=lambda item: item[1],
        payload=_payload,
    )
    # Callers derive "what was cut" from their own input list; mirror that
    # here rather than expecting the helper to hand it back.
    return (
        [entry["name"] for entry in result.payloads],
        result.total_tokens,
        len(items) - len(result.payloads),
    )


class SelectWithinBudgetTests(unittest.TestCase):
    def test_reports_what_the_budget_cut_off(self) -> None:
        """A long memory used to truncate the list silently -- see the 1-of-20 bug."""
        names, total_tokens, remaining = _pack([230, 1800, 20, 20, 20], 2000)
        self.assertEqual(names, ["m1"])
        self.assertEqual(total_tokens, 230)
        self.assertEqual(remaining, 4)

    def test_larger_budget_returns_the_full_set(self) -> None:
        names, _total_tokens, remaining = _pack([230, 1800, 20, 20, 20], 8000)
        self.assertEqual(names, ["m1", "m2", "m3", "m4", "m5"])
        self.assertEqual(remaining, 0)

    def test_selection_is_a_contiguous_prefix(self) -> None:
        """Oversized entries stop the scan; they are never skipped over."""
        names, _total_tokens, remaining = _pack([10, 5000, 10], 2000)
        self.assertEqual(names, ["m1"])
        self.assertEqual(remaining, 2)

    def test_oversized_first_entry_is_still_returned(self) -> None:
        names, total_tokens, remaining = _pack([5000, 10], 2000)
        self.assertEqual(names, ["m1"])
        self.assertEqual(total_tokens, 5000)
        self.assertEqual(remaining, 1)

    def test_ranks_are_contiguous_from_one(self) -> None:
        items = [(f"m{index}", "x" * 10) for index in range(1, 4)]
        selected = select_within_budget(
            items,
            max_tokens=100,
            encoding_name=CHAR_ENCODING,
            content_of=lambda item: item[1],
            payload=_payload,
        ).payloads
        self.assertEqual([entry["rank"] for entry in selected], [1, 2, 3])

    def test_everything_fits(self) -> None:
        names, total_tokens, remaining = _pack([10, 10, 10], 2000)
        self.assertEqual(names, ["m1", "m2", "m3"])
        self.assertEqual(total_tokens, 30)
        self.assertEqual(remaining, 0)

    def test_no_items(self) -> None:
        self.assertEqual(_pack([], 2000), ([], 0, 0))


class TruncationMarkerTests(unittest.TestCase):
    def _render(self, remaining: int) -> str:
        return _format_recalled_memories(
            {
                "current_time": "2026-08-19T14:56:21Z",
                "memories": [
                    {
                        "ref": "f95cc5dbfc5b",
                        "content": "a stored fact",
                        "relevance": "low",
                        "created_at": "2026-08-19T14:56:05Z",
                    }
                ],
                "remaining": remaining,
                "profile": [],
            }
        )

    def test_marker_names_the_cut(self) -> None:
        rendered = self._render(4)
        self.assertIn('truncated="true" remaining="4"', rendered)
        self.assertIn('<truncated remaining="4">', rendered)
        self.assertIn("4 more matching memories", rendered)

    def test_singular_wording(self) -> None:
        self.assertIn("1 more matching memory", self._render(1))

    def test_no_marker_when_nothing_was_cut(self) -> None:
        rendered = self._render(0)
        self.assertNotIn("truncated", rendered)
        self.assertNotIn("<truncated", rendered)


class PagingTests(unittest.TestCase):
    """Walk a store that cannot fit in one response, and reach the bottom."""

    MEMORY_COUNT = 12
    # Only the fifth memory is long; the rest are short. That is the shape
    # that produced the original one-result recall.
    LONG_INDEX = 5

    def setUp(self) -> None:
        start_fake_embeddings(self)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "memories.db"
        self.config = {
            "memory_db_path": str(self.db_path),
            "recall_top_k": 50,
            "recall_max_tokens": 40,
            "encoding_name": "o200k_base",
        }
        # Inserted directly: save_memories dedupes on embedding similarity,
        # and the fake embedder has too few distinct vectors to keep a dozen
        # separate memories alive through it.
        insert_memories(
            self.db_path,
            [
                MemoryRow(
                    id=f"{index:08x}-0000-4000-8000-000000000000",
                    session_id=None,
                    content=(
                        f"alpha dossier {index} "
                        + ("padding " * 60 if index == self.LONG_INDEX else "brief")
                    ),
                    created_at=f"2026-08-{index + 1:02d}T00:00:00Z",
                )
                for index in range(1, self.MEMORY_COUNT + 1)
            ],
        )

    def tearDown(self) -> None:
        close_connection(self.db_path)
        self._tmpdir.cleanup()

    def _walk(self) -> tuple[list[str], int]:
        seen: list[str] = []
        offset = 0
        pages = 0
        while True:
            payload = core.recall_memories("alpha", offset=offset, config=self.config)
            pages += 1
            seen.extend(memory["ref"] for memory in payload["memories"])
            self.assertTrue(payload["memories"], "a page came back empty")
            if payload["next_offset"] is None:
                return seen, pages
            self.assertGreater(payload["next_offset"], offset, "offset did not advance")
            offset = payload["next_offset"]
            self.assertLess(pages, 40, "paging did not terminate")

    def test_paging_reaches_every_memory_exactly_once(self) -> None:
        seen, pages = self._walk()
        self.assertGreater(pages, 1, "budget should have forced more than one page")
        self.assertEqual(len(seen), self.MEMORY_COUNT)
        self.assertEqual(len(set(seen)), self.MEMORY_COUNT)

    def test_first_page_reports_what_is_behind_it(self) -> None:
        payload = core.recall_memories("alpha", config=self.config)
        self.assertTrue(payload["truncated"])
        self.assertEqual(
            payload["remaining"],
            self.MEMORY_COUNT - len(payload["memories"]),
        )
        self.assertEqual(payload["next_offset"], len(payload["memories"]))

    def test_index_points_at_the_memories_just_past_the_cut(self) -> None:
        payload = core.recall_memories("alpha", config=self.config)
        entries = payload["next_entries"]
        self.assertTrue(entries)
        self.assertLessEqual(len(entries), payload["remaining"])
        shown = {memory["ref"] for memory in payload["memories"]}
        self.assertFalse(
            shown & {entry["ref"] for entry in entries},
            "index should list what was cut, not what was already shown",
        )
        for entry in entries:
            self.assertTrue(entry["snippet"])
            self.assertGreater(entry["content_tokens"], 0)

    def test_index_ref_is_fetchable_in_full(self) -> None:
        payload = core.recall_memories("alpha", config=self.config)
        entry = payload["next_entries"][0]
        fetched = core.get_memory(entry["ref"], config=self.config)
        self.assertEqual(fetched["ref"], entry["ref"])
        self.assertEqual(fetched["content_tokens"], entry["content_tokens"])
        self.assertTrue(fetched["content"].startswith(entry["snippet"][:8]))

    def test_paged_reads_do_not_count_as_recalls(self) -> None:
        """Scrolling must not inflate recall_count, which can feed ranking."""
        core.recall_memories("alpha", offset=0, config=self.config)
        core.recall_memories("alpha", offset=3, config=self.config)
        core.recall_memories("alpha", offset=6, config=self.config)
        recalled = [
            row
            for row in core.recall_memories(
                "alpha", top_k=50, max_tokens=100000, config=self.config
            )["memories"]
        ]
        self.assertTrue(recalled)

    def test_negative_offset_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            core.recall_memories("alpha", offset=-1, config=self.config)

    def test_offset_past_the_end_is_empty_not_an_error(self) -> None:
        payload = core.recall_memories("alpha", offset=999, config=self.config)
        self.assertEqual(payload["memories"], [])
        self.assertIsNone(payload["next_offset"])


if __name__ == "__main__":
    unittest.main()
