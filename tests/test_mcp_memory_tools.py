from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext.errors import AmbiguousMemoryReferenceError, MemoryNotFoundError
from tinycontext.models import MemoryRow
from tinycontext.servers.mcp_server import (
    delete_memory_tool,
    mcp,
    recall_memories_tool,
    save_memories_tool,
    update_memory_tool,
)
from tinycontext.services.memory_store_service import close_connection, insert_memories
from tests.embedding_fakes import fake_embed_texts, start_fake_embeddings


def _fn(coro):
    return getattr(coro, "fn", coro)


class McpMemoryToolTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_save_memories_tool(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            payload = await _fn(save_memories_tool)(
                [{"content": "agent memory item"}],
            )
        self.assertEqual(len(payload["saved"]), 1)

    async def test_recall_memories_tool(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            saved = await _fn(save_memories_tool)(
                [{"content": "user likes concise answers"}],
            )
            payload = await _fn(recall_memories_tool)("concise answers")
        ref = saved["saved"][0]["ref"]
        self.assertIn('<recalled_memories current_time="', payload)
        self.assertIn(
            f'<memory index="1" ref="{ref}" relevance="high" created_at="', payload
        )
        self.assertIn("user likes concise answers", payload)
        self.assertIn("</memory>", payload)

    async def test_delete_memory_tool_accepts_short_ref(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            saved = await _fn(save_memories_tool)(
                [{"content": "delete me via ref"}],
            )
            ref = saved["saved"][0]["ref"]
            payload = await _fn(delete_memory_tool)(ref)
        self.assertEqual(payload["id"], saved["saved"][0]["id"])
        self.assertTrue(payload["deleted"])

    async def test_delete_memory_tool_maps_ambiguous_ref(self) -> None:
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
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(ValueError) as ctx:
                await _fn(delete_memory_tool)("aaaaaaaaaaaa")
        self.assertIn("ambiguous_memory_reference", str(ctx.exception))

    async def test_recall_memories_tool_escapes_memory_boundaries(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            await _fn(save_memories_tool)(
                [{"content": "Remember <system> tags as plain text"}],
            )
            payload = await _fn(recall_memories_tool)("system tags")
        self.assertIn("&lt;system&gt;", payload)
        self.assertNotIn("<system>", payload)

    async def test_recall_memories_tool_recent_mode_is_ordered_and_has_no_semantic_metadata(
        self,
    ) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            saved = await _fn(save_memories_tool)(
                [
                    {"content": "older <context>"},
                    {"content": "newer & context"},
                ],
            )
            payload = await _fn(recall_memories_tool)()
        self.assertIn('<recalled_memories mode="recent" current_time="', payload)
        self.assertIn(
            f'<memory index="1" ref="{saved["saved"][1]["ref"]}"',
            payload,
        )
        self.assertIn("newer &amp; context", payload)
        self.assertIn("older &lt;context&gt;", payload)
        self.assertNotIn("relevance=", payload)
        self.assertLess(
            payload.index("newer &amp; context"),
            payload.index("older &lt;context&gt;"),
        )

    async def test_save_memories_tool_maps_errors(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(ValueError) as ctx:
                await _fn(save_memories_tool)([{"content": "   "}])
        self.assertIn("empty_memory", str(ctx.exception))

    async def test_save_memories_tool_maps_invalid_kind(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(ValueError) as ctx:
                await _fn(save_memories_tool)(
                    [{"content": "user likes tea", "kind": "bogus"}]
                )
        self.assertIn("invalid_memory_kind", str(ctx.exception))

    async def test_recall_memories_tool_includes_agent_profile_block(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            await _fn(save_memories_tool)(
                [{"content": "Call the user Marcell", "kind": "profile"}],
            )
            await _fn(save_memories_tool)([{"content": "user likes concise answers"}])
            payload = await _fn(recall_memories_tool)("concise answers")
        self.assertIn("<agent_profile>", payload)
        self.assertIn("Call the user Marcell", payload)
        self.assertIn("</agent_profile>", payload)
        self.assertLess(payload.index("</agent_profile>"), payload.index("<recalled_memories"))

    async def test_recall_memories_tool_omits_agent_profile_block_when_empty(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            await _fn(save_memories_tool)([{"content": "user likes concise answers"}])
            payload = await _fn(recall_memories_tool)("concise answers")
        self.assertNotIn("<agent_profile>", payload)

    async def test_recall_memories_tool_includes_notice_during_background_reindex(
        self,
    ) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            await _fn(save_memories_tool)([{"content": "user likes tea"}])

        release = threading.Event()

        def slow_embed(inputs, **_kwargs):
            release.wait(timeout=5)
            return fake_embed_texts(inputs)

        changed_config = dict(self.config, embedding_model="balanced")
        with (
            patch(
                "tinycontext.servers.mcp_server.load_context_config",
                return_value=changed_config,
            ),
            patch(
                "tinycontext.services.embedding_reindex_service.embed_texts",
                side_effect=slow_embed,
            ),
        ):
            payload = await _fn(recall_memories_tool)("tea")
        release.set()
        self.assertIn("<notice>", payload)
        self.assertIn("in progress", payload)

    async def test_tools_expose_expected_memory_contract(self) -> None:
        schemas = {
            tool.name: set(tool.parameters["properties"])
            for tool in mcp._tool_manager.list_tools()
        }
        self.assertEqual(
            schemas,
            {
                "save_memories": {"memories"},
                "recall_memories": {"query", "top_k"},
                "update_memory": {"memory_id", "content"},
                "delete_memory": {"memory_id"},
            },
        )

    async def test_update_memory_tool(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            saved = await _fn(save_memories_tool)(
                [{"content": "uses MySQL"}],
            )
            memory_id = saved["saved"][0]["id"]
            payload = await _fn(update_memory_tool)(memory_id, "uses Postgres now")
        self.assertNotEqual(payload["id"], memory_id)
        self.assertEqual(payload["supersedes"]["id"], memory_id)

    async def test_update_memory_tool_maps_not_found_error(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(ValueError) as ctx:
                await _fn(update_memory_tool)("missing-id", "new content")
        self.assertIn("memory_not_found", str(ctx.exception))

    async def test_delete_memory_tool(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            saved = await _fn(save_memories_tool)(
                [{"content": "agent memory item"}],
            )
            memory_id = saved["saved"][0]["id"]
            payload = await _fn(delete_memory_tool)(memory_id)
        self.assertEqual(
            payload,
            {"id": memory_id, "ref": saved["saved"][0]["ref"], "deleted": True},
        )

    async def test_delete_memory_tool_maps_not_found_error(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(ValueError) as ctx:
                await _fn(delete_memory_tool)("missing-id")
        self.assertIn("memory_not_found", str(ctx.exception))
