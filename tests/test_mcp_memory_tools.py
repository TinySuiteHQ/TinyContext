from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext.servers.mcp_server import (
    delete_memory_tool,
    mcp,
    recall_memories_tool,
    save_memories_tool,
)
from tinycontext.services.memory_store_service import close_connection
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
            await _fn(save_memories_tool)(
                [{"content": "user likes concise answers"}],
            )
            payload = await _fn(recall_memories_tool)("concise answers")
        self.assertIn('<recalled_memories current_time="', payload)
        self.assertIn('<memory index="1" relevance="high" created_at="', payload)
        self.assertIn("user likes concise answers", payload)
        self.assertIn("</memory>", payload)

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

    async def test_save_memories_tool_maps_errors(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(ValueError) as ctx:
                await _fn(save_memories_tool)([{"content": "   "}])
        self.assertIn("empty_memory", str(ctx.exception))

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

    async def test_tools_expose_only_memories_and_query(self) -> None:
        schemas = {
            tool.name: set(tool.parameters["properties"])
            for tool in mcp._tool_manager.list_tools()
        }
        self.assertEqual(
            schemas,
            {
                "save_memories": {"memories"},
                "recall_memories": {"query"},
                "delete_memory": {"memory_id"},
            },
        )

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
        self.assertEqual(payload, {"id": memory_id, "deleted": True})

    async def test_delete_memory_tool_maps_not_found_error(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(ValueError) as ctx:
                await _fn(delete_memory_tool)("missing-id")
        self.assertIn("memory_not_found", str(ctx.exception))
