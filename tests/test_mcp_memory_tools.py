from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext.servers.mcp_server import (
    mcp,
    recall_memories_tool,
    save_memories_tool,
)
from tinycontext.services.memory_store_service import close_connection
from tests.embedding_fakes import start_fake_embeddings


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
                [{"content": "agent memory item", "tags": ["note"]}],
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
        self.assertGreaterEqual(len(payload["memories"]), 1)

    async def test_save_memories_tool_maps_errors(self) -> None:
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(ValueError) as ctx:
                await _fn(save_memories_tool)([{"content": "   "}])
        self.assertIn("empty_memory", str(ctx.exception))

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
            },
        )
