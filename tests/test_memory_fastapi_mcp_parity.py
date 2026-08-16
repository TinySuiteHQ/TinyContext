from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext import MemoryInput, save_memories
from tinycontext.servers.fastapi_server import (
    MemoryInputModel,
    RecallMemoriesRequest,
    RecallRecentMemoriesRequest,
    SaveMemoriesRequest,
    recall_memories_endpoint,
    recall_recent_memories_endpoint,
    save_memories_endpoint,
)
from tinycontext.servers.mcp_server import (
    recall_memories_tool,
    recall_recent_memories_tool,
    save_memories_tool,
)
from tinycontext.services.memory_store_service import close_connection
from tests.embedding_fakes import start_fake_embeddings


def _fn(coro):
    return getattr(coro, "fn", coro)


class MemoryFastApiMcpParityTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_save_memories_parity(self) -> None:
        # Distinct content per adapter so save-time dedup doesn't skip the
        # second call -- this test checks response-shape parity, not dedup.
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            fastapi_payload = await save_memories_endpoint(
                SaveMemoriesRequest(
                    memories=[MemoryInputModel(content="parity save memory fastapi")],
                )
            )
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            mcp_payload = await _fn(save_memories_tool)(
                [{"content": "parity save memory mcp"}]
            )
        self.assertEqual(
            {item["session_id"] for item in fastapi_payload["saved"]},
            {item["session_id"] for item in mcp_payload["saved"]},
        )
        self.assertEqual(len(fastapi_payload["saved"]), len(mcp_payload["saved"]))

    async def test_recall_memories_adapters_share_recalled_content(self) -> None:
        save_memories(
            [MemoryInput(content="memory about sqlite storage")],
            config=self.config,
        )
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            fastapi_payload = await recall_memories_endpoint(
                RecallMemoriesRequest(query="sqlite")
            )
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            mcp_payload = await _fn(recall_memories_tool)("sqlite")
        self.assertEqual(len(fastapi_payload["memories"]), 1)
        self.assertIn(fastapi_payload["memories"][0]["content"], mcp_payload)
        self.assertIn(
            f'relevance="{fastapi_payload["memories"][0]["relevance"]}"',
            mcp_payload,
        )
        self.assertIn(
            f'created_at="{fastapi_payload["memories"][0]["created_at"]}"',
            mcp_payload,
        )
        self.assertIn(
            f'current_time="{fastapi_payload["current_time"]}"',
            mcp_payload,
        )

    async def test_recent_recall_adapters_share_selected_content_and_order(self) -> None:
        save_memories(
            [MemoryInput(content="first recent"), MemoryInput(content="second recent")],
            config=self.config,
        )
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            fastapi_payload = await recall_recent_memories_endpoint(
                RecallRecentMemoriesRequest(top_k=2)
            )
        with patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            mcp_payload = await _fn(recall_recent_memories_tool)(top_k=2)
        self.assertEqual(
            [memory["content"] for memory in fastapi_payload["memories"]],
            ["second recent", "first recent"],
        )
        self.assertLess(
            mcp_payload.index("second recent"),
            mcp_payload.index("first recent"),
        )
        self.assertIn('<recalled_memories mode="recent"', mcp_payload)
