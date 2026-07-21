from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from servers.fastapi_server import (
    MemoryInputModel,
    RecallMemoriesRequest,
    SaveMemoriesRequest,
    recall_memories_endpoint,
    save_memories_endpoint,
)
from servers.mcp_server import recall_memories_tool, save_memories_tool
from services.memory_service import MemoryInput, save_memories
from services.memory_store_service import close_connection


def _fn(coro):
    return getattr(coro, "fn", coro)


class MemoryFastApiMcpParityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
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
        memories = [{"content": "parity save memory", "tags": ["test"]}]
        with patch(
            "servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            fastapi_payload = await save_memories_endpoint(
                SaveMemoriesRequest(
                    session_id="parity",
                    memories=[MemoryInputModel(**memories[0])],
                )
            )
        with patch(
            "servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            mcp_payload = await _fn(save_memories_tool)(memories, session_id="parity")
        self.assertEqual(
            {item["session_id"] for item in fastapi_payload["saved"]},
            {item["session_id"] for item in mcp_payload["saved"]},
        )
        self.assertEqual(len(fastapi_payload["saved"]), len(mcp_payload["saved"]))

    async def test_recall_memories_parity(self) -> None:
        save_memories(
            [MemoryInput(content="memory about sqlite storage")],
            session_id="parity",
            config=self.config,
        )
        with patch(
            "servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            fastapi_payload = await recall_memories_endpoint(
                RecallMemoriesRequest(query="sqlite", session_id="parity")
            )
        with patch(
            "servers.mcp_server.load_context_config",
            return_value=self.config,
        ):
            mcp_payload = await _fn(recall_memories_tool)(
                "sqlite",
                session_id="parity",
            )
        self.assertEqual(fastapi_payload["query"], mcp_payload["query"])
        self.assertEqual(
            fastapi_payload["total_tokens"],
            mcp_payload["total_tokens"],
        )
        self.assertEqual(
            fastapi_payload["truncated"],
            mcp_payload["truncated"],
        )
        self.assertEqual(
            len(fastapi_payload["memories"]),
            len(mcp_payload["memories"]),
        )
