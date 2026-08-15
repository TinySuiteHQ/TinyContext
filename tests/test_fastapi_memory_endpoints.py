from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from tinycontext.servers.fastapi_server import (
    DeleteMemoryRequest,
    MemoryInputModel,
    RecallMemoriesRequest,
    SaveMemoriesRequest,
    delete_memory_endpoint,
    recall_memories_endpoint,
    save_memories_endpoint,
)
from tinycontext.services.memory_store_service import close_connection
from tests.embedding_fakes import start_fake_embeddings


class FastApiMemoryEndpointTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_save_memories_endpoint(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            payload = await save_memories_endpoint(
                SaveMemoriesRequest(
                    session_id="s1",
                    memories=[MemoryInputModel(content="remember this fact")],
                )
            )
        self.assertEqual(len(payload["saved"]), 1)

    async def test_recall_memories_endpoint(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            await save_memories_endpoint(
                SaveMemoriesRequest(
                    memories=[MemoryInputModel(content="project uses sqlite")],
                )
            )
            payload = await recall_memories_endpoint(
                RecallMemoriesRequest(query="sqlite")
            )
        self.assertGreaterEqual(len(payload["memories"]), 1)

    async def test_delete_memory_endpoint(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            saved = await save_memories_endpoint(
                SaveMemoriesRequest(
                    memories=[MemoryInputModel(content="forget this")],
                )
            )
            memory_id = saved["saved"][0]["id"]
            payload = await delete_memory_endpoint(
                DeleteMemoryRequest(memory_id=memory_id)
            )
        self.assertEqual(
            payload,
            {"id": memory_id, "ref": saved["saved"][0]["ref"], "deleted": True},
        )

    async def test_delete_memory_endpoint_accepts_short_ref(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            saved = await save_memories_endpoint(
                SaveMemoriesRequest(
                    memories=[MemoryInputModel(content="forget this too")],
                )
            )
            ref = saved["saved"][0]["ref"]
            payload = await delete_memory_endpoint(DeleteMemoryRequest(memory_id=ref))
        self.assertEqual(payload["id"], saved["saved"][0]["id"])
        self.assertTrue(payload["deleted"])

    async def test_delete_memory_maps_not_found_error(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await delete_memory_endpoint(
                    DeleteMemoryRequest(memory_id="missing-id")
                )
        self.assertEqual(ctx.exception.status_code, 404)
        detail = ctx.exception.detail
        assert isinstance(detail, dict)
        self.assertEqual(detail["code"], "memory_not_found")

    async def test_save_memories_maps_empty_memory_error(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await save_memories_endpoint(
                    SaveMemoriesRequest(
                        memories=[MemoryInputModel(content="   ")],
                    )
                )
        self.assertEqual(ctx.exception.status_code, 400)
        detail = ctx.exception.detail
        assert isinstance(detail, dict)
        self.assertEqual(detail["code"], "empty_memory")
