from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from tinycontext.servers.fastapi_server import (
    DeleteMemoryRequest,
    MemoryInputModel,
    RecallMemoriesRequest,
    RecallRecentMemoriesRequest,
    SaveMemoriesRequest,
    delete_memory_endpoint,
    recall_memories_endpoint,
    recall_recent_memories_endpoint,
    save_memories_endpoint,
    app,
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

    async def test_recall_recent_memories_endpoint_defaults_and_overrides(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            await save_memories_endpoint(
                SaveMemoriesRequest(
                    memories=[
                        MemoryInputModel(content=f"recent {index}")
                        for index in range(6)
                    ],
                )
            )
            default_payload = await recall_recent_memories_endpoint(
                RecallRecentMemoriesRequest()
            )
            override_payload = await recall_recent_memories_endpoint(
                RecallRecentMemoriesRequest(top_k=2)
            )
        self.assertEqual(len(default_payload["memories"]), 5)
        self.assertEqual(len(override_payload["memories"]), 2)
        self.assertEqual(override_payload["mode"], "recent")

    async def test_recall_recent_memories_http_get_post_and_validation(self) -> None:
        transport = httpx.ASGITransport(app=app)
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                saved = await client.post(
                    "/save_memories",
                    json={
                        "memories": [
                            {"content": f"http recent {index}"}
                            for index in range(6)
                        ]
                    },
                )
                self.assertEqual(saved.status_code, 200)
                get_response = await client.get("/recall_recent_memories")
                post_response = await client.post(
                    "/recall_recent_memories", json={"top_k": 2}
                )
                invalid_get = await client.get(
                    "/recall_recent_memories", params={"top_k": 0}
                )
                invalid_post = await client.post(
                    "/recall_recent_memories", json={"top_k": 0}
                )
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(len(get_response.json()["memories"]), 5)
        self.assertEqual(post_response.status_code, 200)
        self.assertEqual(len(post_response.json()["memories"]), 2)
        self.assertEqual(invalid_get.status_code, 422)
        self.assertEqual(invalid_post.status_code, 422)

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
