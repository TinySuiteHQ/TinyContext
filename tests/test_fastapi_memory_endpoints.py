from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from tinycontext.servers.fastapi_server import (
    DeleteMemoryRequest,
    GetMemoryRequest,
    ListMemoriesRequest,
    MemoryInputModel,
    RecallMemoriesRequest,
    SaveMemoriesRequest,
    UpdateMemoryRequest,
    delete_memory_endpoint,
    get_memory_endpoint,
    list_memories_endpoint,
    recall_memories_endpoint,
    save_memories_endpoint,
    update_memory_endpoint,
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

    async def test_recall_memories_endpoint_includes_profile_block(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            await save_memories_endpoint(
                SaveMemoriesRequest(
                    memories=[
                        MemoryInputModel(content="Call the user Marcell", kind="profile")
                    ],
                )
            )
            await save_memories_endpoint(
                SaveMemoriesRequest(
                    memories=[MemoryInputModel(content="project uses sqlite")],
                )
            )
            payload = await recall_memories_endpoint(
                RecallMemoriesRequest(query="sqlite")
            )
        self.assertEqual(len(payload["profile"]), 1)
        self.assertEqual(payload["profile"][0]["content"], "Call the user Marcell")
        self.assertEqual(len(payload["memories"]), 1)
        self.assertEqual(payload["memories"][0]["content"], "project uses sqlite")

    async def test_recall_memories_endpoint_recent_mode_defaults_and_overrides(self) -> None:
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
            default_payload = await recall_memories_endpoint(RecallMemoriesRequest())
            override_payload = await recall_memories_endpoint(
                RecallMemoriesRequest(top_k=2)
            )
        self.assertEqual(len(default_payload["memories"]), 5)
        self.assertEqual(len(override_payload["memories"]), 2)
        self.assertEqual(override_payload["mode"], "recent")

    async def test_recall_memories_recent_mode_http_get_post_and_validation(self) -> None:
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
                get_response = await client.get("/recall_memories")
                post_response = await client.post(
                    "/recall_memories", json={"top_k": 2}
                )
                invalid_get = await client.get(
                    "/recall_memories", params={"top_k": 0}
                )
                invalid_post = await client.post(
                    "/recall_memories", json={"top_k": 0}
                )
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(len(get_response.json()["memories"]), 5)
        self.assertEqual(post_response.status_code, 200)
        self.assertEqual(len(post_response.json()["memories"]), 2)
        self.assertEqual(invalid_get.status_code, 422)
        self.assertEqual(invalid_post.status_code, 422)

    async def test_update_memory_endpoint(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            saved = await save_memories_endpoint(
                SaveMemoriesRequest(
                    memories=[MemoryInputModel(content="uses MySQL")],
                )
            )
            memory_id = saved["saved"][0]["id"]
            payload = await update_memory_endpoint(
                UpdateMemoryRequest(memory_id=memory_id, content="uses Postgres now")
            )
        self.assertNotEqual(payload["id"], memory_id)
        self.assertEqual(payload["supersedes"]["id"], memory_id)

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

    async def test_list_memories_endpoint_paginates_newest_first(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            await save_memories_endpoint(
                SaveMemoriesRequest(
                    memories=[
                        MemoryInputModel(content=f"listed {index}") for index in range(3)
                    ],
                )
            )
            first_page = await list_memories_endpoint(ListMemoriesRequest(limit=2))
            second_page = await list_memories_endpoint(
                ListMemoriesRequest(limit=2, offset=2)
            )
        self.assertEqual(
            [memory["content"] for memory in first_page["memories"]],
            ["listed 2", "listed 1"],
        )
        self.assertTrue(first_page["has_more"])
        self.assertEqual(
            [memory["content"] for memory in second_page["memories"]], ["listed 0"]
        )
        self.assertFalse(second_page["has_more"])
        self.assertEqual(first_page["total_count"], 3)

    async def test_list_memories_endpoint_stale_sort(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            await save_memories_endpoint(
                SaveMemoriesRequest(memories=[MemoryInputModel(content="cold entry")])
            )
            payload = await list_memories_endpoint(ListMemoriesRequest(sort="stale"))
        self.assertEqual(payload["sort"], "stale")
        self.assertEqual(payload["memories"][0]["recall_count"], 0)

    async def test_list_memories_endpoint_filters_by_since_until(self) -> None:
        transport = httpx.ASGITransport(app=app)
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/list_memories", params={"since": "2099-01-01T00:00:00Z"}
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["memories"], [])

    async def test_get_memory_endpoint_returns_full_content(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            saved = await save_memories_endpoint(
                SaveMemoriesRequest(
                    memories=[MemoryInputModel(content="inspect this fully")],
                )
            )
            payload = await get_memory_endpoint(
                GetMemoryRequest(memory_id=saved["saved"][0]["ref"])
            )
        self.assertEqual(payload["content"], "inspect this fully")
        self.assertEqual(payload["id"], saved["saved"][0]["id"])

    async def test_get_memory_endpoint_maps_not_found_error(self) -> None:
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.config,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await get_memory_endpoint(GetMemoryRequest(memory_id="missing-id"))
        self.assertEqual(ctx.exception.status_code, 404)

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
