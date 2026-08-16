from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from tinycontext import core
from tinycontext.models import MemoryInput
from tinycontext.servers.fastapi_server import app
from tinycontext.servers.hosted_tenancy_middleware import HostedTenancyMiddleware
from tinycontext.services.hosted_tenancy_service import (
    HostedTenancyError,
    bind_hosted_user,
    load_hosted_tenancy_config,
    reset_hosted_user,
    resolve_hosted_user,
    tenant_config,
)
from tinycontext.services.memory_store_service import close_connection
from tests.embedding_fakes import start_fake_embeddings


class HostedTenancyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        start_fake_embeddings(self)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store_dir = Path(self._tmpdir.name) / "tenants"
        self.base_config = {
            "memory_db_path": str(Path(self._tmpdir.name) / "legacy.db"),
            "recall_top_k": 10,
            "recall_max_tokens": 2000,
            "encoding_name": "o200k_base",
        }
        self.env = {
            "TINYCONTEXT_TENANCY": "proxy-header",
            "TINYCONTEXT_TRUSTED_USER_HEADER": "X-TinyContext-User-Id",
            "TINYCONTEXT_TENANT_STORE_DIR": str(self.store_dir),
            "TINYCONTEXT_TENANT_SECRET": "a" * 32,
            "TINYCONTEXT_TRUSTED_PROXY_CIDRS": "127.0.0.0/8",
        }
        self.environment = patch.dict(os.environ, self.env, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def tearDown(self) -> None:
        for database in self.store_dir.glob("*.db"):
            # tenant_config() resolves its store directory before opening SQLite.
            # Resolve the globbed path too, otherwise aliases such as /var ->
            # /private/var miss the pooled connection. macOS permits deleting
            # that open file; Windows correctly rejects the temp-directory
            # cleanup.
            close_connection(database.resolve())
        self._tmpdir.cleanup()

    def _config_for(self, user_id: str) -> dict[str, object]:
        token = bind_hosted_user(user_id)
        try:
            return tenant_config(self.base_config)
        finally:
            reset_hosted_user(token)

    def test_user_databases_are_opaque_and_isolated(self) -> None:
        alice = self._config_for("alice@example.com")
        bob = self._config_for("bob@example.com")
        self.assertNotEqual(alice["memory_db_path"], bob["memory_db_path"])
        self.assertNotIn("alice", str(alice["memory_db_path"]))

        core.save_memories([MemoryInput(content="alice private preference")], config=alice)
        core.save_memories([MemoryInput(content="bob private preference")], config=bob)
        alice_result = core.recall_memories("private preference", config=alice)
        bob_result = core.recall_memories("private preference", config=bob)
        self.assertEqual(
            [row["content"] for row in alice_result["memories"]],
            ["alice private preference"],
        )
        self.assertEqual(
            [row["content"] for row in bob_result["memories"]],
            ["bob private preference"],
        )

    def test_session_id_is_scoped_inside_one_user_store(self) -> None:
        alice = self._config_for("alice")
        core.save_memories(
            [MemoryInput(content="first project only")], session_id="project-a", config=alice
        )
        core.save_memories(
            [MemoryInput(content="second project only")], session_id="project-b", config=alice
        )
        result = core.recall_memories("project", session_id="project-a", config=alice)
        self.assertEqual([row["content"] for row in result["memories"]], ["first project only"])

    def test_resolver_rejects_forged_or_invalid_requests(self) -> None:
        valid_headers = [(b"x-tinycontext-user-id", b"alice")]
        with self.assertRaises(HostedTenancyError):
            resolve_hosted_user({"client": ("192.0.2.10", 123), "headers": valid_headers})
        with self.assertRaises(HostedTenancyError):
            resolve_hosted_user({"client": ("127.0.0.1", 123), "headers": []})
        with self.assertRaises(HostedTenancyError):
            resolve_hosted_user(
                {
                    "client": ("127.0.0.1", 123),
                    "headers": [
                        (b"x-tinycontext-user-id", b"alice"),
                        (b"x-tinycontext-user-id", b"bob"),
                    ],
                }
            )
        self.assertFalse(self.store_dir.exists())

    async def test_fastapi_requires_proxy_identity_and_isolates_users(self) -> None:
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 4321))
        with patch(
            "tinycontext.servers.fastapi_server.load_context_config",
            return_value=self.base_config,
        ):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                denied = await client.post(
                    "/save_memories", json={"memories": [{"content": "must fail"}]}
                )
                self.assertEqual(denied.status_code, 401)

                for user, content in (("alice", "alice only"), ("bob", "bob only")):
                    response = await client.post(
                        "/save_memories",
                        headers={"X-TinyContext-User-Id": user},
                        json={"memories": [{"content": content}]},
                    )
                    self.assertEqual(response.status_code, 200)
                response = await client.post(
                    "/recall_memories",
                    headers={"X-TinyContext-User-Id": "alice"},
                    json={"query": "only"},
                )
                recent_response = await client.get(
                    "/recall_memories",
                    headers={"X-TinyContext-User-Id": "alice"},
                )
        self.assertEqual(
            [row["content"] for row in response.json()["memories"]], ["alice only"]
        )
        self.assertEqual(
            [row["content"] for row in recent_response.json()["memories"]],
            ["alice only"],
        )

    async def test_hosted_mcp_paths_share_the_identity_boundary(self) -> None:
        seen: list[str | None] = []

        async def downstream(_scope, _receive, send):
            seen.append(str(tenant_config(self.base_config)["memory_db_path"]))
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        protected = HostedTenancyMiddleware(downstream)
        for path in ("/mcp", "/mcp/sse"):
            transport = httpx.ASGITransport(app=protected, client=("127.0.0.1", 4321))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                denied = await client.get(path)
                allowed = await client.get(
                    path, headers={"X-TinyContext-User-Id": "alice"}
                )
            self.assertEqual(denied.status_code, 401)
            self.assertEqual(allowed.status_code, 204)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], seen[1])

    def test_invalid_hosted_configuration_fails_closed(self) -> None:
        with patch.dict(os.environ, {"TINYCONTEXT_TENANT_SECRET": "short"}):
            with self.assertRaises(HostedTenancyError):
                load_hosted_tenancy_config()
