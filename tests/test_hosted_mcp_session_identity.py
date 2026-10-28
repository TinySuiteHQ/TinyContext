"""Hosted identity must be decided per MCP message, not per MCP session.

These run the real streamable-HTTP app and the real MCP client: the bug they
guard against lives in how the SDK schedules session handlers, which a
middleware-only test cannot see.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.embedding_fakes import start_fake_embeddings
from tests.mcp_http_harness import RunningMcpServer, mcp_session
from tinycontext.services.memory_store_service import close_connection

HEADER = "X-TinyContext-User-Id"


def _text(result) -> str:  # noqa: ANN001
    return "\n".join(getattr(part, "text", "") for part in result.content)


class HostedMcpSessionIdentityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        start_fake_embeddings(self)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store_dir = Path(self._tmpdir.name) / "tenants"
        base_config = {
            "memory_db_path": str(Path(self._tmpdir.name) / "legacy.db"),
            "recall_top_k": 10,
            "recall_max_tokens": 2000,
            "encoding_name": "o200k_base",
        }
        environment = patch.dict(
            os.environ,
            {
                "TINYCONTEXT_TENANCY": "proxy-header",
                "TINYCONTEXT_TRUSTED_USER_HEADER": HEADER,
                "TINYCONTEXT_TENANT_STORE_DIR": str(self.store_dir),
                "TINYCONTEXT_TENANT_SECRET": "s" * 32,
                "TINYCONTEXT_TRUSTED_PROXY_CIDRS": "127.0.0.0/8",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        config_patch = patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=base_config,
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)

    def tearDown(self) -> None:
        for database in self.store_dir.glob("*.db"):
            close_connection(database.resolve())
        self._tmpdir.cleanup()

    async def test_one_session_follows_each_requests_identity(self) -> None:
        current = {"user": "alice"}
        with RunningMcpServer() as server:
            async with mcp_session(server.url, lambda: {HEADER: current["user"]}) as session:
                await session.call_tool(
                    "save_memories", {"memories": [{"content": "alice private note"}]}
                )
                current["user"] = "bob"
                bob_view = _text(await session.call_tool("list_memories", {}))
                await session.call_tool(
                    "save_memories", {"memories": [{"content": "bob private note"}]}
                )
                current["user"] = "alice"
                alice_view = _text(await session.call_tool("list_memories", {}))

        self.assertNotIn("alice private note", bob_view)
        self.assertIn("alice private note", alice_view)
        self.assertNotIn("bob private note", alice_view)
        self.assertEqual(len(list(self.store_dir.glob("*.db"))), 2)

    async def test_concurrent_users_on_separate_sessions_stay_isolated(self) -> None:
        async def work(user: str) -> str:
            async with mcp_session(server.url, {HEADER: user}) as session:
                for index in range(3):
                    await session.call_tool(
                        "save_memories",
                        {"memories": [{"content": f"{user} note {index}"}]},
                    )
                return _text(await session.call_tool("list_memories", {}))

        with RunningMcpServer() as server:
            alice, bob = await asyncio.gather(work("alice"), work("bob"))

        self.assertIn("alice note 2", alice)
        self.assertNotIn("bob note", alice)
        self.assertIn("bob note 2", bob)
        self.assertNotIn("alice note", bob)


if __name__ == "__main__":
    unittest.main()
