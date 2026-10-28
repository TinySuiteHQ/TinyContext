"""The tinysuite.manage/1 management resources: discovery, views, access, isolation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote, urlencode

from pydantic import AnyUrl

import hashlib
from unittest.mock import patch as _patch

from tests.embedding_fakes import start_fake_embeddings
from tests.mcp_http_harness import RunningMcpServer, mcp_session
from tinycontext import MemoryInput, core
from tinycontext.services import management_service as manage
from tinycontext.services.memory_store_service import close_connection
from tinycontext.servers import mcp_server

HEADER = "X-TinyContext-User-Id"
TOKEN = "t" * 32


def _distinct_embeddings(inputs, **_kwargs):  # noqa: ANN001
    """A different direction per text, so no test memory is dropped as a duplicate."""
    vectors = []
    for text in inputs:
        raw = [b - 127.5 for b in hashlib.sha256(text.encode()).digest()[:16]]
        norm = sum(v * v for v in raw) ** 0.5
        vectors.append([v / norm for v in raw])
    return vectors


def start_distinct_embeddings(test_case) -> None:  # noqa: ANN001
    start_fake_embeddings(test_case)
    for target in (
        "tinycontext.core.embed_texts",
        "tinycontext.pipelines.memory_recall.embed_texts",
        "tinycontext.services.embedding_reindex_service.embed_texts",
    ):
        patcher = _patch(target, side_effect=_distinct_embeddings)
        patcher.start()
        test_case.addCleanup(patcher.stop)


def _close_all(root: Path) -> None:
    # Pooled connections are keyed by the path as opened; a temp dir can
    # have a short (8.3) alias on Windows, so close under both spellings.
    for database in root.rglob("*.db"):
        close_connection(database)
        close_connection(database.resolve())


def _params(**values: object) -> str:
    return quote(urlencode(values), safe="") if values else "-"


async def _read(session, uri: str) -> dict:  # noqa: ANN001
    result = await session.read_resource(AnyUrl(uri))
    return json.loads(result.contents[0].text)


class _Base(unittest.IsolatedAsyncioTestCase):
    env: dict[str, str] = {}

    def setUp(self) -> None:
        start_distinct_embeddings(self)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.base_config = {
            "memory_db_path": str(self.root / "memories.db"),
            "recall_top_k": 10,
            "recall_max_tokens": 2000,
            "encoding_name": "o200k_base",
        }
        environment = patch.dict(
            os.environ,
            {key: value.replace("{root}", str(self.root)) for key, value in self.env.items()},
        )
        environment.start()
        self.addCleanup(environment.stop)
        config_patch = patch(
            "tinycontext.servers.mcp_server.load_context_config",
            return_value=self.base_config,
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)
        mcp_server.register_management_resources()

    def tearDown(self) -> None:
        _close_all(self.root)
        self._tmpdir.cleanup()


HOSTED = {
    "TINYCONTEXT_MANAGEMENT": "on",
    "TINYCONTEXT_TENANCY": "proxy-header",
    "TINYCONTEXT_TRUSTED_USER_HEADER": HEADER,
    "TINYCONTEXT_TENANT_STORE_DIR": "{root}/tenants",
    "TINYCONTEXT_TENANT_SECRET": "s" * 32,
    "TINYCONTEXT_TRUSTED_PROXY_CIDRS": "127.0.0.0/8",
}


class HostedManagementTests(_Base):
    env = HOSTED

    async def test_manifest_is_discoverable_by_media_type(self) -> None:
        with RunningMcpServer() as server:
            async with mcp_session(server.url, {HEADER: "alice"}) as session:
                listed = await session.list_resources()
                found = [r for r in listed.resources if r.mimeType == manage.MANIFEST_MIME]
                self.assertEqual(len(found), 1)
                manifest = await _read(session, str(found[0].uri))
        self.assertEqual(manifest["contract"], "tinysuite.manage/1")
        self.assertEqual([v["id"] for v in manifest["views"]], ["memories", "overview"])
        actions = {a["id"]: a for a in manifest["views"][0]["actions"]}
        self.assertEqual(actions["delete"]["tool"], "delete_memory")
        self.assertTrue(actions["delete"]["destructive"])

    async def test_each_user_sees_only_their_own_memories(self) -> None:
        with RunningMcpServer() as server:
            for user in ("alice", "bob"):
                async with mcp_session(server.url, {HEADER: user}) as session:
                    await session.call_tool(
                        "save_memories",
                        {"memories": [{"content": f"{user} likes tea"}, {"content": f"{user} works remote"}]},
                    )
            async with mcp_session(server.url, {HEADER: "alice"}) as session:
                page = await _read(session, manage.VIEW_URI.format(view="memories").format(params="-"))
                alice_id = page["rows"][0]["id"]
                overview = await _read(session, manage.VIEW_URI.format(view="overview").format(params="-"))
            async with mcp_session(server.url, {HEADER: "bob"}) as session:
                with self.assertRaises(Exception):
                    await _read(session, manage.ITEM_URI.format(id=alice_id))
                bob_page = await _read(session, manage.VIEW_URI.format(view="memories").format(params="-"))

        self.assertEqual(page["total"], 2)
        self.assertTrue(all(row["content"].startswith("alice") for row in page["rows"]))
        self.assertTrue(all(row["content"].startswith("bob") for row in bob_page["rows"]))
        self.assertEqual(overview["items"][0], {"label": "Memories", "value": 2})

    async def test_search_paging_and_detail(self) -> None:
        with RunningMcpServer() as server:
            async with mcp_session(server.url, {HEADER: "alice"}) as session:
                await session.call_tool(
                    "save_memories",
                    {"memories": [{"content": f"note {i} about Python"} for i in range(5)]
                     + [{"content": "prefers green tea"}, {"content": "tea tea tea, always tea"}]},
                )
                memories = manage.VIEW_URI.format(view="memories")
                python = await _read(session, memories.format(params=_params(q="python", limit=2)))
                second = await _read(session, memories.format(params=_params(q="python", limit=2, offset=2)))
                typing = await _read(session, memories.format(params=_params(q="pyth")))
                ranked = await _read(session, memories.format(params=_params(q="tea")))
                both = await _read(session, memories.format(params=_params(q="green tea")))
                detail = await _read(session, manage.ITEM_URI.format(id=python["rows"][0]["id"]))
                with self.assertRaises(Exception):
                    await _read(session, memories.format(params=_params(owner="bob")))

        self.assertEqual((python["total"], len(python["rows"]), python["hasMore"]), (5, 2, True))
        self.assertNotEqual(python["rows"][0]["id"], second["rows"][0]["id"])
        self.assertEqual(typing["total"], 5, "the last word matches as a prefix")
        self.assertEqual(ranked["rows"][0]["content"], "tea tea tea, always tea", "best BM25 match first")
        self.assertEqual([r["content"] for r in both["rows"]], ["prefers green tea"], "every word must match")
        self.assertEqual(detail["row"]["id"], python["rows"][0]["id"])
        self.assertIn("Python", detail["fields"][0]["value"])

    async def test_service_view_is_withheld_without_a_token(self) -> None:
        with RunningMcpServer() as server:
            async with mcp_session(server.url, {HEADER: "alice"}) as session:
                with self.assertRaises(Exception):
                    await _read(session, manage.VIEW_URI.format(view="service").format(params="-"))

    async def test_edits_go_through_the_ordinary_tools_and_stay_scoped(self) -> None:
        with RunningMcpServer() as server:
            async with mcp_session(server.url, {HEADER: "alice"}) as session:
                saved = await session.call_tool("save_memories", {"memories": [{"content": "old text"}]})
                memory_id = json.loads(saved.content[0].text)["saved"][0]["id"]
            async with mcp_session(server.url, {HEADER: "bob"}) as session:
                denied = await session.call_tool("delete_memory", {"memory_id": memory_id})
            async with mcp_session(server.url, {HEADER: "alice"}) as session:
                await session.call_tool("update_memory", {"memory_id": memory_id, "content": "new text"})
                page = await _read(session, manage.VIEW_URI.format(view="memories").format(params="-"))
        self.assertTrue(denied.isError)
        self.assertEqual([r["content"] for r in page["rows"]], ["new text"])


class TokenManagementTests(_Base):
    env = {**HOSTED, "TINYCONTEXT_MANAGEMENT_TOKEN": TOKEN}

    async def test_the_token_is_required_and_unlocks_the_service_view(self) -> None:
        with RunningMcpServer() as server:
            async with mcp_session(server.url, {HEADER: "alice"}) as session:
                with self.assertRaises(Exception):
                    await _read(session, manage.MANIFEST_URI)
            async with mcp_session(server.url, {HEADER: "alice", "Authorization": "Bearer wrong" + TOKEN}) as session:
                with self.assertRaises(Exception):
                    await _read(session, manage.MANIFEST_URI)
            async with mcp_session(server.url, {HEADER: "alice", "Authorization": f"Bearer {TOKEN}"}) as session:
                manifest = await _read(session, manage.MANIFEST_URI)
                service = await _read(session, manage.VIEW_URI.format(view="service").format(params="-"))
        self.assertIn("service", [v["id"] for v in manifest["views"]])
        labels = {item["label"]: item["value"] for item in service["items"]}
        self.assertEqual(labels["Storage"], "one store per user (hosted)")


class AccessRuleTests(unittest.TestCase):
    def test_off_by_default(self) -> None:
        with patch.dict(os.environ, {"TINYCONTEXT_MANAGEMENT": ""}):
            self.assertFalse(manage.management_enabled())
            with self.assertRaises(manage.ManagementAccessError):
                manage.authorize("user")

    def test_http_without_authentication_refuses_to_start(self) -> None:
        with patch.dict(os.environ, {"TINYCONTEXT_MANAGEMENT": "on", "TINYCONTEXT_TENANCY": "",
                                     "TINYCONTEXT_MANAGEMENT_TOKEN": ""}):
            with self.assertRaises(ValueError):
                manage.check_management_startup("streamable-http")
            manage.check_management_startup("stdio")
        with patch.dict(os.environ, {"TINYCONTEXT_MANAGEMENT": "on", "TINYCONTEXT_TENANCY": "",
                                     "TINYCONTEXT_MANAGEMENT_TOKEN": TOKEN}):
            manage.check_management_startup("sse")

    def test_short_tokens_and_bad_switches_are_rejected(self) -> None:
        with patch.dict(os.environ, {"TINYCONTEXT_MANAGEMENT_TOKEN": "short"}):
            with self.assertRaises(ValueError):
                manage.management_token()
        with patch.dict(os.environ, {"TINYCONTEXT_MANAGEMENT": "maybe"}):
            with self.assertRaises(ValueError):
                manage.management_enabled()

    def test_a_local_client_is_the_operator(self) -> None:
        with patch.dict(os.environ, {"TINYCONTEXT_MANAGEMENT": "on", "TINYCONTEXT_TENANCY": ""}):
            manage.authorize("instance")
            self.assertIn("service", [v["id"] for v in manage.manifest()["views"]])

    def test_an_http_message_without_its_request_is_refused_not_trusted(self) -> None:
        # If the SDK ever stops attaching each message's HTTP request, nothing
        # may fall back to the session-wide identity or to "local operator".
        from tinycontext.services import hosted_tenancy_service as tenancy

        hosted = {**{k: v.replace("{root}", tempfile.gettempdir()) for k, v in HOSTED.items()},
                  "MCP_TRANSPORT": "streamable-http"}
        with patch.dict(os.environ, hosted),                 patch.object(tenancy, "mcp_message_context", return_value=(True, None)),                 patch.object(manage, "mcp_message_scope", return_value=None):
            token = tenancy.bind_hosted_user("first-user-of-the-session")
            try:
                with self.assertRaises(tenancy.HostedTenancyError):
                    tenancy.current_hosted_user()
            finally:
                tenancy.reset_hosted_user(token)
            with self.assertRaises(manage.ManagementAccessError):
                manage.authorize("user")
            self.assertFalse(manage._instance_allowed())

    def test_unknown_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            manage.decode_params(quote("q=a&user=bob", safe=""), {"q"})
        self.assertEqual(manage.decode_params("-", {"q"}), {})


class SearchFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        start_distinct_embeddings(self)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.config = {"memory_db_path": str(Path(self._tmpdir.name) / "m.db"), "encoding_name": "o200k_base"}
        core.save_memories(
            [MemoryInput(content=c) for c in (
                "Likes PYTHON for scripts", "Python Python Python everywhere",
                "uses snake_case names", "drinks coffee", 'quotes "AND" OR NOT*',
            )],
            config=self.config,
        )

    def tearDown(self) -> None:
        _close_all(Path(self._tmpdir.name))
        self._tmpdir.cleanup()

    def _list(self, search: str | None, **kwargs) -> dict:  # noqa: ANN003
        return core.list_memories(search=search, config=self.config, **kwargs)

    def _contents(self, search: str | None, **kwargs) -> list[str]:  # noqa: ANN003
        return [m["content"] for m in self._list(search, **kwargs)["memories"]]

    def test_keyword_search_is_ranked_and_case_insensitive(self) -> None:
        page = self._list("python")
        self.assertEqual(page["sort"], "relevance")
        self.assertEqual(page["total_count"], 2)
        self.assertEqual(page["memories"][0]["content"], "Python Python Python everywhere")

    def test_explicit_orders_still_apply_to_a_search(self) -> None:
        self.assertEqual(self._contents("python", sort="recent")[0], "Python Python Python everywhere")
        self.assertEqual(len(self._contents("python", sort="stale")), 2)

    def test_every_word_must_match_and_the_last_is_a_prefix(self) -> None:
        self.assertEqual(self._contents("likes pyth"), ["Likes PYTHON for scripts"])
        self.assertEqual(self._contents("snake_case"), ["uses snake_case names"])
        self.assertEqual(self._contents("coffee python"), [])

    def test_search_syntax_in_the_text_is_inert(self) -> None:
        self.assertEqual(self._contents('"AND" OR NOT*'), ['quotes "AND" OR NOT*'])
        self.assertEqual(self._contents("NEAR(a b)"), [])

    def test_no_words_means_no_search(self) -> None:
        for blank in (None, "", "   ", "%%", "*"):
            with self.subTest(blank=blank):
                page = self._list(blank)
                self.assertEqual((page["total_count"], page["sort"]), (5, "recent"))

    def test_relevance_needs_a_search_and_long_text_is_rejected(self) -> None:
        from tinycontext.errors import InvalidFilterError, InvalidSortError

        with self.assertRaises(InvalidSortError):
            self._list(None, sort="relevance")
        with self.assertRaises(InvalidFilterError):
            self._list("x" * 201)


if __name__ == "__main__":
    unittest.main()
