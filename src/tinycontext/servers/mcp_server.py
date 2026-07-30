from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from starlette.datastructures import Headers
from starlette.routing import BaseRoute, Mount, Route

from tinycontext import core
from tinycontext.errors import MEMORY_ERROR_MAP, MemoryError
from tinycontext.models import MemoryInput
from tinycontext.services.context_config_service import load_context_config


def _mcp_host() -> str:
    return os.environ.get("MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _mcp_port() -> int:
    raw = os.environ.get("MCP_PORT", "8000").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("MCP_PORT must be an integer") from exc


def _mcp_cors_origins() -> list[str]:
    raw = os.environ.get("MCP_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _streamable_http_cors_middleware() -> list[Any]:
    from mcp.server.streamable_http import (
        MCP_PROTOCOL_VERSION_HEADER,
        MCP_SESSION_ID_HEADER,
    )
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware

    return [
        Middleware(
            CORSMiddleware,
            allow_origins=_mcp_cors_origins(),
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=[
                "accept",
                "content-type",
                MCP_SESSION_ID_HEADER,
                MCP_PROTOCOL_VERSION_HEADER,
            ],
            expose_headers=[MCP_SESSION_ID_HEADER, MCP_PROTOCOL_VERSION_HEADER],
        )
    ]


class _StreamablePathLegacySseBridge:
    def __init__(self, streamable_asgi: Any, sse_starlette: Any, sse_path: str) -> None:
        self._streamable = streamable_asgi
        self._sse = sse_starlette
        self._sse_path = sse_path

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._streamable(scope, receive, send)
            return
        if scope.get("method", "GET").upper() == "GET":
            headers = Headers(scope=scope)
            if not (headers.get("mcp-session-id") or "").strip():
                sse_scope = dict(scope)
                sse_scope["path"] = self._sse_path
                sse_scope["raw_path"] = self._sse_path.encode("ascii")
                await self._sse(sse_scope, receive, send)
                return
        await self._streamable(scope, receive, send)


def _route_identity(route: BaseRoute) -> tuple[Any, ...]:
    if isinstance(route, Route):
        methods = route.methods
        key_methods: tuple[str, ...] = (
            tuple(sorted(methods)) if methods is not None else ("*",)
        )
        return ("Route", route.path, key_methods)
    if isinstance(route, Mount):
        return ("Mount", route.path)
    return ("other", type(route).__name__, id(route))


async def _run_streamable_http_combined_async() -> None:
    import uvicorn
    from starlette.applications import Starlette

    stream_app = mcp.streamable_http_app()
    sse_starlette = mcp.sse_app()
    mcp_path = mcp.settings.streamable_http_path
    sse_path = mcp.settings.sse_path

    streamable_asgi: Any = None
    bridged_stream_routes: list[BaseRoute] = []
    for route in stream_app.routes:
        if isinstance(route, Route) and route.path == mcp_path:
            streamable_asgi = route.endpoint
            bridged_stream_routes.append(
                Route(
                    mcp_path,
                    endpoint=_StreamablePathLegacySseBridge(
                        streamable_asgi,
                        sse_starlette,
                        sse_path,
                    ),
                    methods=route.methods,
                )
            )
        else:
            bridged_stream_routes.append(route)

    if streamable_asgi is None:
        raise RuntimeError(f"No Route found for Streamable HTTP path {mcp_path!r}")

    primary_keys = {_route_identity(route) for route in bridged_stream_routes}
    extra_sse = [
        route
        for route in sse_starlette.routes
        if _route_identity(route) not in primary_keys
    ]
    app = Starlette(
        debug=mcp.settings.debug,
        routes=bridged_stream_routes + extra_sse,
        middleware=_streamable_http_cors_middleware() + stream_app.user_middleware,
        lifespan=stream_app.router.lifespan_context,
    )
    config = uvicorn.Config(
        app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
    )
    await uvicorn.Server(config).serve()


MCP_INSTRUCTIONS = """
This MCP server exposes two tools:

1. save_memories(memories)
2. recall_memories(query)

Use save_memories to persist short, durable facts, preferences, or session notes
for later retrieval. Each memory should be concise and self-contained.

Use recall_memories when you need relevant prior context for the current task.
Pass the user's question or task description as query. Results are ranked with
hybrid BM25 and dense retrieval, then trimmed to the token budget so they fit
local LLM context windows.
""".strip()


def _log(message: str) -> None:
    print(f"[tinycontext] {message}", file=sys.stderr, flush=True)


def _memory_tool_error(exc: MemoryError) -> ValueError:
    code = MEMORY_ERROR_MAP.get(type(exc), ("internal_error", 500))[0]
    return ValueError(f"{code}: {exc}")


def _normalize_memory_items(
    memories: list[dict[str, Any]],
) -> list[MemoryInput]:
    items: list[MemoryInput] = []
    for memory in memories:
        content = str(memory.get("content", "")).strip()
        tags_raw = memory.get("tags")
        metadata_raw = memory.get("metadata")
        tags = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else None
        metadata = metadata_raw if isinstance(metadata_raw, dict) else None
        items.append(MemoryInput(content=content, tags=tags, metadata=metadata))
    return items


mcp = FastMCP(
    "tinycontext",
    instructions=MCP_INSTRUCTIONS,
    host=_mcp_host(),
    port=_mcp_port(),
    sse_path="/mcp/sse",
    message_path="/mcp/messages/",
)


@mcp.tool(
    name="save_memories",
    title="Save Memories",
    description=(
        "Persist one or more concise memories for later recall. Each memory needs "
        "content; optional tags and metadata help with organization."
    ),
)
async def save_memories_tool(
    memories: Annotated[
        list[dict[str, Any]],
        Field(
            description=(
                "List of memory objects. Each object must include content and may "
                "include tags (list of strings) and metadata (object)."
            )
        ),
    ],
) -> dict[str, Any]:
    started = time.monotonic()
    _log(f"save_memories called count={len(memories)}")
    config = load_context_config()
    try:
        payload = core.save_memories(
            _normalize_memory_items(memories),
            config=config,
        )
    except MemoryError as exc:
        elapsed = time.monotonic() - started
        code = MEMORY_ERROR_MAP.get(type(exc), ("internal_error", 500))[0]
        _log(f"save_memories failed elapsed={elapsed:.2f}s code={code} error={exc!r}")
        raise _memory_tool_error(exc) from exc
    elapsed = time.monotonic() - started
    _log(f"save_memories returning saved={len(payload['saved'])} elapsed={elapsed:.2f}s")
    return payload


@mcp.tool(
    name="recall_memories",
    title="Recall Memories",
    description=(
        "Retrieve ranked memories relevant to query within a token budget. "
        "Use this before answering when prior context may help."
    ),
)
async def recall_memories_tool(
    query: Annotated[
        str,
        Field(description="Question or task description to match against memories."),
    ],
) -> dict[str, Any]:
    started = time.monotonic()
    _log(f"recall_memories called query={query!r}")
    config = load_context_config()
    try:
        payload = core.recall_memories(
            query,
            config=config,
        )
    except MemoryError as exc:
        elapsed = time.monotonic() - started
        code = MEMORY_ERROR_MAP.get(type(exc), ("internal_error", 500))[0]
        _log(f"recall_memories failed elapsed={elapsed:.2f}s code={code} error={exc!r}")
        raise _memory_tool_error(exc) from exc
    elapsed = time.monotonic() - started
    _log(
        "recall_memories returning "
        f"count={len(payload['memories'])} total_tokens={payload['total_tokens']} "
        f"elapsed={elapsed:.2f}s"
    )
    return payload


def main() -> None:
    from tinycontext.services.onnx_bundle_service import ensure_onnx_bundle_sync

    config = load_context_config()
    ensure_onnx_bundle_sync(
        str(config["embedding_model"]),
        models_dir=str(config["models_dir"]),
    )
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip() or "stdio"
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(
            "MCP_TRANSPORT must be one of: stdio, sse, streamable-http "
            "(default stdio for IDE-spawned MCP; set env only for standalone HTTP/SSE)"
        )
    if transport == "streamable-http":
        import anyio

        anyio.run(_run_streamable_http_combined_async)
    else:
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
