"""Read-only management views for a client UI (the ``tinysuite.manage/1`` contract).

A client that understands the contract finds a resource with the
``application/vnd.tinysuite.manage+json`` media type in ``resources/list``,
reads it as a manifest, and then reads the views it names. Everything goes
over the same MCP session as the tools, so the same identity rules apply:
in hosted mode every view is the calling user's own store. Changes go through
the ordinary ``update_memory`` / ``delete_memory`` tools, which the manifest
names; nothing here writes.

Off unless ``TINYCONTEXT_MANAGEMENT=on``. Over HTTP it also needs an
authenticated caller -- hosted tenancy's trusted proxy, or a bearer
``TINYCONTEXT_MANAGEMENT_TOKEN`` -- or the server refuses to start. The
instance-wide view (store count, disk use) is only offered to a token holder
or to a local stdio client: in hosted mode an ordinary user's connection has
no business learning how many other users there are.
"""

from __future__ import annotations

import hmac
import importlib.metadata
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote

from tinycontext import core
from tinycontext.config import resolve_config
from tinycontext.models import MEMORY_KINDS
from tinycontext.services.hosted_tenancy_service import (
    hosted_tenancy_enabled,
    http_transport,
    load_hosted_tenancy_config,
    mcp_message_scope,
)
from tinycontext.services.memory_store_service import memory_stats, search_match_query

MANIFEST_URI = "tinycontext://manage/manifest"
MANIFEST_MIME = "application/vnd.tinysuite.manage+json"
CONTRACT = "tinysuite.manage/1"
VIEW_URI = "tinycontext://manage/views/{view}/{{params}}"
ITEM_URI = "tinycontext://manage/items/memory/{id}"

_ON = {"on", "1", "true", "yes"}
_OFF = {"", "off", "0", "false", "no"}
_MIN_TOKEN_CHARS = 24
_PAGE_MAX = 100
_SEARCH_MAX = 200


class ManagementAccessError(PermissionError):
    """The caller may not read this management view."""


def management_enabled() -> bool:
    raw = os.environ.get("TINYCONTEXT_MANAGEMENT", "").strip().lower()
    if raw in _ON:
        return True
    if raw in _OFF:
        return False
    raise ValueError("TINYCONTEXT_MANAGEMENT must be 'on' or 'off'")


def management_token() -> str | None:
    token = os.environ.get("TINYCONTEXT_MANAGEMENT_TOKEN", "").strip()
    if token and len(token) < _MIN_TOKEN_CHARS:
        raise ValueError(
            f"TINYCONTEXT_MANAGEMENT_TOKEN must be at least {_MIN_TOKEN_CHARS} characters"
        )
    return token or None


def check_management_startup(transport: str) -> None:
    """Fail closed: management over HTTP must have someone to authenticate."""
    if not management_enabled():
        return
    token = management_token()
    if transport != "stdio" and not token and not hosted_tenancy_enabled():
        raise ValueError(
            "TINYCONTEXT_MANAGEMENT=on over HTTP needs hosted tenancy "
            "(TINYCONTEXT_TENANCY=proxy-header) or TINYCONTEXT_MANAGEMENT_TOKEN"
        )


def _bearer(scope: Mapping[str, Any]) -> str | None:
    for key, value in scope.get("headers", []):
        if bytes(key).lower() == b"authorization":
            text = bytes(value).decode("latin-1").strip()
            if text[:7].lower() == "bearer ":
                return text[7:].strip()
    return None


def _token_matches(scope: Mapping[str, Any]) -> bool:
    token = management_token()
    supplied = _bearer(scope)
    return bool(token and supplied and hmac.compare_digest(token, supplied))


def _is_local_operator(scope: Mapping[str, Any] | None) -> bool:
    # No HTTP request *and* a stdio server: decided by how the server was
    # started, so a missing request over HTTP is never mistaken for one.
    return scope is None and not http_transport()


def _instance_allowed() -> bool:
    scope = mcp_message_scope()
    if _is_local_operator(scope):
        return True
    return scope is not None and _token_matches(scope)


def authorize(view_scope: str) -> None:
    """Raise unless this call may read a view of the given scope.

    A local stdio client is the operator. Over HTTP, a configured token must
    be presented; without one, only hosted mode's per-user views are open,
    since the trusted proxy already authenticated the caller and the views
    only ever show that caller's own store.
    """
    if not management_enabled():
        raise ManagementAccessError("management views are disabled on this server")
    scope = mcp_message_scope()
    if _is_local_operator(scope):
        return
    if scope is None:
        raise ManagementAccessError("management views need an HTTP request to authenticate")
    if management_token():
        if not _token_matches(scope):
            raise ManagementAccessError("management views need the management token")
        return
    if view_scope == "user" and hosted_tenancy_enabled():
        return
    raise ManagementAccessError("this management view is not available to this caller")


def _version() -> str:
    try:
        return importlib.metadata.version("tinysuite-context")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def manifest() -> dict[str, Any]:
    authorize("user")
    views: list[dict[str, Any]] = [
        {
            "id": "memories",
            "title": "Memories",
            "kind": "table",
            "scope": "user",
            "resource": VIEW_URI.format(view="memories"),
            "pageSize": 50,
            "params": [
                {"name": "q", "label": "Search", "type": "text", "maxLength": _SEARCH_MAX},
                {"name": "kind", "label": "Type", "type": "enum", "values": sorted(MEMORY_KINDS)},
                {"name": "sort", "label": "Order", "type": "enum",
                 "values": ["relevance", "recent", "stale"]},
            ],
            "rowKey": "id",
            "columns": [
                {"key": "content", "label": "Memory", "format": "text"},
                {"key": "kind", "label": "Type", "format": "text"},
                {"key": "created_at", "label": "Saved", "format": "datetime"},
                {"key": "recall_count", "label": "Recalled", "format": "number"},
            ],
            "detail": {"resource": ITEM_URI},
            "actions": [
                {
                    "id": "edit",
                    "label": "Edit",
                    "tool": "update_memory",
                    "input": {
                        "label": "Memory",
                        "initial": "content",
                        "maxLength": 20000,
                        "multiline": True,
                    },
                    "args": {"memory_id": "$row.id", "content": "$input"},
                },
                {
                    "id": "delete",
                    "label": "Delete",
                    "tool": "delete_memory",
                    "args": {"memory_id": "$row.id"},
                    "destructive": True,
                    "confirm": "Delete this memory? This cannot be undone.",
                },
            ],
        },
        {
            "id": "overview",
            "title": "Overview",
            "kind": "stats",
            "scope": "user",
            "resource": VIEW_URI.format(view="overview"),
        },
    ]
    if _instance_allowed():
        views.append(
            {
                "id": "service",
                "title": "Service",
                "kind": "stats",
                "scope": "instance",
                "resource": VIEW_URI.format(view="service"),
            }
        )
    return {
        "contract": CONTRACT,
        "id": "tinycontext",
        "title": "TinyContext",
        "description": "Memories the assistant keeps between conversations.",
        "icon": "memory",
        "version": _version(),
        "views": views,
    }


def decode_params(raw: str, allowed: set[str]) -> dict[str, str]:
    """The view parameters a client percent-encoded into one URI segment."""
    if raw in {"", "-"}:
        return {}
    pairs = parse_qsl(unquote(raw), keep_blank_values=True, max_num_fields=16)
    params: dict[str, str] = {}
    for key, value in pairs:
        if key not in allowed:
            raise ValueError(f"unknown view parameter {key!r}")
        params[key] = value
    return params


def _int_param(params: Mapping[str, str], key: str, default: int, low: int, high: int) -> int:
    raw = params.get(key)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    return max(low, min(high, value))


def memories_view(raw: str, config: Mapping[str, Any]) -> dict[str, Any]:
    authorize("user")
    params = decode_params(raw, {"q", "kind", "sort", "offset", "limit"})
    search = (params.get("q") or "")[:_SEARCH_MAX] or None
    sort = params.get("sort") or None
    if sort == "relevance" and search_match_query(search) is None:
        sort = None  # nothing to rank by: the default order
    page = core.list_memories(
        kind=params.get("kind") or None,
        sort=sort,
        search=search,
        offset=_int_param(params, "offset", 0, 0, 10_000_000),
        limit=_int_param(params, "limit", 50, 1, _PAGE_MAX),
        config=config,
    )
    rows = [
        {
            "id": row["id"],
            "ref": row["ref"],
            "content": row["content"],
            "kind": row["kind"],
            "created_at": row["created_at"],
            "recall_count": row["recall_count"],
            "last_recalled_at": row["last_recalled_at"],
        }
        for row in page["memories"]
    ]
    return {
        "rows": rows,
        "total": page["total_count"],
        "offset": page["offset"],
        "limit": page["limit"],
        "hasMore": page["has_more"],
    }


def memory_detail(memory_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    authorize("user")
    item = core.get_memory(unquote(memory_id), config=config)
    fields = [
        {"label": "Memory", "value": item["content"], "format": "multiline"},
        {"label": "Type", "value": item["kind"], "format": "text"},
        {"label": "Saved", "value": item["created_at"], "format": "datetime"},
        {"label": "Reference", "value": item["ref"], "format": "text"},
        {"label": "Session", "value": item["session_id"], "format": "text"},
        {"label": "Recalled", "value": item["recall_count"], "format": "number"},
        {"label": "Last recalled", "value": item["last_recalled_at"], "format": "datetime"},
        {"label": "Tokens", "value": item["content_tokens"], "format": "number"},
    ]
    if item["superseded_by"]:
        fields.append({"label": "Replaced by", "value": item["superseded_by"], "format": "text"})
    return {
        "fields": fields,
        "row": {"id": item["id"], "ref": item["ref"], "content": item["content"], "kind": item["kind"]},
    }


def overview_view(raw: str, config: Mapping[str, Any]) -> dict[str, Any]:
    authorize("user")
    decode_params(raw, set())
    db_path = Path(str(resolve_config(config).to_dict()["memory_db_path"]))
    stats = memory_stats(db_path) if db_path.exists() else {}
    return {
        "items": [
            {"label": "Memories", "value": stats.get("active", 0)},
            {"label": "Profile facts", "value": stats.get("profile", 0)},
            {"label": "Episodic", "value": stats.get("episodic", 0)},
            {"label": "Never recalled", "value": stats.get("never_recalled", 0),
             "hint": "Candidates to review, merge or delete."},
            {"label": "Replaced by edits", "value": stats.get("superseded", 0)},
            {"label": "Oldest", "value": stats.get("oldest"), "format": "datetime"},
            {"label": "Newest", "value": stats.get("newest"), "format": "datetime"},
            {"label": "Last recalled", "value": stats.get("last_recalled"), "format": "datetime"},
        ]
    }


def _store_bytes(path: Path) -> int:
    total = 0
    for suffix in ("", "-wal", "-shm"):
        candidate = path.with_name(path.name + suffix)
        if candidate.exists():
            total += candidate.stat().st_size
    return total


def service_view(raw: str, base_config: Mapping[str, Any]) -> dict[str, Any]:
    authorize("instance")
    if not _instance_allowed():
        raise ManagementAccessError("the service view needs the management token")
    decode_params(raw, set())
    resolved = resolve_config(base_config).to_dict()
    if hosted_tenancy_enabled():
        stores = sorted(load_hosted_tenancy_config().store_dir.glob("*.db"))
        mode = "one store per user (hosted)"
    else:
        single = Path(str(resolved["memory_db_path"]))
        stores = [single] if single.exists() else []
        mode = "single store"
    return {
        "items": [
            {"label": "Version", "value": _version()},
            {"label": "Storage", "value": mode},
            {"label": "Stores", "value": len(stores)},
            {"label": "Disk used", "value": sum(_store_bytes(p) for p in stores), "unit": "bytes"},
            {"label": "Embedding backend", "value": str(resolved["embedding_backend"])},
            {"label": "Embedding model", "value": str(resolved["embedding_model"])},
        ]
    }
