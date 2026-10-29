"""Trusted-proxy tenant isolation for hosted HTTP adapters only.

A request is believed to come from the proxy only when it arrives from a
trusted address *and* carries the proxy's shared token. The address alone is
not enough: anything else that can send from that network (another container,
a local program on a loopback deployment) could otherwise name any user.
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TENANCY_MODE = "proxy-header"
_IDENTITY_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
PROXY_TOKEN_HEADER = "X-TinyContext-Proxy-Token"


class HostedTenancyError(ValueError):
    """A hosted request could not be safely assigned to a tenant."""


@dataclass(frozen=True)
class HostedTenancyConfig:
    header_name: str
    store_dir: Path
    secret: bytes
    trusted_proxy_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
    proxy_token: bytes


_current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tinycontext_hosted_user_id", default=None
)


def hosted_tenancy_enabled() -> bool:
    mode = os.environ.get("TINYCONTEXT_TENANCY", "").strip()
    if not mode:
        return False
    if mode != _TENANCY_MODE:
        raise HostedTenancyError(
            "TINYCONTEXT_TENANCY must be 'proxy-header' when configured"
        )
    return True


def load_hosted_tenancy_config() -> HostedTenancyConfig:
    """Load the explicit configuration needed before accepting hosted tenants."""
    if not hosted_tenancy_enabled():
        raise HostedTenancyError("hosted tenancy is not enabled")
    header_name = os.environ.get(
        "TINYCONTEXT_TRUSTED_USER_HEADER", "X-TinyContext-User-Id"
    ).strip()
    store_dir_raw = os.environ.get("TINYCONTEXT_TENANT_STORE_DIR", "").strip()
    secret = os.environ.get("TINYCONTEXT_TENANT_SECRET", "").encode()
    networks_raw = os.environ.get("TINYCONTEXT_TRUSTED_PROXY_CIDRS", "").strip()
    proxy_token = os.environ.get("TINYCONTEXT_TRUSTED_PROXY_TOKEN", "").strip().encode()
    if not header_name or not re.fullmatch(r"[A-Za-z0-9-]+", header_name):
        raise HostedTenancyError("TINYCONTEXT_TRUSTED_USER_HEADER is invalid")
    if header_name.lower() == PROXY_TOKEN_HEADER.lower():
        raise HostedTenancyError(
            f"TINYCONTEXT_TRUSTED_USER_HEADER cannot be {PROXY_TOKEN_HEADER}"
        )
    if len(proxy_token) < 32:
        raise HostedTenancyError(
            "TINYCONTEXT_TRUSTED_PROXY_TOKEN must be at least 32 bytes: "
            f"the proxy sends it as {PROXY_TOKEN_HEADER} on every request"
        )
    if not store_dir_raw:
        raise HostedTenancyError("TINYCONTEXT_TENANT_STORE_DIR is required")
    if len(secret) < 32:
        raise HostedTenancyError(
            "TINYCONTEXT_TENANT_SECRET must be at least 32 bytes"
        )
    if not networks_raw:
        raise HostedTenancyError("TINYCONTEXT_TRUSTED_PROXY_CIDRS is required")
    try:
        networks = tuple(
            ipaddress.ip_network(value.strip(), strict=False)
            for value in networks_raw.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise HostedTenancyError("TINYCONTEXT_TRUSTED_PROXY_CIDRS is invalid") from exc
    if not networks:
        raise HostedTenancyError("TINYCONTEXT_TRUSTED_PROXY_CIDRS is required")
    return HostedTenancyConfig(
        header_name=header_name,
        store_dir=Path(store_dir_raw).expanduser().resolve(),
        secret=secret,
        trusted_proxy_networks=networks,
        proxy_token=proxy_token,
    )


def resolve_hosted_user(scope: Mapping[str, Any]) -> str:
    """Validate the direct proxy peer and return its injected user identifier."""
    config = load_hosted_tenancy_config()
    client = scope.get("client")
    client_host = client[0] if client else None
    try:
        client_ip = ipaddress.ip_address(client_host)
    except ValueError as exc:
        raise HostedTenancyError("request did not come from a trusted proxy") from exc
    if not any(client_ip in network for network in config.trusted_proxy_networks):
        raise HostedTenancyError("request did not come from a trusted proxy")

    def header(name: str) -> list[bytes]:
        key = name.lower().encode("ascii")
        return [bytes(v) for k, v in scope.get("headers", []) if bytes(k).lower() == key]

    # Same answer as a wrong address: a caller learns nothing about which check failed.
    tokens = header(PROXY_TOKEN_HEADER)
    if len(tokens) != 1 or not hmac.compare_digest(tokens[0].strip(), config.proxy_token):
        raise HostedTenancyError("request did not come from a trusted proxy")
    values = header(config.header_name)
    if len(values) != 1:
        raise HostedTenancyError("trusted user identity is missing or invalid")
    raw_user_id = values[0]
    try:
        user_id = raw_user_id.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise HostedTenancyError("trusted user identity is invalid") from exc
    if not _IDENTITY_RE.fullmatch(user_id):
        raise HostedTenancyError("trusted user identity is missing or invalid")
    return user_id


def bind_hosted_user(user_id: str) -> contextvars.Token[str | None]:
    return _current_user_id.set(user_id)


def reset_hosted_user(token: contextvars.Token[str | None]) -> None:
    _current_user_id.reset(token)


def mcp_message_context() -> tuple[bool, Mapping[str, Any] | None]:
    """(inside an MCP message, the ASGI scope of the HTTP request that carried it).

    An MCP session's handler task is created by the request that opened the
    session, so a contextvar bound by the ASGI middleware only ever reaches
    tool handlers with *that first request's* user. Each message's own HTTP
    request travels in the SDK's request context instead, which is what has
    to decide the tenant: otherwise a client that reuses one session for
    several users (a shared chat UI, a gateway) runs every user's calls in
    the store of whoever connected first. The scope is None for stdio.
    """
    try:
        from mcp.server.lowlevel.server import request_ctx
    except ImportError:
        return False, None
    context = request_ctx.get(None)
    if context is None:
        return False, None
    scope = getattr(getattr(context, "request", None), "scope", None)
    return True, scope if isinstance(scope, Mapping) else None


def mcp_message_scope() -> Mapping[str, Any] | None:
    return mcp_message_context()[1]


def http_transport() -> bool:
    """Whether the MCP server was started over HTTP (sse or streamable-http)."""
    return (os.environ.get("MCP_TRANSPORT", "stdio").strip() or "stdio") != "stdio"


def current_hosted_user() -> str | None:
    """The verified user for the current call, or None outside hosted mode.

    MCP messages are re-resolved from their own HTTP request (peer address
    and identity header, every message); FastAPI requests use the identity
    the middleware bound for that request. Hosted mode only runs over HTTP,
    so an MCP message with no request means the SDK stopped passing it:
    that is refused, never answered from the session-wide fallback.
    """
    if not hosted_tenancy_enabled():
        return None
    inside, scope = mcp_message_context()
    if inside:
        if scope is None:
            raise HostedTenancyError("MCP message carried no HTTP request; refusing to guess the tenant")
        return resolve_hosted_user(scope)
    user_id = _current_user_id.get()
    if user_id is None:
        raise HostedTenancyError("hosted request has no authenticated user")
    return user_id


def tenant_config(base_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the per-user configuration for an authenticated hosted request."""
    user_id = current_hosted_user()
    if user_id is None:
        return dict(base_config)
    config = load_hosted_tenancy_config()
    digest = hmac.new(config.secret, user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    resolved = dict(base_config)
    resolved["memory_db_path"] = str(config.store_dir / f"{digest}.db")
    return resolved
