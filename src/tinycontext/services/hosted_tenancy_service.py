"""Trusted-proxy tenant isolation for hosted HTTP adapters only."""

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


class HostedTenancyError(ValueError):
    """A hosted request could not be safely assigned to a tenant."""


@dataclass(frozen=True)
class HostedTenancyConfig:
    header_name: str
    store_dir: Path
    secret: bytes
    trusted_proxy_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]


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
    if not header_name or not re.fullmatch(r"[A-Za-z0-9-]+", header_name):
        raise HostedTenancyError("TINYCONTEXT_TRUSTED_USER_HEADER is invalid")
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
    header_key = config.header_name.lower().encode("ascii")
    values = [
        bytes(value)
        for key, value in scope.get("headers", [])
        if bytes(key).lower() == header_key
    ]
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


def tenant_config(base_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the per-user configuration for an authenticated hosted request."""
    user_id = _current_user_id.get()
    if user_id is None:
        if hosted_tenancy_enabled():
            raise HostedTenancyError("hosted request has no authenticated user")
        return dict(base_config)
    config = load_hosted_tenancy_config()
    digest = hmac.new(config.secret, user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    resolved = dict(base_config)
    resolved["memory_db_path"] = str(config.store_dir / f"{digest}.db")
    return resolved
