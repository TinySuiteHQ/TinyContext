"""ASGI middleware that binds trusted proxy identity to hosted requests."""

from __future__ import annotations

import json
from typing import Any

from tinycontext.services.hosted_tenancy_service import (
    HostedTenancyError,
    bind_hosted_user,
    hosted_tenancy_enabled,
    reset_hosted_user,
    resolve_hosted_user,
)


class HostedTenancyMiddleware:
    """Reject untrusted hosted requests before adapter code can access storage."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope["type"] != "http"
            or scope.get("path") == "/health"
            or not hosted_tenancy_enabled()
        ):
            await self.app(scope, receive, send)
            return
        try:
            user_id = resolve_hosted_user(scope)
        except HostedTenancyError as exc:
            body = json.dumps(
                {"detail": {"code": "unauthorized", "message": str(exc)}}
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        token = bind_hosted_user(user_id)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_hosted_user(token)
