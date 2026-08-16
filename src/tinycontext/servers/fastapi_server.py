from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from tinycontext import core
from tinycontext.errors import MEMORY_ERROR_MAP, MemoryError
from tinycontext.models import MemoryInput
from tinycontext.services.context_config_service import load_context_config
from tinycontext.services.hosted_tenancy_service import (
    hosted_tenancy_enabled,
    load_hosted_tenancy_config,
    tenant_config,
)
from tinycontext.servers.hosted_tenancy_middleware import HostedTenancyMiddleware


def _tinycontext_version() -> str:
    return os.environ.get("TINYCONTEXT_VERSION", "dev").strip() or "dev"


app = FastAPI(
    title="TinyContext API",
    description="Token-light hybrid memory save and recall endpoints for agents.",
    version=_tinycontext_version(),
)
app.add_middleware(HostedTenancyMiddleware)


@app.on_event("startup")
async def _prepare_embedding_model() -> None:
    from tinycontext.services.embedding_service import normalize_embedding_backend
    from tinycontext.services.onnx_bundle_service import ensure_onnx_bundle_sync

    if hosted_tenancy_enabled():
        load_hosted_tenancy_config()
    config = load_context_config()
    if normalize_embedding_backend(str(config["embedding_backend"])) == "onnx":
        await asyncio.to_thread(
            ensure_onnx_bundle_sync,
            str(config["embedding_model"]),
            models_dir=str(config["models_dir"]),
        )
    # Hosted mode has one database per user, so reindex work starts on the
    # first authenticated request rather than a nonexistent global database.
    notice = None if hosted_tenancy_enabled() else await asyncio.to_thread(
        core.start_background_reembed_if_needed, config
    )
    if notice:
        print(f"[tinycontext] {notice}", file=sys.stderr, flush=True)


class MemoryInputModel(BaseModel):
    content: str = Field(..., min_length=1)
    kind: str = Field(default="episodic")


class SaveMemoriesRequest(BaseModel):
    session_id: str | None = None
    memories: list[MemoryInputModel] = Field(..., min_length=1)


class RecallMemoriesRequest(BaseModel):
    query: str | None = None
    session_id: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    top_k: int | None = Field(default=None, ge=1)


class DeleteMemoryRequest(BaseModel):
    memory_id: str = Field(..., min_length=1)


class UpdateMemoryRequest(BaseModel):
    memory_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


def _raise_memory_http_error(exc: Exception) -> None:
    mapping = MEMORY_ERROR_MAP.get(type(exc))
    if mapping is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "internal_error", "message": "internal error"},
        ) from exc
    code, status_code = mapping
    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": str(exc)},
    ) from exc


def _to_memory_inputs(items: list[MemoryInputModel]) -> list[MemoryInput]:
    return [MemoryInput(content=item.content, kind=item.kind) for item in items]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/save_memories")
async def save_memories_endpoint(request: SaveMemoriesRequest) -> dict[str, Any]:
    config = tenant_config(load_context_config())
    try:
        return core.save_memories(
            _to_memory_inputs(request.memories),
            session_id=request.session_id,
            config=config,
        )
    except MemoryError as exc:
        _raise_memory_http_error(exc)


@app.get("/save_memories")
async def save_memories_get(
    content: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    return await save_memories_endpoint(
        SaveMemoriesRequest(
            session_id=session_id,
            memories=[MemoryInputModel(content=content)],
        )
    )


@app.post("/recall_memories")
async def recall_memories_endpoint(request: RecallMemoriesRequest) -> dict[str, Any]:
    config = tenant_config(load_context_config())
    try:
        return core.recall_memories(
            request.query,
            session_id=request.session_id,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            config=config,
        )
    except MemoryError as exc:
        _raise_memory_http_error(exc)


@app.get("/recall_memories")
async def recall_memories_get(
    query: str | None = None,
    session_id: str | None = None,
    max_tokens: int | None = Query(default=None, ge=1),
    top_k: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    return await recall_memories_endpoint(
        RecallMemoriesRequest(
            query=query,
            session_id=session_id,
            max_tokens=max_tokens,
            top_k=top_k,
        )
    )


@app.post("/delete_memory")
async def delete_memory_endpoint(request: DeleteMemoryRequest) -> dict[str, Any]:
    config = tenant_config(load_context_config())
    try:
        return core.delete_memory(request.memory_id, config=config)
    except MemoryError as exc:
        _raise_memory_http_error(exc)


@app.post("/update_memory")
async def update_memory_endpoint(request: UpdateMemoryRequest) -> dict[str, Any]:
    config = tenant_config(load_context_config())
    try:
        return core.update_memory(request.memory_id, request.content, config=config)
    except MemoryError as exc:
        _raise_memory_http_error(exc)
