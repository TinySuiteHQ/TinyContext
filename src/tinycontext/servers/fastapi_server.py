from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tinycontext import core
from tinycontext.errors import MEMORY_ERROR_MAP, MemoryError
from tinycontext.models import MemoryInput
from tinycontext.services.context_config_service import load_context_config


def _tinycontext_version() -> str:
    return os.environ.get("TINYCONTEXT_VERSION", "dev").strip() or "dev"


app = FastAPI(
    title="TinyContext API",
    description="Token-light hybrid memory save and recall endpoints for agents.",
    version=_tinycontext_version(),
)


@app.on_event("startup")
async def _prepare_embedding_model() -> None:
    from tinycontext.services.onnx_bundle_service import ensure_onnx_bundle_sync

    config = load_context_config()
    await asyncio.to_thread(
        ensure_onnx_bundle_sync,
        str(config["embedding_model"]),
        models_dir=str(config["models_dir"]),
    )
    notice = await asyncio.to_thread(core.start_background_reembed_if_needed, config)
    if notice:
        print(f"[tinycontext] {notice}", file=sys.stderr, flush=True)


class MemoryInputModel(BaseModel):
    content: str = Field(..., min_length=1)


class SaveMemoriesRequest(BaseModel):
    session_id: str | None = None
    memories: list[MemoryInputModel] = Field(..., min_length=1)


class RecallMemoriesRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    top_k: int | None = Field(default=None, ge=1)


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
    return [MemoryInput(content=item.content) for item in items]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/save_memories")
async def save_memories_endpoint(request: SaveMemoriesRequest) -> dict[str, Any]:
    config = load_context_config()
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
    config = load_context_config()
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
    query: str,
    session_id: str | None = None,
    max_tokens: int | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    return await recall_memories_endpoint(
        RecallMemoriesRequest(
            query=query,
            session_id=session_id,
            max_tokens=max_tokens,
            top_k=top_k,
        )
    )
