from __future__ import annotations

import os
import re
import threading
from hashlib import sha256
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from tinycontext.paths import native_data_dir, native_models_dir


DEFAULT_EMBEDDING_MODEL = "fast"
DEFAULT_EMBEDDING_BACKEND = "onnx"
DEFAULT_EMBEDDING_OPENAI_ENV_FILE = ".env"
SUPPORTED_EMBEDDING_BACKENDS = (
    "onnx",
    "openai_compatible",
)
_EMBED_LOCK = threading.Lock()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_COMMON_ONNX_ALLOW_PATTERNS = (
    "model.onnx",
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "tokenizer.model",
    "config.json",
)

_PRESET_MODELS: dict[str, dict[str, Any]] = {
    "fast": {
        "repo_id": "onnx-models/all-MiniLM-L6-v2-onnx",
        "local_dir": "all-minilm-l6-v2-onnx",
        "onnx_paths": ("model.onnx",),
        "pooling": "auto",
        "normalize": False,
        "max_length": 256,
        "allow_patterns": (
            "model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ),
    },
    "balanced": {
        "repo_id": "BAAI/bge-small-en-v1.5",
        "local_dir": "bge-small-en-v1.5-onnx",
        "onnx_paths": ("onnx/model.onnx", "model.onnx"),
        "pooling": "cls",
        "normalize": True,
        "max_length": 512,
        "allow_patterns": _COMMON_ONNX_ALLOW_PATTERNS,
    },
    "quality": {
        "repo_id": "BAAI/bge-base-en-v1.5",
        "local_dir": "bge-base-en-v1.5-onnx",
        "onnx_paths": ("onnx/model.onnx", "model.onnx"),
        "pooling": "cls",
        "normalize": True,
        "max_length": 512,
        "allow_patterns": _COMMON_ONNX_ALLOW_PATTERNS,
    },
}


@dataclass(frozen=True)
class LocalEmbeddingModelSpec:
    requested_model: str
    repo_id: str
    local_dir: Path
    onnx_paths: tuple[str, ...]
    pooling: str
    normalize: bool
    max_length: int
    allow_patterns: tuple[str, ...]
    is_preset: bool


@dataclass(frozen=True)
class _LoadedOnnxBundle:
    session: Any
    tokenizer: Any
    spec: LocalEmbeddingModelSpec
    model_path: Path


def _safe_model_dir_name(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name.strip()).strip("-._")
    return f"{slug.lower() or 'custom-embedding-model'}-onnx"


def _resolve_models_dir(models_dir: str | Path | None) -> Path:
    path = Path(models_dir).expanduser() if models_dir else native_models_dir()
    if not path.is_absolute():
        path = native_data_dir() / path
    return Path(os.path.normpath(str(path)))


def resolve_local_embedding_model_spec(
    embedding_model: str | None = None,
    *,
    models_dir: str | Path | None = None,
) -> LocalEmbeddingModelSpec:
    requested = (
        (embedding_model or DEFAULT_EMBEDDING_MODEL).strip()
        or DEFAULT_EMBEDDING_MODEL
    )
    preset = _PRESET_MODELS.get(requested.lower())
    root = _resolve_models_dir(models_dir)
    if preset is not None:
        return LocalEmbeddingModelSpec(
            requested_model=requested,
            repo_id=str(preset["repo_id"]),
            local_dir=root / str(preset["local_dir"]),
            onnx_paths=tuple(preset["onnx_paths"]),
            pooling=str(preset["pooling"]),
            normalize=bool(preset["normalize"]),
            max_length=int(preset["max_length"]),
            allow_patterns=tuple(preset["allow_patterns"]),
            is_preset=True,
        )
    return LocalEmbeddingModelSpec(
        requested_model=requested,
        repo_id=requested,
        local_dir=root / _safe_model_dir_name(requested),
        onnx_paths=("model.onnx", "onnx/model.onnx"),
        pooling="auto",
        normalize=False,
        max_length=512,
        allow_patterns=_COMMON_ONNX_ALLOW_PATTERNS,
        is_preset=False,
    )


def normalize_embedding_backend(backend: str | None) -> str:
    key = (backend or DEFAULT_EMBEDDING_BACKEND).strip().lower()
    if key in ("onnx", "default", "local"):
        return "onnx"
    if key in ("openai_compatible", "openai"):
        return "openai_compatible"
    raise ValueError(
        f"unknown embedding_backend {backend!r}; expected one of "
        f"{SUPPORTED_EMBEDDING_BACKENDS} (aliases: default, local -> onnx; openai -> openai_compatible)"
    )


def _resolve_openai_env_path(openai_env_file: str | Path | None) -> Path:
    raw = str(openai_env_file or DEFAULT_EMBEDDING_OPENAI_ENV_FILE).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def _parse_openai_env_file(path: Path) -> tuple[str | None, str, str]:
    """Read base URL, API key, and embedding model name from a .env-style file."""
    if not path.is_file():
        raise RuntimeError(
            f"openai_compatible backend requires {path} with OPENAI_BASE_URL (optional), "
            "OPENAI_API_KEY, and OPENAI_EMBEDDING_MODEL (or EMBEDDING_MODEL)"
        )
    text = path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        name, _, rest = stripped.partition("=")
        key = name.strip().upper()
        val = rest.strip().strip('"').strip("'")
        if key:
            values[key] = val

    def pick(*keys: str) -> str | None:
        for k in keys:
            v = values.get(k.upper())
            if v:
                return v
        return None

    api_key = pick("OPENAI_API_KEY", "API_KEY")
    if not api_key:
        raise RuntimeError(
            f"{path} must set OPENAI_API_KEY (or API_KEY) for openai_compatible embeddings"
        )
    base_raw = pick("OPENAI_BASE_URL", "BASE_URL", "API_URL")
    base_url = base_raw.strip() if base_raw else None
    if base_url == "":
        base_url = None
    model = pick(
        "OPENAI_EMBEDDING_MODEL",
        "EMBEDDING_MODEL",
        "MODEL_NAME",
        "MODEL",
    )
    if not model:
        raise RuntimeError(
            f"{path} must set OPENAI_EMBEDDING_MODEL (or EMBEDDING_MODEL / MODEL_NAME) "
            "for openai_compatible embeddings"
        )
    return base_url, api_key, model


@lru_cache(maxsize=8)
def _load_openai_client(base_url: str | None, api_key: str) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "The openai_compatible backend requires the `openai` package. "
            "Install with: pip install openai"
        ) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return OpenAI(**kwargs)
    except Exception as exc:
        raise RuntimeError("failed to construct OpenAI-compatible client") from exc


def _embed_openai_compatible_sync(
    client: Any,
    model_name: str,
    inputs: list[str],
    *,
    base_url: str | None,
) -> list[list[float]]:
    if not inputs:
        return []
    try:
        response = client.embeddings.create(model=model_name, input=inputs)
    except Exception as exc:
        raise RuntimeError(
            f"failed to generate embeddings with openai_compatible model {model_name!r} "
            f"(base_url={base_url!r})"
        ) from exc
    return [list(item.embedding) for item in response.data]


def resolve_embedding_model_display(
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    embedding_model: str | None = None,
    *,
    models_dir: str | Path | None = None,
    openai_env_file: str | Path | None = None,
) -> str:
    """Human-readable "what model is actually in use" string, for doctor.py."""
    backend_key = normalize_embedding_backend(backend)
    if backend_key == "onnx":
        spec = resolve_local_embedding_model_spec(embedding_model, models_dir=models_dir)
        return f"{spec.requested_model} ({spec.repo_id}, local ONNX)"
    base_url, _, model_name = _parse_openai_env_file(_resolve_openai_env_path(openai_env_file))
    return f"{model_name} (openai_compatible, base_url={base_url or 'api.openai.com'})"


def embedding_model_key(
    embedding_model: str | None = None,
    *,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    models_dir: str | Path | None = None,
    openai_env_file: str | Path | None = None,
    document_prefix: str = "",
) -> str:
    backend_key = normalize_embedding_backend(backend)
    if backend_key == "onnx":
        spec = resolve_local_embedding_model_spec(
            embedding_model,
            models_dir=models_dir,
        )
        model_identifier = spec.repo_id
    else:
        _, _, model_identifier = _parse_openai_env_file(_resolve_openai_env_path(openai_env_file))
    prefix_digest = sha256(document_prefix.encode("utf-8")).hexdigest()[:12]
    return f"{backend_key}:{model_identifier}:document-prefix:{prefix_digest}"


def _find_onnx_model_path(spec: LocalEmbeddingModelSpec) -> Path | None:
    for relative_path in spec.onnx_paths:
        path = spec.local_dir / relative_path
        if path.is_file():
            return path
    for path in sorted(spec.local_dir.rglob("*.onnx")):
        return path
    return None


def _tokenizer_ready(bundle_dir: Path) -> bool:
    return (bundle_dir / "tokenizer.json").is_file()


def onnx_bundle_ready(
    embedding_model: str | None = None,
    *,
    models_dir: str | Path | None = None,
) -> bool:
    spec = resolve_local_embedding_model_spec(
        embedding_model,
        models_dir=models_dir,
    )
    return (
        _find_onnx_model_path(spec) is not None
        and _tokenizer_ready(spec.local_dir)
    )


@lru_cache(maxsize=8)
def _load_onnx_runtime_bundle_cached(
    embedding_model: str,
    models_dir: str,
) -> _LoadedOnnxBundle:
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require `onnxruntime` and `tokenizers`."
        ) from exc

    spec = resolve_local_embedding_model_spec(
        embedding_model,
        models_dir=models_dir,
    )
    model_path = _find_onnx_model_path(spec)
    if model_path is None or not _tokenizer_ready(spec.local_dir):
        raise RuntimeError(
            f"ONNX embedding bundle for {spec.requested_model!r} is incomplete "
            f"under {spec.local_dir}; TinyContext normally downloads it on first use."
        )
    tokenizer = Tokenizer.from_file(str(spec.local_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=spec.max_length)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return _LoadedOnnxBundle(
        session=session,
        tokenizer=tokenizer,
        spec=spec,
        model_path=model_path,
    )


def clear_onnx_runtime_cache() -> None:
    _load_onnx_runtime_bundle_cached.cache_clear()


def _normalize_rows(value: Any) -> Any:
    import numpy as np

    rows = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def _pick_token_output(outputs: list[Any]) -> Any | None:
    for output in outputs:
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape) == 3:
            return output
    return None


def _pick_pooled_output(outputs: list[Any]) -> Any | None:
    for output in outputs:
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape) == 2:
            return output
    return None


def _pool_onnx_outputs(outputs: list[Any], spec: LocalEmbeddingModelSpec) -> Any:
    if spec.pooling == "cls":
        token_output = _pick_token_output(outputs)
        if token_output is None:
            raise RuntimeError(
                f"ONNX model {spec.repo_id!r} does not expose a token output "
                "required for CLS pooling"
            )
        pooled = token_output[:, 0]
    else:
        pooled = _pick_pooled_output(outputs)
        if pooled is None:
            token_output = _pick_token_output(outputs)
            if token_output is None:
                raise RuntimeError(
                    f"ONNX model {spec.repo_id!r} has unsupported outputs"
                )
            pooled = token_output[:, 0]
    return _normalize_rows(pooled) if spec.normalize else pooled


def _as_vectors(value: Any) -> list[list[float]]:
    if hasattr(value, "reshape") and getattr(value, "ndim", None) == 1:
        value = value.reshape(1, -1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return []
    if isinstance(value[0], (int, float)):
        return [[float(item) for item in value]]
    return [[float(item) for item in vector] for vector in value]


def embed_texts(
    inputs: Sequence[str],
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    models_dir: str | Path | None = None,
    openai_env_file: str | Path | None = None,
    batch_size: int = 32,
) -> list[list[float]]:
    texts = list(inputs)
    if not texts:
        return []

    if normalize_embedding_backend(backend) == "openai_compatible":
        env_path = _resolve_openai_env_path(openai_env_file)
        base_url, api_key, model_name = _parse_openai_env_file(env_path)
        client = _load_openai_client(base_url, api_key)
        return _embed_openai_compatible_sync(client, model_name, texts, base_url=base_url)

    return _embed_texts_onnx(texts, embedding_model=embedding_model, models_dir=models_dir, batch_size=batch_size)


def _embed_texts_onnx(
    texts: list[str],
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    models_dir: str | Path | None = None,
    batch_size: int = 32,
) -> list[list[float]]:
    import numpy as np

    from tinycontext.services.onnx_bundle_service import ensure_onnx_bundle_sync

    ensure_onnx_bundle_sync(
        embedding_model,
        models_dir=models_dir,
    )
    loaded = _load_onnx_runtime_bundle_cached(
        embedding_model,
        str(_resolve_models_dir(models_dir)),
    )
    batch_size = max(1, int(batch_size))
    all_rows: list[list[float]] = []
    with _EMBED_LOCK:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = loaded.tokenizer.encode_batch(batch)
            max_length = max((len(item.ids) for item in encoded), default=0)
            input_ids = np.asarray(
                [
                    item.ids + [0] * (max_length - len(item.ids))
                    for item in encoded
                ],
                dtype=np.int64,
            )
            attention_mask = np.asarray(
                [
                    item.attention_mask
                    + [0] * (max_length - len(item.attention_mask))
                    for item in encoded
                ],
                dtype=np.int64,
            )
            input_names = {item.name for item in loaded.session.get_inputs()}
            ort_inputs: dict[str, Any] = {}
            if "input_ids" in input_names:
                ort_inputs["input_ids"] = input_ids
            if "attention_mask" in input_names:
                ort_inputs["attention_mask"] = attention_mask
            if "token_type_ids" in input_names:
                ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

            output_names = [item.name for item in loaded.session.get_outputs()]
            if "sentence_embedding" in output_names:
                output = loaded.session.run(
                    ("sentence_embedding",),
                    ort_inputs,
                )[0]
                if loaded.spec.normalize:
                    output = _normalize_rows(output)
            else:
                output = _pool_onnx_outputs(
                    loaded.session.run(None, ort_inputs),
                    loaded.spec,
                )
            all_rows.extend(_as_vectors(output))
    if len(all_rows) != len(texts):
        raise RuntimeError(
            f"embedding model returned {len(all_rows)} vectors for {len(texts)} inputs"
        )
    return all_rows
