from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext.services.embedding_service import (
    _load_openai_client,
    _parse_openai_env_file,
    _resolve_openai_env_path,
    embed_texts,
    embedding_model_key,
    normalize_embedding_backend,
    onnx_bundle_ready,
    resolve_embedding_model_display,
    resolve_local_embedding_model_spec,
)
from tinycontext.services.onnx_bundle_service import ensure_onnx_bundle_sync


class EmbeddingServiceTests(unittest.TestCase):
    def test_fast_preset_matches_tinysearch_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = resolve_local_embedding_model_spec(
                "fast",
                models_dir=temp_dir,
            )
        self.assertEqual(spec.repo_id, "onnx-models/all-MiniLM-L6-v2-onnx")
        self.assertEqual(spec.onnx_paths, ("model.onnx",))

    def test_model_key_changes_with_document_prefix(self) -> None:
        plain = embedding_model_key("fast", document_prefix="")
        prefixed = embedding_model_key("fast", document_prefix="passage: ")
        self.assertNotEqual(plain, prefixed)

    def test_missing_bundle_is_downloaded_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)

            def fake_snapshot_download(**kwargs: object) -> None:
                destination = Path(str(kwargs["local_dir"]))
                (destination / "model.onnx").write_bytes(b"test-model")
                (destination / "tokenizer.json").write_text(
                    "{}",
                    encoding="utf-8",
                )

            with patch(
                "tinycontext.services.onnx_bundle_service.snapshot_download",
                side_effect=fake_snapshot_download,
            ) as download:
                ensure_onnx_bundle_sync("fast", models_dir=models_dir)
                ensure_onnx_bundle_sync("fast", models_dir=models_dir)

            self.assertEqual(download.call_count, 1)
            self.assertTrue(onnx_bundle_ready("fast", models_dir=models_dir))


class OpenAICompatibleBackendTests(unittest.TestCase):
    def test_normalize_embedding_backend_aliases(self) -> None:
        self.assertEqual(normalize_embedding_backend(None), "onnx")
        self.assertEqual(normalize_embedding_backend("onnx"), "onnx")
        self.assertEqual(normalize_embedding_backend("default"), "onnx")
        self.assertEqual(normalize_embedding_backend("local"), "onnx")
        self.assertEqual(normalize_embedding_backend("openai"), "openai_compatible")
        self.assertEqual(normalize_embedding_backend("openai_compatible"), "openai_compatible")

    def test_normalize_embedding_backend_rejects_unknown_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown embedding_backend"):
            normalize_embedding_backend("llama_cpp")

    def test_parse_openai_env_file_reads_values_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "embeddings.env"
            env_path.write_text(
                "OPENAI_BASE_URL=http://gateway:8080/internal/embeddings/v1\n"
                "API_KEY=secret-value\n"
                "EMBEDDING_MODEL=jina-v5-small-retrieval\n",
                encoding="utf-8",
            )
            base_url, api_key, model = _parse_openai_env_file(env_path)
        self.assertEqual(base_url, "http://gateway:8080/internal/embeddings/v1")
        self.assertEqual(api_key, "secret-value")
        self.assertEqual(model, "jina-v5-small-retrieval")

    def test_parse_openai_env_file_requires_api_key_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "embeddings.env"
            env_path.write_text("OPENAI_BASE_URL=http://example.test\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                _parse_openai_env_file(env_path)

            env_path.write_text("OPENAI_API_KEY=secret-value\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "OPENAI_EMBEDDING_MODEL"):
                _parse_openai_env_file(env_path)

    def test_parse_openai_env_file_missing_file_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "openai_compatible backend requires"):
            _parse_openai_env_file(Path("/does/not/exist.env"))

    def test_resolve_openai_env_path_relative_to_project_root(self) -> None:
        resolved = _resolve_openai_env_path("some/relative.env")
        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved.name, "relative.env")
        self.assertEqual(resolved.parent.name, "some")

    def test_embedding_model_key_differs_by_backend_even_with_same_model_string(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "embeddings.env"
            env_path.write_text(
                "OPENAI_API_KEY=secret-value\nOPENAI_EMBEDDING_MODEL=fast\n",
                encoding="utf-8",
            )
            onnx_key = embedding_model_key("fast", backend="onnx")
            openai_key = embedding_model_key(
                "fast",
                backend="openai_compatible",
                openai_env_file=env_path,
            )
        self.assertNotEqual(onnx_key, openai_key)
        self.assertTrue(onnx_key.startswith("onnx:"))
        self.assertTrue(openai_key.startswith("openai_compatible:"))

    def test_embed_texts_openai_compatible_round_trips_through_fake_client(self) -> None:
        _load_openai_client.cache_clear()
        self.addCleanup(_load_openai_client.cache_clear)

        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "embeddings.env"
            env_path.write_text(
                "OPENAI_BASE_URL=http://gateway.local/internal/embeddings/v1\n"
                "OPENAI_API_KEY=secret-value\n"
                "OPENAI_EMBEDDING_MODEL=jina-v5-small-retrieval\n",
                encoding="utf-8",
            )

            captured: dict[str, object] = {}

            class _FakeItem:
                def __init__(self, embedding: list[float]) -> None:
                    self.embedding = embedding

            class _FakeResponse:
                def __init__(self, vectors: list[list[float]]) -> None:
                    self.data = [_FakeItem(vector) for vector in vectors]

            class _FakeEmbeddings:
                def create(self, *, model: str, input: list[str]):  # noqa: A002
                    captured["model"] = model
                    captured["input"] = input
                    return _FakeResponse([[0.1, 0.2, 0.3] for _ in input])

            class _FakeClient:
                def __init__(self) -> None:
                    self.embeddings = _FakeEmbeddings()

            with patch(
                "tinycontext.services.embedding_service._load_openai_client",
                return_value=_FakeClient(),
            ):
                vectors = embed_texts(
                    ["hello", "world"],
                    backend="openai_compatible",
                    openai_env_file=env_path,
                )

        self.assertEqual(vectors, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        self.assertEqual(captured["model"], "jina-v5-small-retrieval")
        self.assertEqual(captured["input"], ["hello", "world"])

    def test_resolve_embedding_model_display_covers_both_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_display = resolve_embedding_model_display("onnx", "fast", models_dir=temp_dir)
            self.assertIn("local ONNX", onnx_display)

            env_path = Path(temp_dir) / "embeddings.env"
            env_path.write_text(
                "OPENAI_API_KEY=secret-value\nOPENAI_EMBEDDING_MODEL=jina-v5-small-retrieval\n",
                encoding="utf-8",
            )
            openai_display = resolve_embedding_model_display(
                "openai_compatible",
                openai_env_file=env_path,
            )
        self.assertIn("jina-v5-small-retrieval", openai_display)
        self.assertIn("openai_compatible", openai_display)


if __name__ == "__main__":
    unittest.main()
