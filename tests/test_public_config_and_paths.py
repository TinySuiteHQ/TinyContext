from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext import TinyContextConfig
from tinycontext.config import resolve_config
from tinycontext.paths import native_memory_db_path
from tinycontext.services.context_config_service import (
    load_context_config,
    resolve_context_config_path,
)


class PublicConfigTests(unittest.TestCase):
    def test_defaults_are_json_serializable_and_use_native_storage(self) -> None:
        config = TinyContextConfig()
        self.assertEqual(
            Path(config.memory_db_path),
            native_memory_db_path().resolve(),
        )
        self.assertEqual(config["recall_top_k"], 10)
        self.assertEqual(config["recall_rrf_cutoff"], 0.0)
        self.assertEqual(config["embedding_backend"], "onnx")
        self.assertEqual(config["embedding_openai_env_file"], ".env")
        json.dumps(config.to_dict())

    def test_explicit_file_resolves_database_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config" / "context.json"
            config_path.parent.mkdir()
            config_path.write_text(
                json.dumps(
                    {
                        "memory_db_path": "../data/memories.db",
                        "models_dir": "../models",
                        "recall_top_k": 5,
                    }
                ),
                encoding="utf-8",
            )
            config = TinyContextConfig.from_json(config_path)
            expected = Path(temp_dir) / "data" / "memories.db"
            expected_models = Path(temp_dir) / "models"
            self.assertEqual(Path(config.memory_db_path), expected.resolve())
            self.assertEqual(Path(config.models_dir), expected_models.resolve())
            self.assertEqual(config.recall_top_k, 5)

    def test_explicit_file_then_call_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "context.json"
            config_path.write_text(
                json.dumps({"recall_top_k": 5}),
                encoding="utf-8",
            )
            config = resolve_config({"recall_top_k": 3}, path=config_path)
            self.assertEqual(config.recall_top_k, 3)

    def test_programmatic_config_ignores_environment_and_working_directory(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TINYCONTEXT_CONFIG_PATH": "/does/not/matter.json",
                "TINYCONTEXT_MEMORY_DB_PATH": "/also/ignored.db",
            },
            clear=True,
        ), patch("os.getcwd", side_effect=AssertionError("cwd must not be read")):
            config = TinyContextConfig()
        self.assertEqual(
            Path(config.memory_db_path),
            native_memory_db_path().resolve(),
        )

    def test_rejects_unknown_and_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            TinyContextConfig.from_mapping({"unknown": True})
        with self.assertRaisesRegex(ValueError, "at least 1"):
            TinyContextConfig(recall_top_k=0)
        with self.assertRaisesRegex(ValueError, "encoding_name"):
            TinyContextConfig(encoding_name=" ")
        with self.assertRaisesRegex(ValueError, "recall_dense_weight"):
            TinyContextConfig(recall_dense_weight=1.1)
        with self.assertRaisesRegex(ValueError, "recall_rrf_cutoff"):
            TinyContextConfig(recall_rrf_cutoff=1.1)
        with self.assertRaisesRegex(ValueError, "embedding_backend"):
            TinyContextConfig(embedding_backend="not-a-real-backend")
        with self.assertRaisesRegex(ValueError, "embedding_openai_env_file"):
            TinyContextConfig(embedding_openai_env_file=" ")
        with self.assertRaisesRegex(ValueError, "dedup_similarity_threshold"):
            TinyContextConfig(dedup_similarity_threshold=1.1)
        with self.assertRaisesRegex(ValueError, "dedup_similarity_threshold"):
            TinyContextConfig(dedup_similarity_threshold=0.0)
        with self.assertRaisesRegex(ValueError, "recall_access_weight"):
            TinyContextConfig(recall_access_weight=1.1)
        with self.assertRaisesRegex(ValueError, "recall_access_weight"):
            TinyContextConfig(recall_access_weight=-0.1)

    def test_embedding_backend_normalizes_aliases(self) -> None:
        self.assertEqual(TinyContextConfig(embedding_backend="onnx").embedding_backend, "onnx")
        self.assertEqual(TinyContextConfig(embedding_backend="default").embedding_backend, "onnx")
        self.assertEqual(
            TinyContextConfig(embedding_backend="openai").embedding_backend,
            "openai_compatible",
        )
        self.assertEqual(
            TinyContextConfig(embedding_backend="openai_compatible").embedding_backend,
            "openai_compatible",
        )


class ServerConfigTests(unittest.TestCase):
    def test_missing_file_uses_native_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"TINYCONTEXT_CONFIG_PATH": str(Path(temp_dir) / "missing.json")},
            clear=True,
        ):
            config = load_context_config()
        self.assertEqual(
            Path(config["memory_db_path"]),
            native_memory_db_path().resolve(),
        )

    def test_environment_overrides_server_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "custom.db"
            with patch.dict(
                os.environ,
                {
                    "TINYCONTEXT_CONFIG_PATH": str(Path(temp_dir) / "missing.json"),
                    "TINYCONTEXT_MEMORY_DB_PATH": str(db_path),
                    "TINYCONTEXT_RECALL_TOP_K": "4",
                    "TINYCONTEXT_RECALL_MAX_TOKENS": "500",
                    "TINYCONTEXT_ENCODING_NAME": "cl100k_base",
                    "TINYCONTEXT_MODELS_DIR": str(Path(temp_dir) / "models"),
                    "TINYCONTEXT_EMBEDDING_MODEL": "balanced",
                    "TINYCONTEXT_EMBEDDING_BACKEND": "openai",
                    "TINYCONTEXT_EMBEDDING_OPENAI_ENV_FILE": "/custom/embeddings.env",
                    "TINYCONTEXT_RECALL_RRF_CUTOFF": "0.8",
                    "TINYCONTEXT_RECALL_DENSE_WEIGHT": "0.75",
                    "TINYCONTEXT_DENSE_QUERY_PREFIX": "query: ",
                    "TINYCONTEXT_DEDUP_SIMILARITY_THRESHOLD": "0.9",
                    "TINYCONTEXT_RECALL_ACCESS_WEIGHT": "0.2",
                },
                clear=True,
            ):
                config = load_context_config()
        self.assertEqual(Path(config["memory_db_path"]), db_path.resolve())
        self.assertEqual(config["recall_top_k"], 4)
        self.assertEqual(config["recall_max_tokens"], 500)
        self.assertEqual(config["encoding_name"], "cl100k_base")
        self.assertEqual(
            os.path.normcase(str(Path(config["models_dir"]))),
            os.path.normcase(str(Path(temp_dir) / "models")),
        )
        self.assertEqual(config["embedding_model"], "balanced")
        self.assertEqual(config["embedding_backend"], "openai_compatible")
        self.assertEqual(config["embedding_openai_env_file"], "/custom/embeddings.env")
        self.assertEqual(config["recall_rrf_cutoff"], 0.8)
        self.assertEqual(config["recall_dense_weight"], 0.75)
        self.assertEqual(config["dense_query_prefix"], "query: ")
        self.assertEqual(config["dedup_similarity_threshold"], 0.9)
        self.assertEqual(config["recall_access_weight"], 0.2)

    def test_explicit_config_path_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"TINYCONTEXT_CONFIG_PATH": "/ignored.json"},
            clear=True,
        ):
            explicit = Path(temp_dir) / "context.json"
            self.assertEqual(resolve_context_config_path(explicit), explicit.resolve())


if __name__ == "__main__":
    unittest.main()
