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
        json.dumps(config.to_dict())

    def test_explicit_file_resolves_database_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config" / "context.json"
            config_path.parent.mkdir()
            config_path.write_text(
                json.dumps(
                    {
                        "memory_db_path": "../data/memories.db",
                        "recall_top_k": 5,
                    }
                ),
                encoding="utf-8",
            )
            config = TinyContextConfig.from_json(config_path)
            expected = Path(temp_dir) / "data" / "memories.db"
            self.assertEqual(Path(config.memory_db_path), expected.resolve())
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
                },
                clear=True,
            ):
                config = load_context_config()
        self.assertEqual(Path(config["memory_db_path"]), db_path.resolve())
        self.assertEqual(config["recall_top_k"], 4)
        self.assertEqual(config["recall_max_tokens"], 500)
        self.assertEqual(config["encoding_name"], "cl100k_base")

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
