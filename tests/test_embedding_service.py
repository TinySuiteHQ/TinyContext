from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext.services.embedding_service import (
    embedding_model_key,
    onnx_bundle_ready,
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


if __name__ == "__main__":
    unittest.main()
