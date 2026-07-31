from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinycontext import doctor


class DoctorTests(unittest.TestCase):
    def test_doctor_passes_with_writable_database_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "context.json"
            config = {
                "memory_db_path": str(Path(temp_dir) / "data" / "memories.db"),
                "recall_top_k": 10,
                "recall_max_tokens": 2000,
                "encoding_name": "o200k_base",
            }
            with patch.object(
                doctor,
                "resolve_context_config_path",
                return_value=config_path,
            ), patch.object(doctor, "load_context_config", return_value=config):
                self.assertEqual(doctor.run(), 0)
            self.assertFalse(Path(config["memory_db_path"]).exists())

    def test_doctor_reports_invalid_config(self) -> None:
        with patch.object(
            doctor,
            "resolve_context_config_path",
            side_effect=ValueError("bad config"),
        ):
            self.assertEqual(doctor.run(), 1)


if __name__ == "__main__":
    unittest.main()
