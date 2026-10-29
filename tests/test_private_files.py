"""Memory stores are owner-only on systems with file modes."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from tinycontext.paths import private_file

POSIX = os.name != "nt"


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class PrivateFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)

    def test_creates_the_file_and_its_folder(self) -> None:
        db = self.root / "tenants" / "store.db"
        private_file(db)
        private_file(db)  # idempotent
        self.assertTrue(db.is_file())

    @unittest.skipUnless(POSIX, "file modes are POSIX; AppData is already per-user on Windows")
    def test_new_stores_and_folders_are_owner_only(self) -> None:
        db = self.root / "tenants" / "store.db"
        private_file(db)
        self.assertEqual(mode(db), 0o600)
        self.assertEqual(mode(db.parent), 0o700)

    @unittest.skipUnless(POSIX, "file modes are POSIX; AppData is already per-user on Windows")
    def test_an_existing_readable_store_is_tightened_but_a_chosen_folder_is_left(self) -> None:
        folder = self.root / "shared"
        folder.mkdir(mode=0o755)
        os.chmod(folder, 0o755)
        db = folder / "memories.db"
        for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
            path.write_bytes(b"")
            os.chmod(path, 0o644)
        private_file(db)
        self.assertEqual([mode(p) for p in (db, Path(f"{db}-wal"), Path(f"{db}-shm"))], [0o600] * 3)
        self.assertEqual(mode(folder), 0o755)


if __name__ == "__main__":
    unittest.main()
