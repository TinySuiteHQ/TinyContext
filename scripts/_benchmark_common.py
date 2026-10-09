"""Shared helpers for the LoCoMo / LongMemEval retrieval benchmarks."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from tinycontext import core
from tinycontext.models import MemoryInput

DATASET_CACHE_DIR = _PROJECT_ROOT / ".cache" / "benchmarks"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset(url: str, dest: Path, *, sha256: str) -> Path:
    """Download ``url`` to ``dest`` once and verify it against a pinned SHA-256.

    The hash pins the exact file the published numbers were produced from; a
    mismatch means upstream changed (or the download was truncated), so the
    results would no longer be comparable.
    """
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[benchmark] downloading {url} -> {dest}", flush=True)
        partial = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(dest)
    actual = sha256_of(dest)
    if actual != sha256:
        raise SystemExit(
            f"{dest} has SHA-256 {actual}, expected {sha256}. Delete it and retry, or "
            "update the pinned hash if upstream intentionally changed the dataset."
        )
    return dest


@dataclass
class Ingested:
    """Maps stored memory ids back to the dataset units they represent."""

    units_by_memory: dict[str, set[str]] = field(default_factory=dict)
    ref_to_memory: dict[str, str] = field(default_factory=dict)

    def add(self, memory_id: str, ref: str, unit: str) -> None:
        self.units_by_memory.setdefault(memory_id, set()).add(unit)
        self.ref_to_memory[ref] = memory_id


def ingest(
    items: Iterable[tuple[str, str]], *, config: dict[str, Any], session_id: str
) -> Ingested:
    """Save ``(unit_id, text)`` pairs one at a time with the store's default dedup.

    One call per item keeps the mapping exact: when dedup skips an item as a
    near-duplicate, that item's unit is attributed to the memory it duplicates,
    so a recall of that memory still counts as retrieving the evidence.
    """
    result = Ingested()
    for unit, text in items:
        saved = core.save_memories(
            [MemoryInput(content=text)], session_id=session_id, config=config
        )
        if saved["saved"]:
            item = saved["saved"][0]
            result.add(item["id"], item["ref"], unit)
            continue
        ref = saved["skipped_duplicates"][0]["duplicate_of"]
        if ref in result.ref_to_memory:
            result.units_by_memory[result.ref_to_memory[ref]].add(unit)
    return result


def recalled_units(
    ingested: Ingested, query: str, *, config: dict[str, Any], session_id: str,
    top_k: int | None, max_tokens: int | None,
) -> tuple[list[set[str]], int]:
    """Recall for ``query``; return the units behind each returned memory, in rank order."""
    payload = core.recall_memories(
        query, session_id=session_id, top_k=top_k, max_tokens=max_tokens, config=config
    )
    ranked = [ingested.units_by_memory.get(m["id"], set()) for m in payload["memories"]]
    return ranked, int(payload["total_tokens"])


def first_hit_rank(ranked: list[set[str]], evidence: set[str]) -> int | None:
    """1-based rank of the first returned memory that carries any evidence unit."""
    for index, units in enumerate(ranked, start=1):
        if units & evidence:
            return index
    return None


def covered_fraction(ranked: list[set[str]], evidence: set[str]) -> float:
    """Fraction of the evidence units present anywhere in the returned set."""
    found: set[str] = set()
    for units in ranked:
        found |= units & evidence
    return len(found) / len(evidence)
