"""Benchmark TinyContext's write throughput and recall latency as the store grows.

## What this measures

TinyContext keeps a persistent SQLite store (FTS5 + sqlite-vec) that every
``save_memories`` call writes to and every ``recall_memories`` call reads
from. This script answers two operational questions:

1. How many memories/sec can be written (embedding + FTS/vector indexing),
   and does that throughput hold steady as the store grows?
2. How does recall latency change as the corpus grows from a handful of
   memories to several thousand?

It calls ``core.save_memories`` / ``core.recall_memories`` directly (no MCP
or FastAPI transport in the loop) against an isolated, throwaway SQLite file
-- never the real configured store -- so timings measure the storage/recall
engine itself. It reuses the *real* configured embedding model and
``models_dir`` (so no extra download is needed) via
``load_context_config()``, only overriding ``memory_db_path``.

## Usage

    python scripts/benchmark_index_recall_speed.py
    python scripts/benchmark_index_recall_speed.py --checkpoints 100 1000 5000
    python scripts/benchmark_index_recall_speed.py --json-out scripts/benchmark_index_recall_speed.latest.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from _synthetic_memories import synthetic_memory
from tinycontext import core
from tinycontext.models import MemoryInput
from tinycontext.services.context_config_service import load_context_config
from tinycontext.services.memory_store_service import close_connection
from tinycontext.services.onnx_bundle_service import ensure_onnx_bundle_sync

DEFAULT_CHECKPOINTS = [100, 500, 2000, 5000]
BATCH_SIZE = 20
RECALL_REPEATS = 5
RECALL_QUERIES = [
    "what does the user prefer for code review",
    "database migration decisions and rollout plan",
    "known bugs in the authentication module",
    "deployment configuration for the staging environment",
    "project deadline and stakeholders",
]


@dataclass
class WriteBatch:
    corpus_size_before: int
    batch_size: int
    elapsed_s: float
    memories_per_s: float


@dataclass
class CheckpointReport:
    corpus_size: int
    write_memories_per_s: float
    recall_p50_ms: float
    recall_p95_ms: float
    recall_min_ms: float
    recall_max_ms: float


def _write_up_to(
    target_size: int,
    current_size: int,
    *,
    config: dict[str, Any],
    session_id: str,
) -> tuple[int, list[WriteBatch]]:
    batches: list[WriteBatch] = []
    while current_size < target_size:
        batch_size = min(BATCH_SIZE, target_size - current_size)
        memories = [
            MemoryInput(content=synthetic_memory(current_size + i))
            for i in range(batch_size)
        ]
        t0 = time.perf_counter()
        core.save_memories(memories, session_id=session_id, config=config)
        elapsed = time.perf_counter() - t0
        batches.append(
            WriteBatch(
                corpus_size_before=current_size,
                batch_size=batch_size,
                elapsed_s=elapsed,
                memories_per_s=batch_size / elapsed if elapsed > 0 else float("inf"),
            )
        )
        current_size += batch_size
    return current_size, batches


def _benchmark_recall(
    corpus_size: int, *, config: dict[str, Any]
) -> list[float]:
    latencies_ms: list[float] = []
    for _ in range(RECALL_REPEATS):
        for query in RECALL_QUERIES:
            t0 = time.perf_counter()
            core.recall_memories(query, config=config)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return latencies_ms


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def _run(checkpoints: list[int]) -> tuple[list[CheckpointReport], list[WriteBatch]]:
    base_config = load_context_config()
    ensure_onnx_bundle_sync(
        str(base_config["embedding_model"]), models_dir=str(base_config["models_dir"])
    )

    reports: list[CheckpointReport] = []
    all_batches: list[WriteBatch] = []
    current_size = 0

    with tempfile.TemporaryDirectory(prefix="tinycontext-benchmark-") as tmp_dir:
        db_path = Path(tmp_dir) / "benchmark_memories.db"
        config = dict(base_config)
        config["memory_db_path"] = str(db_path)
        session_id = "benchmark-session"

        for checkpoint in checkpoints:
            print(f"[benchmark] writing up to {checkpoint} memories...", flush=True)
            current_size, batches = _write_up_to(
                checkpoint, current_size, config=config, session_id=session_id
            )
            all_batches.extend(batches)
            write_rate = statistics.mean(b.memories_per_s for b in batches[-5:] if batches)

            print(f"[benchmark] recalling at corpus_size={current_size}...", flush=True)
            latencies_ms = _benchmark_recall(current_size, config=config)

            report = CheckpointReport(
                corpus_size=current_size,
                write_memories_per_s=write_rate,
                recall_p50_ms=_percentile(latencies_ms, 50),
                recall_p95_ms=_percentile(latencies_ms, 95),
                recall_min_ms=min(latencies_ms),
                recall_max_ms=max(latencies_ms),
            )
            reports.append(report)
            print(
                f"[benchmark]   write={report.write_memories_per_s:.1f} mem/s "
                f"recall_p50={report.recall_p50_ms:.1f}ms "
                f"recall_p95={report.recall_p95_ms:.1f}ms",
                flush=True,
            )

        close_connection(db_path)

    return reports, all_batches


def _print_report(reports: list[CheckpointReport]) -> None:
    print()
    print(
        f"{'corpus_size':>11} {'write mem/s':>12} {'recall p50':>11} "
        f"{'recall p95':>11} {'recall min':>11} {'recall max':>11}"
    )
    print("-" * 72)
    for r in reports:
        print(
            f"{r.corpus_size:>11} {r.write_memories_per_s:>12.1f} "
            f"{r.recall_p50_ms:>9.1f}ms {r.recall_p95_ms:>9.1f}ms "
            f"{r.recall_min_ms:>9.1f}ms {r.recall_max_ms:>9.1f}ms"
        )
    print("-" * 72)


def _write_json_report(
    path: Path, reports: list[CheckpointReport], batches: list[WriteBatch]
) -> None:
    payload = {
        "methodology": (
            "Writes and recalls run against an isolated, throwaway SQLite store "
            "(never the real configured database) via core.save_memories / "
            "core.recall_memories directly -- no MCP or FastAPI transport in the "
            "loop. write_memories_per_s is the mean over the last 5 batches "
            "written to reach each checkpoint; recall latencies are over "
            f"{RECALL_REPEATS} repeats x {len(RECALL_QUERIES)} queries at that "
            "corpus size."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoints": [asdict(r) for r in reports],
        "write_batches": [asdict(b) for b in batches],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[benchmark] wrote {path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=None,
        help="Corpus sizes to measure at, in increasing order (default: 100 500 2000 5000).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report to.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    checkpoints = sorted(args.checkpoints or DEFAULT_CHECKPOINTS)

    reports, batches = _run(checkpoints)
    _print_report(reports)
    if args.json_out:
        _write_json_report(args.json_out, reports, batches)
