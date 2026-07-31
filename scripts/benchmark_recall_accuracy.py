"""Measure TinyContext's retrieval *accuracy*, not just its speed/tokens.

## What this measures

The other two benchmarks in this directory measure recall latency and token
compaction, but neither checks whether recall actually returns the *right*
memory. This script is a needle-in-haystack retrieval-accuracy eval:

1. Plant a fixed set of distinct "needle" facts (each phrased once, as a
   memory) inside a much larger pool of unrelated "haystack" filler memories
   (same generator as the other two benchmarks).
2. For each needle, query with a *paraphrase* of the fact -- different
   wording, not a substring match -- so a hit actually demonstrates semantic
   (dense) + keyword (BM25) retrieval working together, not just an exact
   string match.
3. Check whether the needle's memory id appears in what ``recall_memories``
   actually returns for that query, within TinyContext's own configured
   token budget -- i.e. what an agent would really receive, not an
   unbounded top-k.

Grading is exact-match on memory id, so this needs no LLM judge and is fully
deterministic and reproducible -- unlike LoCoMo/LongMemEval-style benchmarks,
which grade free-text answers with an LLM judge against human-labeled
conversational QA. That makes this a narrower, cheaper signal (pure
retrieval hit-rate on planted facts, not end-to-end answer quality), but a
real one: a memory system that can't retrieve the fact it stored can't
possibly answer correctly downstream either.

Reports, at each haystack size:

- Recall@k: fraction of needles whose memory made it into the final,
  token-budgeted result set.
- MRR (mean reciprocal rank): among hits, how high they ranked (1.0 = always
  the top result).

## Usage

    python scripts/benchmark_recall_accuracy.py
    python scripts/benchmark_recall_accuracy.py --haystack-sizes 100 1000 5000
    python scripts/benchmark_recall_accuracy.py --json-out scripts/benchmark_recall_accuracy.latest.json
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_HAYSTACK_SIZES = [100, 1000, 5000]
BATCH_SIZE = 20

# Each needle is a distinct, specific fact. Its query is a paraphrase --
# different vocabulary, sometimes different sentence shape -- so a hit
# requires actual semantic/keyword matching, not a copy-pasted substring.
NEEDLES: list[tuple[str, str]] = [
    (
        "The production database credential rotation policy requires changing "
        "all service-account passwords every 45 days, enforced by Dana Kim on the security team.",
        "how often do we have to rotate database credentials and who enforces it",
    ),
    (
        "Customer Fenwick Labs is on a custom enterprise contract that caps their "
        "monthly API usage at 2.5 million requests before overage billing kicks in.",
        "what's the request cap for Fenwick Labs before they get billed extra",
    ),
    (
        "The checkout service has a hard 3-second timeout on the payment gateway call, "
        "after which it falls back to a queued async charge.",
        "what happens if the payment gateway takes too long during checkout",
    ),
    (
        "Incident postmortem for the March 2026 outage concluded the root cause was a "
        "misconfigured connection pool size on the order-processing worker.",
        "what caused the outage we had back in March",
    ),
    (
        "The mobile app's minimum supported OS version is iOS 16 and Android 12, "
        "dropping older versions starting next release.",
        "which OS versions are we cutting support for on mobile",
    ),
    (
        "Legal flagged that the old auth middleware stores session tokens in a way "
        "that doesn't meet the new compliance requirements, driving the rewrite.",
        "why are we actually rewriting the auth middleware",
    ),
    (
        "The data warehouse nightly ETL job runs at 2am UTC and typically takes 40 "
        "minutes, alerting if it exceeds 90 minutes.",
        "when does the nightly ETL run and how long should it normally take",
    ),
    (
        "Support escalations for enterprise tier customers must get a first response "
        "within 1 hour during business hours, per the SLA signed in January.",
        "what's the response time SLA for enterprise support tickets",
    ),
    (
        "The recommendation engine was switched from a collaborative-filtering model "
        "to a two-tower embedding model to fix cold-start recommendations for new users.",
        "why did we change the recommendation algorithm",
    ),
    (
        "Vendor contract renewal for the observability platform is due at the end of "
        "Q3, and the team is evaluating whether to switch providers instead.",
        "when is the observability vendor contract up for renewal",
    ),
    (
        "The onboarding flow drop-off analysis found most users abandon at the email "
        "verification step, not at account creation.",
        "where in onboarding do most users actually drop off",
    ),
    (
        "Feature flag `new_billing_ui` is enabled for 10% of traffic and being ramped "
        "by 10 percentage points each week pending no error-rate regressions.",
        "what's the current rollout percentage for the new billing UI",
    ),
    (
        "The API rate limiter uses a token bucket with a burst capacity of 50 requests "
        "and a steady refill rate of 5 requests per second per API key.",
        "how does our API rate limiting actually work",
    ),
    (
        "Warehouse capacity planning shows we'll run out of disk on the primary "
        "Postgres instance in roughly 5 months at current growth.",
        "how much runway do we have before the primary database runs out of disk",
    ),
    (
        "The design system's color palette was updated to meet WCAG AA contrast "
        "requirements after an accessibility audit flagged several components.",
        "why did the color palette change in the design system",
    ),
]


@dataclass
class NeedleResult:
    needle_index: int
    query: str
    hit: bool
    rank: int | None
    total_tokens: int


@dataclass
class HaystackReport:
    haystack_size: int
    recall_at_k: float
    mrr: float
    hits: int
    total_needles: int
    elapsed_s: float


def _seed_haystack(size: int, *, config: dict[str, Any], session_id: str) -> None:
    for start in range(0, size, BATCH_SIZE):
        count = min(BATCH_SIZE, size - start)
        memories = [MemoryInput(content=synthetic_memory(start + i)) for i in range(count)]
        core.save_memories(memories, session_id=session_id, config=config)


def _plant_needles(*, config: dict[str, Any], session_id: str) -> dict[int, str]:
    """Save each needle individually and return {needle_index: memory_id}."""
    needle_ids: dict[int, str] = {}
    for index, (fact, _query) in enumerate(NEEDLES):
        result = core.save_memories(
            [MemoryInput(content=fact)], session_id=session_id, config=config
        )
        needle_ids[index] = result["saved"][0]["id"]
    return needle_ids


def _evaluate(
    needle_ids: dict[int, str], *, config: dict[str, Any]
) -> list[NeedleResult]:
    results: list[NeedleResult] = []
    for index, (_fact, query) in enumerate(NEEDLES):
        payload = core.recall_memories(query, config=config)
        returned_ids = [m["id"] for m in payload["memories"]]
        target_id = needle_ids[index]
        hit = target_id in returned_ids
        rank = returned_ids.index(target_id) + 1 if hit else None
        results.append(
            NeedleResult(
                needle_index=index,
                query=query,
                hit=hit,
                rank=rank,
                total_tokens=int(payload["total_tokens"]),
            )
        )
    return results


def _run(haystack_sizes: list[int]) -> list[HaystackReport]:
    base_config = load_context_config()
    ensure_onnx_bundle_sync(
        str(base_config["embedding_model"]), models_dir=str(base_config["models_dir"])
    )

    reports: list[HaystackReport] = []
    current_size = 0

    with tempfile.TemporaryDirectory(prefix="tinycontext-benchmark-") as tmp_dir:
        db_path = Path(tmp_dir) / "benchmark_memories.db"
        config = dict(base_config)
        config["memory_db_path"] = str(db_path)
        session_id = "benchmark-session"

        print(f"[benchmark] planting {len(NEEDLES)} needles...", flush=True)
        needle_ids = _plant_needles(config=config, session_id=session_id)

        for haystack_size in haystack_sizes:
            print(f"[benchmark] growing haystack to {haystack_size} filler memories...", flush=True)
            if haystack_size > current_size:
                _seed_haystack(
                    haystack_size - current_size, config=config, session_id=session_id
                )
                current_size = haystack_size

            print("[benchmark] querying for each needle...", flush=True)
            t0 = time.perf_counter()
            results = _evaluate(needle_ids, config=config)
            elapsed = time.perf_counter() - t0

            hits = sum(1 for r in results if r.hit)
            recall_at_k = hits / len(results) if results else 0.0
            mrr = (
                sum(1.0 / r.rank for r in results if r.hit) / len(results)
                if results
                else 0.0
            )
            report = HaystackReport(
                haystack_size=haystack_size + len(NEEDLES),
                recall_at_k=recall_at_k,
                mrr=mrr,
                hits=hits,
                total_needles=len(results),
                elapsed_s=elapsed,
            )
            reports.append(report)
            print(
                f"[benchmark]   recall@k={report.recall_at_k:.2f} mrr={report.mrr:.2f} "
                f"({hits}/{len(results)} hit) in {elapsed:.1f}s",
                flush=True,
            )
            for r in results:
                if not r.hit:
                    print(f"[benchmark]   MISS: {r.query!r}", flush=True)

        close_connection(db_path)

    return reports


def _print_report(reports: list[HaystackReport]) -> None:
    print()
    print(f"{'haystack_size':>13} {'recall@k':>9} {'mrr':>6} {'hits/total':>11}")
    print("-" * 44)
    for r in reports:
        print(
            f"{r.haystack_size:>13} {r.recall_at_k:>9.2f} {r.mrr:>6.2f} "
            f"{r.hits:>5}/{r.total_needles:<5}"
        )
    print("-" * 44)


def _write_json_report(path: Path, reports: list[HaystackReport]) -> None:
    payload = {
        "methodology": (
            f"{len(NEEDLES)} distinct 'needle' facts, each saved once and queried with "
            "a paraphrase (not a substring match), planted inside a synthetic filler "
            "haystack of increasing size. Grading is exact-match on memory id within "
            "recall_memories' own token-budgeted result set -- i.e. what an agent "
            "would actually receive. This is a narrower, cheaper, fully-deterministic "
            "signal than LLM-judged conversational benchmarks (LoCoMo/LongMemEval): "
            "pure retrieval hit-rate on known facts, not end-to-end answer quality."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "needle_count": len(NEEDLES),
        "haystacks": [asdict(r) for r in reports],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[benchmark] wrote {path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--haystack-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Filler-memory pool sizes to test at, in increasing order (default: 100 1000 5000).",
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
    haystack_sizes = sorted(args.haystack_sizes or DEFAULT_HAYSTACK_SIZES)

    reports = _run(haystack_sizes)
    _print_report(reports)
    if args.json_out:
        _write_json_report(args.json_out, reports)
