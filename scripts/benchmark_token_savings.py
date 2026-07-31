"""Measure how many tokens TinyContext saves vs. a naive "resend everything" agent.

## What this measures

An agent with no memory compaction has one option for preserving context
across turns: resend everything it has ever recorded, every time, and let
the model's own context window absorb it. That's the naive baseline this
script isolates: the full, unfiltered text of every memory in the store,
concatenated, at the model's own token price.

TinyContext does that filtering locally: it hybrid-ranks (BM25 + dense) the
stored memories against the query and returns only the token-budgeted subset
that's actually relevant, with each item labeled by relevance.

For each benchmark query this script:

1. Seeds a throwaway store with a fixed pool of synthetic memories (same
   generator as ``benchmark_index_recall_speed.py``), isolated from the real
   configured database.
2. Counts tokens in the naive baseline: every stored memory's raw content,
   concatenated, via the same tokenizer TinyContext itself uses
   (``token_counter_service.token_count``, configured encoding).
3. Counts tokens in what TinyContext actually returns for the same query --
   the ``total_tokens`` field ``recall_memories`` already computes for its
   own budget accounting.
4. Reports the difference, and converts the tokens saved into a dollar
   figure using a configurable per-input-token price (default: current
   Claude Sonnet 5 API pricing) so the savings read as "$ saved per N
   recalls", not just a token count.

Held constant on purpose: which memories exist. The naive baseline is the
same store TinyContext itself recalls from, so the delta isolates local
compaction (hybrid rerank + token budget), not what got saved in the first
place.

## Usage

    python scripts/benchmark_token_savings.py
    python scripts/benchmark_token_savings.py --corpus-size 500
    python scripts/benchmark_token_savings.py --price-per-mtok-input 3.00
    python scripts/benchmark_token_savings.py --json-out scripts/benchmark_token_savings.latest.json
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
from tinycontext.services.token_counter_service import token_count

# Claude Sonnet 5 input price, per million tokens, as of this writing.
# Anthropic's own pricing is authoritative and can change -- pass
# --price-per-mtok-input to override rather than trusting this default blindly.
DEFAULT_PRICE_PER_MTOK_INPUT = 3.00

DEFAULT_CORPUS_SIZE = 300
BENCHMARK_QUERIES = [
    "what does the user prefer for code review",
    "database migration decisions and rollout plan",
    "known bugs in the authentication module",
    "deployment configuration for the staging environment",
    "project deadline and stakeholders",
    "which tool did we decide to use for session storage",
    "customer issues reported this quarter",
    "on-call handoff and incident review schedule",
]


@dataclass
class QueryBenchmark:
    query: str
    naive_tokens: int
    tinycontext_tokens: int
    tokens_saved: int
    pct_saved: float
    elapsed_s: float


def _seed_corpus(corpus_size: int, *, config: dict[str, Any], session_id: str) -> None:
    batch_size = 20
    for start in range(0, corpus_size, batch_size):
        count = min(batch_size, corpus_size - start)
        memories = [MemoryInput(content=synthetic_memory(start + i)) for i in range(count)]
        core.save_memories(memories, session_id=session_id, config=config)


def _benchmark_query(
    query: str, *, config: dict[str, Any], all_content: list[str], encoding_name: str
) -> QueryBenchmark:
    t0 = time.perf_counter()

    naive_text = "\n".join(all_content)
    naive_tokens = token_count(naive_text, encoding_name)

    result = core.recall_memories(query, config=config)
    tinycontext_tokens = int(result["total_tokens"])

    saved = naive_tokens - tinycontext_tokens
    pct_saved = (saved / naive_tokens * 100.0) if naive_tokens else 0.0

    return QueryBenchmark(
        query=query,
        naive_tokens=naive_tokens,
        tinycontext_tokens=tinycontext_tokens,
        tokens_saved=saved,
        pct_saved=pct_saved,
        elapsed_s=time.perf_counter() - t0,
    )


def _run(corpus_size: int, queries: list[str]) -> list[QueryBenchmark]:
    base_config = load_context_config()
    ensure_onnx_bundle_sync(
        str(base_config["embedding_model"]), models_dir=str(base_config["models_dir"])
    )
    encoding_name = str(base_config["encoding_name"])

    results: list[QueryBenchmark] = []
    with tempfile.TemporaryDirectory(prefix="tinycontext-benchmark-") as tmp_dir:
        db_path = Path(tmp_dir) / "benchmark_memories.db"
        config = dict(base_config)
        config["memory_db_path"] = str(db_path)
        session_id = "benchmark-session"

        print(f"[benchmark] seeding {corpus_size} synthetic memories...", flush=True)
        _seed_corpus(corpus_size, config=config, session_id=session_id)
        all_content = [synthetic_memory(i) for i in range(corpus_size)]

        for query in queries:
            print(f"[benchmark] running {query!r} ...", flush=True)
            bench = _benchmark_query(
                query, config=config, all_content=all_content, encoding_name=encoding_name
            )
            results.append(bench)
            print(
                f"[benchmark]   naive={bench.naive_tokens} tinycontext={bench.tinycontext_tokens} "
                f"saved={bench.tokens_saved} ({bench.pct_saved:.1f}%) in {bench.elapsed_s:.2f}s",
                flush=True,
            )

        close_connection(db_path)

    return results


def _print_report(results: list[QueryBenchmark], price_per_mtok_input: float) -> None:
    print()
    print(f"{'query':<50} {'naive':>8} {'tinycontext':>11} {'saved':>8} {'%saved':>7}")
    print("-" * 90)
    for bench in results:
        label = bench.query if len(bench.query) <= 49 else bench.query[:46] + "..."
        print(
            f"{label:<50} {bench.naive_tokens:>8} {bench.tinycontext_tokens:>11} "
            f"{bench.tokens_saved:>8} {bench.pct_saved:>6.1f}%"
        )
    print("-" * 90)

    total_naive = sum(b.naive_tokens for b in results)
    total_tinycontext = sum(b.tinycontext_tokens for b in results)
    total_saved = total_naive - total_tinycontext
    overall_pct = (total_saved / total_naive * 100.0) if total_naive else 0.0
    avg_pct = sum(b.pct_saved for b in results) / len(results) if results else 0.0

    print(
        f"{'TOTAL':<50} {total_naive:>8} {total_tinycontext:>11} {total_saved:>8} {overall_pct:>6.1f}%"
    )
    print()
    print(
        f"[benchmark] {len(results)} queries: naive baseline used {total_naive} tokens, "
        f"TinyContext used {total_tinycontext} tokens -> {overall_pct:.1f}% fewer tokens overall "
        f"({avg_pct:.1f}% average per query)."
    )

    avg_saved_per_query = total_saved / len(results) if results else 0
    cost_per_1k_recalls = avg_saved_per_query * 1000 / 1_000_000 * price_per_mtok_input
    print(
        f"[benchmark] at ${price_per_mtok_input:.2f}/MTok input: "
        f"~${cost_per_1k_recalls:.2f} saved per 1,000 recalls "
        f"(avg {avg_saved_per_query:.0f} tokens saved/recall)."
    )


def _write_json_report(
    path: Path, results: list[QueryBenchmark], price_per_mtok_input: float
) -> None:
    total_naive = sum(b.naive_tokens for b in results)
    total_tinycontext = sum(b.tinycontext_tokens for b in results)
    total_saved = total_naive - total_tinycontext
    avg_saved_per_query = total_saved / len(results) if results else 0
    payload = {
        "methodology": (
            "Naive baseline = every memory in the store, concatenated raw, tokenized "
            "with TinyContext's own configured encoding. TinyContext value = the "
            "total_tokens field recall_memories reports for the same query against "
            "the same store. Store contents are held constant so the delta isolates "
            "local hybrid-rerank + token-budget compaction, not what got saved."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "price_per_mtok_input": price_per_mtok_input,
        "queries": [asdict(b) for b in results],
        "totals": {
            "naive_tokens": total_naive,
            "tinycontext_tokens": total_tinycontext,
            "tokens_saved": total_saved,
            "pct_saved": (total_saved / total_naive * 100.0) if total_naive else 0.0,
            "usd_saved_per_1k_recalls": avg_saved_per_query * 1000 / 1_000_000 * price_per_mtok_input,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[benchmark] wrote {path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=DEFAULT_CORPUS_SIZE,
        help=f"Number of synthetic memories to seed (default: {DEFAULT_CORPUS_SIZE}).",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help="Queries to benchmark (default: a fixed built-in pool).",
    )
    parser.add_argument(
        "--price-per-mtok-input",
        type=float,
        default=DEFAULT_PRICE_PER_MTOK_INPUT,
        help=(
            "USD per million input tokens, used to convert tokens saved into a "
            f"dollar figure (default: ${DEFAULT_PRICE_PER_MTOK_INPUT:.2f}, current "
            "Claude Sonnet 5 pricing -- pass your own model's rate to match it)."
        ),
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
    queries = args.queries or BENCHMARK_QUERIES

    results = _run(args.corpus_size, queries)
    _print_report(results, args.price_per_mtok_input)
    if args.json_out:
        _write_json_report(args.json_out, results, args.price_per_mtok_input)
