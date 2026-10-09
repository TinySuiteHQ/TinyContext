"""Measure TinyContext retrieval on LongMemEval (Wu et al., 2024).

## What this measures

LongMemEval (https://github.com/xiaowu0162/LongMemEval) has 500 questions. Each
one comes with its own chat history of ~50 sessions (the ``_s`` "small" split,
~115k tokens) and is labelled with the session(s) holding the answer
(``answer_session_ids``) and, within them, the turns that do (``has_answer``).

This script measures *retrieval only*, with no LLM in the loop:

1. For each question, its haystack is loaded into a fresh, throwaway store, one
   memory per turn (``[date] role: text``). No session scoping is used, so the
   evidence must be found among every session of that haystack.
2. The question is run through ``recall_memories`` within TinyContext's
   configured token budget and top-k.
3. It is graded on whether the labelled evidence was returned.

Like ``benchmark_locomo.py``, this is not the LLM-judged answer accuracy that
Mem0 and others publish, so it is not directly comparable to those numbers.

Reported per ``question_type``:

- Session Hit@k: any labelled answer session was returned.
- Session full recall: *every* answer session was returned (multi-session).
- Turn Hit@k: any ``has_answer`` turn itself was returned.
- MRR: reciprocal rank of the first memory from an answer session.

Abstention questions (``*_abs``) have no evidence to retrieve and are skipped.

## Usage

    python scripts/benchmark_longmemeval.py --limit 20
    python scripts/benchmark_longmemeval.py                      # all 500, slow
    python scripts/benchmark_longmemeval.py --top-k 20 --max-tokens 4000
    python scripts/benchmark_longmemeval.py --json-out scripts/benchmark_longmemeval.latest.json

``--limit N`` takes every (500 // N)-th question so the sample spans all question
types instead of just the first few.

Building 500 separate ~500-turn stores is the slow part; expect it to take hours
on CPU. Use ``--limit`` for a quick sample.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from _benchmark_common import (
    DATASET_CACHE_DIR,
    ensure_dataset,
    first_hit_rank,
    ingest,
    recalled_units,
)
from tinycontext.services.context_config_service import load_context_config
from tinycontext.services.memory_store_service import close_connection
from tinycontext.services.onnx_bundle_service import ensure_onnx_bundle_sync

DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_s_cleaned.json"
)
# Pin this after the first verified download (the script prints the hash it saw).
DATASET_SHA256: str | None = None
_SEP = "|"


def _turns(item: dict[str, Any]) -> list[tuple[str, str]]:
    """Flatten a haystack into ``(session_id|turn_index[|answer], text)`` in order."""
    turns: list[tuple[str, str]] = []
    for session_id, date, session in zip(
        item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"],
        strict=True,
    ):
        for index, turn in enumerate(session):
            unit = f"{session_id}{_SEP}{index}{_SEP}{int(bool(turn.get('has_answer')))}"
            turns.append((unit, f"[{date}] {turn['role']}: {turn['content']}"))
    return turns


def _evaluate_question(
    item: dict[str, Any], *, config: dict[str, Any], top_k: int | None, max_tokens: int | None
) -> dict[str, Any]:
    session_id = "longmemeval"
    turns = _turns(item)
    ingested = ingest(turns, config=config, session_id=session_id)

    answer_sessions = set(item["answer_session_ids"])
    answer_turns = {unit for unit, _ in turns if unit.endswith(f"{_SEP}1")}
    ranked, tokens = recalled_units(
        ingested, item["question"], config=config, session_id=session_id,
        top_k=top_k, max_tokens=max_tokens,
    )
    ranked_sessions = [{u.split(_SEP)[0] for u in units} for units in ranked]
    returned_sessions = set().union(*ranked_sessions) if ranked_sessions else set()
    rank = first_hit_rank(ranked_sessions, answer_sessions)
    return {
        "question_type": item["question_type"],
        "session_hit": rank is not None,
        "session_full": answer_sessions <= returned_sessions,
        "turn_hit": first_hit_rank(ranked, answer_turns) is not None if answer_turns else None,
        "rank": rank,
        "memories_returned": len(ranked),
        "tokens": tokens,
        "haystack_turns": len(turns),
    }


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"questions": 0}
    turn_rows = [r for r in rows if r["turn_hit"] is not None]
    return {
        "questions": n,
        "session_hit_at_k": sum(r["session_hit"] for r in rows) / n,
        "session_full_recall": sum(r["session_full"] for r in rows) / n,
        "turn_hit_at_k": (sum(r["turn_hit"] for r in turn_rows) / len(turn_rows))
        if turn_rows else None,
        "mrr": sum(1.0 / r["rank"] for r in rows if r["rank"]) / n,
        "mean_memories_returned": sum(r["memories_returned"] for r in rows) / n,
        "mean_tokens": sum(r["tokens"] for r in rows) / n,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = DATASET_CACHE_DIR / "longmemeval" / "longmemeval_s_cleaned.json"
    ensure_dataset(DATASET_URL, dataset, sha256=DATASET_SHA256)
    items = [
        item for item in json.loads(dataset.read_text(encoding="utf-8"))
        if not str(item["question_id"]).endswith("_abs")
    ]
    if args.limit and args.limit < len(items):
        step = len(items) / args.limit
        items = [items[int(i * step)] for i in range(args.limit)]

    base_config = load_context_config()
    ensure_onnx_bundle_sync(
        str(base_config["embedding_model"]), models_dir=str(base_config["models_dir"])
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(items, start=1):
        with tempfile.TemporaryDirectory(prefix="tinycontext-longmemeval-") as tmp_dir:
            db_path = Path(tmp_dir) / "longmemeval.db"
            config = dict(base_config)
            config["memory_db_path"] = str(db_path)
            row = _evaluate_question(
                item, config=config, top_k=args.top_k, max_tokens=args.max_tokens
            )
            close_connection(db_path)
        rows.append(row)
        print(
            f"[longmemeval] {index}/{len(items)} {item['question_type']}: "
            f"session_hit={row['session_hit']} ({row['haystack_turns']} turns, "
            f"{time.perf_counter() - started:.0f}s elapsed)",
            flush=True,
        )

    by_type = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
    return {
        "dataset": {
            "name": "LongMemEval-S (cleaned)", "url": DATASET_URL,
            "sha256": DATASET_SHA256,
        },
        "settings": {
            "top_k": args.top_k or int(base_config["recall_top_k"]),
            "max_tokens": args.max_tokens or int(base_config["recall_max_tokens"]),
            "embedding_model": str(base_config["embedding_model"]),
            "granularity": "one memory per turn; evidence graded by session and by turn",
            "sampled": bool(args.limit),
        },
        "overall": _summarise(rows),
        "by_question_type": {t: _summarise(by_type[t]) for t in sorted(by_type)},
        "elapsed_s": time.perf_counter() - started,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _fmt(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:>6.3f}"


def _print_report(report: dict[str, Any]) -> None:
    s = report["settings"]
    print(f"\nLongMemEval-S retrieval (top_k={s['top_k']}, max_tokens={s['max_tokens']}, "
          f"model={s['embedding_model']})")
    header = f"{'question type':<28} {'n':>4} {'sess@k':>7} {'full':>7} {'turn@k':>7} {'mrr':>6}"
    print(header)
    print("-" * len(header))
    for name, m in {**report["by_question_type"], "overall": report["overall"]}.items():
        print(
            f"{name:<28} {m['questions']:>4} {_fmt(m['session_hit_at_k'])} "
            f"{_fmt(m['session_full_recall'])} {_fmt(m['turn_hit_at_k'])} {_fmt(m['mrr'])}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate an evenly spaced sample of N questions.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override recall_top_k (default: from config).")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override recall_max_tokens (default: from config).")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = _parse_args()
    result = _run(cli_args)
    _print_report(result)
    if cli_args.json_out:
        cli_args.json_out.parent.mkdir(parents=True, exist_ok=True)
        cli_args.json_out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[longmemeval] wrote {cli_args.json_out}")
