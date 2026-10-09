"""Measure TinyContext retrieval on LoCoMo (long-term conversational memory).

## What this measures

LoCoMo (Maharana et al., 2024, https://github.com/snap-research/locomo) has 10
very long two-person conversations and ~2,000 questions. Every question is
labelled with the dialogue turns ("evidence", e.g. ``D1:3``) that contain the
answer.

This script measures *retrieval only*, with no LLM in the loop:

1. Each conversation is loaded into a fresh, throwaway store, one memory per
   dialogue turn (``[date] speaker: text``).
2. Each question is run through ``recall_memories`` exactly as an agent would,
   within TinyContext's configured token budget and top-k.
3. A question is graded on whether the returned memories contain its labelled
   evidence turns.

That is deliberately **not** the headline number other memory systems publish.
Mem0, Zep and Letta report an LLM-judged *answer accuracy* (retrieve, have a
model answer, have a judge grade). That depends on the answering model, the
judge and the prompt, so it isn't comparable across harnesses. Retrieval
metrics isolate the part TinyContext controls, but they are not a drop-in
substitute for those scores.

Reported per question category (1 multi-hop, 2 temporal, 3 open-domain,
4 single-hop):

- Hit@k: any evidence turn was returned.
- Full recall: *every* evidence turn was returned (the strict one for multi-hop).
- Evidence coverage: mean fraction of evidence turns returned.
- MRR: reciprocal rank of the first memory carrying evidence.

Category 5 (adversarial / unanswerable) and questions with no evidence labels
are excluded because there is nothing to retrieve.

## Usage

    python scripts/benchmark_locomo.py
    python scripts/benchmark_locomo.py --limit-conversations 2
    python scripts/benchmark_locomo.py --top-k 20 --max-tokens 4000
    python scripts/benchmark_locomo.py --json-out scripts/benchmark_locomo.latest.json
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from _benchmark_common import (
    DATASET_CACHE_DIR,
    covered_fraction,
    ensure_dataset,
    first_hit_rank,
    ingest,
    recalled_units,
)
from tinycontext.services.context_config_service import load_context_config
from tinycontext.services.memory_store_service import close_connection
from tinycontext.services.onnx_bundle_service import ensure_onnx_bundle_sync

DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
_DIA_ID = re.compile(r"D\d+:\d+")


def _turns(conversation: dict[str, Any]) -> list[tuple[str, str]]:
    """Flatten a conversation into ``(dia_id, memory_text)`` in chronological order."""
    session_numbers = sorted(
        int(key.split("_")[1])
        for key in conversation
        if re.fullmatch(r"session_\d+", key)
    )
    turns: list[tuple[str, str]] = []
    for number in session_numbers:
        date = conversation.get(f"session_{number}_date_time", "")
        for turn in conversation[f"session_{number}"]:
            text = turn["text"]
            if turn.get("blip_caption"):
                text += f" (shares a photo: {turn['blip_caption']})"
            turns.append((turn["dia_id"], f"[{date}] {turn['speaker']}: {text}"))
    return turns


def _evaluate_conversation(
    sample: dict[str, Any], *, config: dict[str, Any], top_k: int | None, max_tokens: int | None
) -> list[dict[str, Any]]:
    session_id = "locomo"
    turns = _turns(sample["conversation"])
    known = {dia_id for dia_id, _ in turns}
    ingested = ingest(turns, config=config, session_id=session_id)

    rows: list[dict[str, Any]] = []
    for qa in sample["qa"]:
        category = qa.get("category")
        evidence = {e for e in _DIA_ID.findall(" ".join(qa.get("evidence", []))) if e in known}
        if category not in CATEGORY_NAMES or not evidence:
            continue
        ranked, tokens = recalled_units(
            ingested, qa["question"], config=config, session_id=session_id,
            top_k=top_k, max_tokens=max_tokens,
        )
        rank = first_hit_rank(ranked, evidence)
        coverage = covered_fraction(ranked, evidence)
        rows.append(
            {
                "category": category,
                "hit": rank is not None,
                "rank": rank,
                "full": coverage == 1.0,
                "coverage": coverage,
                "evidence_count": len(evidence),
                "memories_returned": len(ranked),
                "tokens": tokens,
            }
        )
    return rows


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"questions": 0}
    return {
        "questions": n,
        "hit_at_k": sum(r["hit"] for r in rows) / n,
        "full_recall": sum(r["full"] for r in rows) / n,
        "evidence_coverage": sum(r["coverage"] for r in rows) / n,
        "mrr": sum(1.0 / r["rank"] for r in rows if r["rank"]) / n,
        "mean_memories_returned": sum(r["memories_returned"] for r in rows) / n,
        "mean_tokens": sum(r["tokens"] for r in rows) / n,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = ensure_dataset(
        DATASET_URL, DATASET_CACHE_DIR / "locomo" / "locomo10.json", sha256=DATASET_SHA256
    )
    samples = json.loads(dataset.read_text(encoding="utf-8"))
    if args.limit_conversations:
        samples = samples[: args.limit_conversations]

    base_config = load_context_config()
    ensure_onnx_bundle_sync(
        str(base_config["embedding_model"]), models_dir=str(base_config["models_dir"])
    )

    all_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        with tempfile.TemporaryDirectory(prefix="tinycontext-locomo-") as tmp_dir:
            db_path = Path(tmp_dir) / "locomo.db"
            config = dict(base_config)
            config["memory_db_path"] = str(db_path)
            t0 = time.perf_counter()
            rows = _evaluate_conversation(
                sample, config=config, top_k=args.top_k, max_tokens=args.max_tokens
            )
            close_connection(db_path)
        all_rows.extend(rows)
        summary = _summarise(rows)
        print(
            f"[locomo] {index}/{len(samples)} {sample.get('sample_id', '?')}: "
            f"{summary['questions']} questions, hit@k={summary.get('hit_at_k', 0):.2f} "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    by_category = defaultdict(list)
    for row in all_rows:
        by_category[row["category"]].append(row)
    return {
        "dataset": {"name": "LoCoMo (locomo10)", "url": DATASET_URL, "sha256": DATASET_SHA256},
        "settings": {
            "top_k": args.top_k or int(base_config["recall_top_k"]),
            "max_tokens": args.max_tokens or int(base_config["recall_max_tokens"]),
            "embedding_model": str(base_config["embedding_model"]),
            "granularity": "one memory per dialogue turn",
        },
        "overall": _summarise(all_rows),
        "by_category": {
            CATEGORY_NAMES[c]: _summarise(by_category[c]) for c in sorted(by_category)
        },
        "elapsed_s": time.perf_counter() - started,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _print_report(report: dict[str, Any]) -> None:
    s = report["settings"]
    print(f"\nLoCoMo retrieval (top_k={s['top_k']}, max_tokens={s['max_tokens']}, "
          f"model={s['embedding_model']})")
    header = f"{'category':<12} {'n':>5} {'hit@k':>7} {'full':>7} {'cover':>7} {'mrr':>6} {'tokens':>7}"
    print(header)
    print("-" * len(header))
    rows = {**report["by_category"], "overall": report["overall"]}
    for name, m in rows.items():
        print(
            f"{name:<12} {m['questions']:>5} {m['hit_at_k']:>7.3f} {m['full_recall']:>7.3f} "
            f"{m['evidence_coverage']:>7.3f} {m['mrr']:>6.3f} {m['mean_tokens']:>7.0f}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-conversations", type=int, default=None)
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
        print(f"[locomo] wrote {cli_args.json_out}")
