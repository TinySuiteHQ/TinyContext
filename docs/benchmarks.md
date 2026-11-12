# Benchmarks

Numbers below come from `scripts/benchmark_index_recall_speed.py` and
`scripts/benchmark_token_savings.py`, run against an isolated, throwaway
SQLite store (never a real database) with the default `balanced` ONNX embedding
model. Reproduce them yourself:

```bash
python scripts/benchmark_index_recall_speed.py --checkpoints 100 500 2000 5000 10000 20000 --json-out speed.json
python scripts/benchmark_token_savings.py --json-out savings.json
python scripts/benchmark_recall_accuracy.py --json-out accuracy.json
```

## Write throughput and recall latency

| Corpus size | Write throughput | Recall p50 | Recall p95 |
| --- | --- | --- | --- |
| 100 | 170.0 mem/s | 8.9ms | 9.9ms |
| 500 | 210.3 mem/s | 9.0ms | 10.1ms |
| 2,000 | 208.2 mem/s | 10.5ms | 12.3ms |
| 5,000 | 204.3 mem/s | 10.8ms | 12.6ms |
| 10,000 | 180.5 mem/s | 11.0ms | 12.3ms |
| 20,000 | 186.5 mem/s | 11.4ms | 12.5ms |

Recall stays flat well past a few thousand memories: p95 only grows from
9.9ms to 12.5ms across a 200x increase in corpus size (100 to 20,000). Each
retriever (BM25, dense) hands over its own top-scoring candidates rather than
every stored memory being hydrated and ranked on each call; only that
bounded, unioned pool is fused and scored. There's still no ANN index
underneath — SQLite has to evaluate every candidate row to find those
top-scoring matches, so this is bounded brute force rather than sublinear
search, and will eventually bend upward again well beyond the sizes tested
here. Write throughput holds steady regardless of corpus size.

## Token savings vs. a naive "resend everything" agent

Against 300 synthetic memories and 8 queries: **96.7% fewer tokens** than
concatenating every stored memory raw, or roughly **$16.42 saved per 1,000
recalls** at $3/MTok input pricing (Claude Sonnet 5).

## How this compares to the market

Published numbers from [Mem0](https://mem0.ai/research) (~90%+ token
reduction, ~200ms p95 latency) and [Zep](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
(~65–200ms p95 latency) put TinyContext at or ahead on token compaction, and
ahead on latency at the corpus sizes tested here (single-digit-to-low-teens
ms p95 vs. 65-200ms). That's not an apples-to-apples claim, though — those
figures come from real conversational benchmarks (LoCoMo, LongMemEval) with
retrieval-accuracy grading in the loop, run against hosted vector databases
at larger scale than tested above.

## Retrieval accuracy — an open question, not a claim

`scripts/benchmark_recall_accuracy.py` plants 15 distinct facts inside a
growing pool of filler memories and queries each with a paraphrase, checking
whether hybrid recall returns the right memory id. Locally this comes back
at **100% recall@k and MRR 1.00** from 100 up to 5,000 filler memories — but
the planted facts are semantically distinct from the filler, so this mostly
shows the mechanism works, not that it holds up against confusable,
near-duplicate memories or a real labeled benchmark like LoCoMo/LongMemEval.

**This is the one number here we're not standing behind as-is.** If you run
a harder or larger-scale accuracy eval against TinyContext — adversarial
near-duplicates, a real conversational dataset, whatever — we'd genuinely
like to see it, good or bad. Open an issue or a PR with what you found.

## LoCoMo and LongMemEval (retrieval only)

`scripts/benchmark_locomo.py` and `scripts/benchmark_longmemeval.py` run
TinyContext against the two standard conversational-memory datasets, grading
whether `recall_memories` returns the turns or sessions the dataset labels as
evidence, within the configured token budget and top-k. They use no LLM, so the
results are deterministic and need no API key. Datasets are downloaded to
`.cache/benchmarks/` on first run.

These are **retrieval** metrics, not the LLM-judged answer accuracy that Mem0,
Zep and Letta publish, so they are not directly comparable to those figures.

```bash
python scripts/benchmark_locomo.py --json-out scripts/benchmark_locomo.latest.json
python scripts/benchmark_longmemeval.py --limit 50 --json-out scripts/benchmark_longmemeval.latest.json
```

Results: not yet run.
