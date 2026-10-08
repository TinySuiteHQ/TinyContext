# Memory gardener: deferred idea

TinyContext is currently working well, and the GitHub repository has no open
issues identifying a memory-quality problem that needs a gardener. This
document records the idea for future reference; it is not an implementation
plan or a commitment to add an LLM, background job, entity graph, or retention
system.

## Current position

- Keep TinyContext's existing local SQLite memory store, hybrid recall,
  profile memories, duplicate protection, and explicit correction workflow.
- Do not build a gardener speculatively. The added model, schema, lifecycle,
  and operations complexity needs a demonstrated user problem to justify it.
- Revisit this idea only when concrete reports or evaluation results show a
  recurring failure in saving, correcting, or recalling useful memories.

## Evidence to collect if a problem appears

For each case, record a representative sequence of saved memories, the query,
what TinyContext returned, what should have been returned, and why the result
was wrong. Group cases into the smallest clear failure categories, such as:

- redundant saves that materially reduce useful recall;
- corrected facts continuing to appear after an explicit update;
- relevant memories missing from results under a realistic token budget;
- stale details outranking current information in a way users cannot correct
  with the existing update and delete operations.

First check whether retrieval tuning, clearer save guidance, or the existing
explicit update/delete workflow fixes the cases. Consider automated grooming
only if those smaller changes do not address a measured, recurring problem.

## Reconsideration gate

Before proposing implementation, define a small evaluation set from real or
carefully anonymized failure cases. Compare the current behavior against the
smallest candidate change. A gardener would need to show a meaningful
improvement in recall quality without unacceptable regressions, latency,
resource use, or loss of user control. Until that evidence exists, the gardener
remains deferred.
