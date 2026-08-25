# TinyContext Memory Gardener: goals and delivery plan

## Product intent

TinyContext should accept a generous stream of observations from an agent without
turning that convenience into an ever-growing, contradictory prompt dump. A
small, optional LLM acts as a **memory gardener**: it reviews new material,
finds entities and relationships, combines redundant observations into useful
memories, and lets stale detail gradually lose priority and eventually leave
the active store.

This is deliberately an adaptation of patterns that established memory systems
have already put into production, not a new memory theory. In particular, the
plan uses Mem0's add-first extraction, entity-aware hybrid retrieval, soft
access-based decay, and scheduled background consolidation, together with
Zep/Graphiti's separation of raw episodes from derived entities, facts,
summaries, and temporal validity. The first delivery phase is a comparative
spike and evaluation, not a commitment to reproduce either product's API or
graph infrastructure.

The gardener is an advisor, not the source of truth. TinyContext remains the
local, deterministic memory system; the model proposes changes and the core
validates, records, and applies them. This keeps the feature useful with small
local models and safe when a model makes a bad inference.

The intended experience is:

1. An agent may save as many candidate memories as are useful during a task.
2. A small model can periodically groom those candidates, rather than making
   every write expensive or requiring a large model at the end of every turn.
3. Every recall includes a bounded short-term working set for the active
   session, plus the existing durable profile block and query-relevant
   long-term memories.
4. Memories become less prominent when they are old, unconfirmed, redundant,
   or no longer useful; they move through explicit states before any permanent
   deletion.

## What exists today

This plan extends the current local SQLite store and should preserve its useful
properties:

- episodic and global `profile` memories;
- hybrid BM25 and dense retrieval with a token budget;
- duplicate detection on save, supersession, `last_recalled_at`, and
  `recall_count` lifecycle fields;
- prompt-ready MCP recall, plus structured Python and HTTP responses;
- explicit list, get, update, and delete operations.

The current `profile` block is automatically included with every recall. The
new working-memory block must be similarly bounded and clearly labelled as
background context, never as instructions.

## Established patterns to adopt deliberately

| Proven pattern | Reference | TinyContext adaptation |
| --- | --- | --- |
| Add-first fact extraction preserves source history; entities support retrieval alongside semantic and keyword signals. | [Mem0's current memory algorithm](https://docs.mem0.ai/migration/platform-v2-to-v3) | Keep source observations and create derived memories/links rather than overwriting the original text. Add entity links as a ranking boost to TinyContext's existing hybrid RRF. |
| A background “dream” pass merges, supersedes, and synthesizes memories without delaying add or search. | [Mem0 Dream consolidation](https://mem0.ai/blog/dream-background-memory-consolidation-for-ai-agents) | Run grooming off the request path in bounded batches; require conditional, auditable state changes. |
| Decay is a reversible, bounded search-time ranking modifier, not silent loss of recall. | [Mem0 Memory Decay](https://docs.mem0.ai/platform/features/memory-decay) | Build on `last_recalled_at` and `recall_count`; make decay opt-in, floor its effect, and measure it before adding archival policy. |
| Raw episodes, derived entities/facts, and compact summaries are separate artifacts with provenance. | [Zep graph creation](https://help.getzep.com/how-graph-creation-works) | Store raw source memories separately from derived consolidation/entity links and session working summaries. |
| Facts have temporal validity rather than a destructive replacement of history. | [Zep facts and summaries](https://help.getzep.com/v2/facts) | Capture `valid_at`/`invalid_at` where evidence exists; use supersession/invalidation links before considering removal. |
| A prompt-ready context block can consistently include a durable user/session summary plus relevant facts. | [Zep context types](https://help.getzep.com/context-types) | Return the bounded working-memory block on every TinyContext recall, alongside the existing profile and relevant-memory blocks. |

These references are design inputs, not proof that their trade-offs fit every
local deployment. TinyContext should retain its local SQLite-first, lightweight
scope: no mandatory cloud account, graph database, or opaque managed pipeline.

## Goals

### 1. A memory intake that favours capture over premature judgement

`save_memories` should continue to accept batches. It should not require the
calling model to decide perfectly whether a note is permanent, what entity it
belongs to, or how it should be worded. Save raw observations quickly, with
their source/session/time and a stable id. Cheap deterministic exact and
near-duplicate protection may still reject obvious repeats.

Incoming items enter an **inbox** (or equivalent `pending` lifecycle state),
following the add-first pattern rather than deleting history during extraction.
They are retrievable when appropriate, but they are not automatically elevated
to durable profile facts merely because the writer says so. This makes it safe
for a model to save freely while giving the gardener material to improve.

### 2. An optional, provider-neutral small-LLM gardener

Support a locally runnable or operator-supplied model through a narrow adapter,
not a hard-coded vendor integration. The adapter receives a bounded batch of
candidate memories and a constrained task schema. It returns JSON proposals,
never executable database operations.

Initial proposal types:

- `extract_entities`: identify named people, organisations, projects, places,
  tools, and concepts; link aliases to a canonical entity when evidence is
  sufficient;
- `consolidate`: propose one concise memory from related source memories;
- `supersede`: state that a newer/corrected memory replaces another;
- `classify`: suggest an explicit memory kind, importance, and confidence;
- `refresh`: update a concise living summary of an active topic/session;
- `deprecate` or `archive`: recommend removal from active recall with a reason.

The core validates ids, ownership/session scope, size limits, proposal schema,
and allowed state transitions. It must reject a whole invalid proposal rather
than guessing what the model meant. Every accepted proposal stores its source
memory ids, model/adapter identifier, timestamp, reason, and confidence.

The gardener is opt-in. Existing installations must retain current save and
recall behaviour until the feature is enabled.

### 3. Entity-aware, evidence-preserving memory

Add a first-class entity record and a many-to-many memory-to-entity link. Store
canonical name, type, aliases, and normalised matching keys; do not require a
large knowledge graph in the first release. Entities are retrieval and
consolidation aids, not facts on their own.

An entity link must cite the memory or memories that justified it. Conflicting
aliases and uncertain links remain reviewable proposals rather than silently
merging people or projects. Entity extraction should use a deterministic
normalisation layer after the LLM suggestion so punctuation/casing variants do
not produce needless duplicates.

### 4. Short-term working memory that surfaces on every recall

Add a session-scoped **working-memory store** for current goals, open questions,
recent decisions, constraints, and handoffs. It is separate from chronological
episodic history:

- Each `recall_memories` response includes a `<working_memory>` block before
  ranked/recent long-term memories, whether the request is semantic or recent.
- The block has its own small token budget, deterministic ordering, and a
  visible `updated_at`; it does not consume the long-term recall budget.
- Its contents are a compact, groomed summary plus a bounded list of active
  items, not a raw transcript.
- A session has at most one current working summary/version. Updating it
  creates an auditable revision; it never erases source observations.
- Calls with no session use no session working block. Global profiles remain
  separate and continue to apply as today.

This means "surface with every call" at TinyContext's recall boundary. It does
not mean automatically attaching memory to unrelated application RPCs or
silently sending local content to a model provider.

### 5. Deliberate lifecycle, degradation, and expiry

Every non-profile memory receives a lifecycle state and enough timestamps to
explain its treatment. The initial focus is non-destructive visibility and
ranking, following the reversible decay and temporal-invalidation patterns;
physical deletion remains a separately approved retention function.

`inbox -> active -> archived -> eligible_for_purge -> purged`

`superseded` is a parallel terminal-active state that points to its replacement.
The system uses a transparent score from recency, recall/usefulness, source
confidence, duplication, entity/topic activity, and an optional explicit
importance. It may lower ranking weight over time, but it must not fabricate a
claim that age proves false. Decay is a bounded, opt-in ranking modifier with a
non-zero floor; it is never a hidden candidate filter.

Default policy principles:

- Working memory expires or is rewritten when its session becomes inactive.
- Inbox material is groomed after a configurable age or batch size; unpromoted
  material decays out of normal recall first.
- Active episodic detail can archive after sustained inactivity. A surviving
  consolidation keeps provenance to the details it summarised.
- Archives are excluded from ordinary recall but remain listable/restorable.
- Permanent purge is a separate retention-policy action, with a grace period
  and audit record. It is disabled by default until policy, export, and restore
  paths are proven.
- Profile facts do not quietly age out. Conflicting profile candidates require
  explicit correction/supersession and remain globally scoped.

## Proposed architecture

```text
agent writes freely
        |
        v
raw memory inbox -- deterministic dedup --> SQLite source memories
        |                                      |
        | bounded batches                      | provenance / audit
        v                                      v
small LLM gardener --> validated proposals --> entities, summaries,
                                           working memory, lifecycle changes
        |
        v
recall: profile + working memory + hybrid long-term retrieval
```

The original raw memory remains the provenance record. A consolidation creates
a new memory with `derived_from` links; it does not overwrite its sources.
Supersession is reserved for a correction or replacement, not for a summary.

### Storage additions

Use forward-only SQLite migrations. Likely tables/columns are:

- `memories`: `state`, `updated_at`, `last_confirmed_at`, `importance`,
  `confidence`, `archived_at`, `purge_eligible_at`, and `source_type`;
- `memory_relations`: typed `derived_from`, `supersedes`, `contradicts`, and
  `supports` edges with proposal/audit ids;
- `entities` and `memory_entities`: canonical entities, aliases, and cited
  links;
- `working_memory_revisions`: session id, revision, content, active items,
  source ids, created/update times, and expiry;
- `gardener_runs` and `gardener_actions`: input bounds, adapter/model metadata,
  validated result, rejection reason, and before/after values;
- `retention_events`: archive, restore, eligibility, and purge decisions.

No migration may discard existing memories. Existing rows start as `active`
episodic/profile records and retain their current recall behaviour.

### Gardener boundaries

- Run asynchronously after a save threshold, on a scheduled local job, or by
  an explicit API/CLI command. Never add model latency to ordinary recall.
- Give the model only the bounded batch, relevant existing summaries/entities,
  and a strict output schema. Do not put the whole database in its context.
- Limit each run's inputs, outputs, tokens, relation fan-out, and database
  writes. A model may save many raw observations across calls, but one grooming
  pass cannot create an unbounded graph or rewrite unrelated memory.
- Treat model output as untrusted content. It cannot set policy, change tenant
  scope, invoke tools, or delete rows directly.
- Keep the model interface pluggable: a local command/HTTP adapter is enough
  initially. The core should not depend on a particular model, provider, or
  prompt framework.

## Delivery phases

### Phase 0 — contract and measurement

Read and trial the relevant Mem0 and Zep/Graphiti designs against TinyContext's
actual constraints before choosing data structures. Define lifecycle vocabulary,
a versioned gardener proposal schema, and acceptance/rejection semantics. Add
fixtures containing duplicates, corrections, changing preferences, stale
project details, and ambiguous entity names. Record baseline recall quality,
token use, row growth, and latency, then compare add-only versus update-at-write
and decay-only versus archive policies.

### Phase 1 — deterministic working memory and lifecycle foundation

Add schema migrations, lifecycle state, relationship/audit records, and the
working-memory revision store. Expose working memory in Python, HTTP, and MCP
recall without changing the meaning of existing ranked memory fields. Add a
safe CLI/API to inspect, restore, and clear a session's working memory.

### Phase 2 — gardener adapter and dry-run mode

Introduce the provider-neutral adapter and JSON schema. Implement a dry-run
that produces proposals and audit records but changes nothing. Add deterministic
validation and an operator-visible report of accepted/rejected proposals.

### Phase 3 — consolidation and entity extraction

Enable `consolidate`, `supersede`, `extract_entities`, and working-summary
refresh only after dry-run fixtures meet the quality bar. Preserve full
provenance, require scope checks, and provide restore/undo for accepted actions.

### Phase 4 — degradation and retention

Apply ranking decay and archive transitions first. Measure recall regressions
and false archive rates. Only then enable configurable purge eligibility; keep
physical deletion opt-in and subject to a grace period, export, and audit.

### Phase 5 — operations and documentation

Document how to enable the gardener, choose/run a small local model, configure
budgets and retention, review actions, restore archives, and fully disable the
feature. Provide doctor output for migration/adapter status and metrics that
contain counts and timings but no raw memory content by default.

## Acceptance criteria

- A busy agent can save a large batch of raw observations without hand-curating
  each one, and a later grooming pass consolidates them with source links.
- Every session recall includes a bounded, current working-memory block when a
  working summary exists; it is absent rather than misleading for other scopes.
- No gardener action can cross a session/tenant boundary, mutate a profile
  without explicit evidence, or physically delete a memory by default.
- Every derived, superseded, archived, restored, or purged memory is
  explainable from an audit record and recoverable until the configured purge
  policy actually executes.
- Entity extraction improves grouping and recall without causing uncertain
  names to be silently merged.
- Existing users who do not enable gardening see no change to save, recall,
  MCP, HTTP, or Python behaviour.
- Tests cover migrations, malformed model output, proposal validation,
  provenance, scope isolation, expiration, restore, token budgets, and MCP/HTTP
  parity.

## Decisions to settle before implementation

1. What local model/adapters are supported in the first release, and is there a
   built-in no-model deterministic grooming mode?
2. Are gardener actions automatically applied below a confidence threshold, or
   must initial releases require review for all destructive/lifecycle actions?
3. What are the default inactivity and grace periods for working memory,
   archival, and purge eligibility?
4. Which callers are allowed to run gardening or override retention policy in
   hosted deployments?
5. Should entity names be searchable metadata by default, given privacy and
   tenant-isolation requirements?
