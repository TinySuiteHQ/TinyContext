"""Shared synthetic-memory generator for the benchmark scripts in this directory.

Not a package module -- imported directly by sibling scripts via sys.path,
same convention as the rest of this directory.
"""

from __future__ import annotations

_TOPICS = [
    "User prefers {style} code review comments and avoids {avoid}",
    "The {service} service was migrated from {old} to {new} on {date}",
    "Known bug: {component} intermittently fails under {condition}",
    "Deployment for {env} runs on {infra} with {replicas} replicas",
    "Project deadline for {feature} is {date}, owned by {owner}",
    "Decision: use {tool} instead of {alt} for {purpose}",
    "The {team} team meets on {day} to review {topic}",
    "Customer {customer} reported an issue with {feature} on {date}",
]
_FILL = {
    "style": ["terse", "verbose", "inline", "summary-first"],
    "avoid": ["nitpicks", "style-only comments", "long threads"],
    "service": ["billing", "auth", "search", "notifications", "ingest"],
    "old": ["Postgres 12", "MySQL", "SQLite", "a flat file"],
    "new": ["Postgres 16", "CockroachDB", "SQLite+FTS5", "DynamoDB"],
    "component": ["the retry queue", "the token refresh path", "the cache layer"],
    "condition": ["high concurrency", "slow network", "cold start", "large payloads"],
    "env": ["staging", "production", "canary", "preview"],
    "infra": ["Kubernetes", "a single VM", "ECS", "bare metal"],
    "replicas": ["1", "2", "3", "5"],
    "feature": ["the dashboard", "the export flow", "SSO login", "the billing portal"],
    "owner": ["Alex", "Priya", "the platform team", "the on-call engineer"],
    "tool": ["Redis", "SQLite", "Kafka", "gRPC"],
    "alt": ["Memcached", "Postgres", "RabbitMQ", "REST"],
    "purpose": ["session storage", "job queues", "search", "internal APIs"],
    "team": ["backend", "platform", "growth", "infra"],
    "day": ["Monday", "Wednesday", "Friday"],
    "topic": ["incidents", "roadmap", "on-call handoff"],
    "customer": ["Acme Corp", "Globex", "Initech", "a beta user"],
    "date": ["2026-03-01", "2026-05-14", "2026-07-02", "2026-01-20"],
}


def synthetic_memory(index: int) -> str:
    """A deterministic, templated "filler" memory -- realistic noise content."""
    template = _TOPICS[index % len(_TOPICS)]
    values = {key: options[index % len(options)] for key, options in _FILL.items()}
    return f"{template.format(**values)} (entry #{index})"
