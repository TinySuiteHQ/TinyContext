from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch


_SEMANTIC_GROUPS = (
    {"sqlite", "database", "storage", "state"},
    {"python", "backend", "fastapi"},
    {"concise", "answer", "answers", "preference", "likes"},
    {"hike", "hiking", "outdoor", "weekend", "weekends"},
    {"canine", "dog", "puppy"},
)


def fake_embed_texts(
    inputs: Sequence[str],
    **_kwargs: Any,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in inputs:
        tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
        vector = [
            float(len(tokens & group))
            for group in _SEMANTIC_GROUPS
        ]
        if not any(vector):
            vector[sum(ord(char) for char in text) % len(vector)] = 0.25
        norm = math.sqrt(sum(value * value for value in vector))
        vectors.append([value / norm for value in vector])
    return vectors


def start_fake_embeddings(test_case: Any) -> None:
    for target in (
        "tinycontext.core.embed_texts",
        "tinycontext.pipelines.memory_recall.embed_texts",
    ):
        patcher = patch(target, side_effect=fake_embed_texts)
        patcher.start()
        test_case.addCleanup(patcher.stop)
