from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, NamedTuple, Sequence, TypeVar


DEFAULT_ENCODING_NAME = "o200k_base"

T = TypeVar("T")


class CharacterTokenizerAdapter:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class TiktokenAdapter:
    def __init__(self, encoding: Any) -> None:
        self._encoding = encoding

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)


def _get_tiktoken_encoding(name: str):
    try:
        import tiktoken
    except Exception:
        return None
    try:
        return TiktokenAdapter(tiktoken.get_encoding(name))
    except Exception:
        return None


@lru_cache(maxsize=16)
def resolve_tokenizer(encoding_name: str | None = DEFAULT_ENCODING_NAME):
    name = str(encoding_name or DEFAULT_ENCODING_NAME).strip() or DEFAULT_ENCODING_NAME
    encoding = _get_tiktoken_encoding(name)
    if encoding is not None:
        return encoding
    return CharacterTokenizerAdapter()


def token_count(
    text: str,
    encoding_name: str | None = DEFAULT_ENCODING_NAME,
) -> int:
    return len(resolve_tokenizer(encoding_name).encode(text))


class BudgetedSelection(NamedTuple):
    """What fit in the budget. Callers derive "what was cut" from their own
    input list -- both call sites count against the full ranked set, not just
    the window handed here, so carrying an overflow list would be misleading
    as well as unused."""

    payloads: list[dict[str, Any]]
    total_tokens: int


def select_within_budget(
    items: Sequence[T],
    *,
    max_tokens: int,
    encoding_name: str | None,
    content_of: Callable[[T], str],
    payload: Callable[[T, int, int], dict[str, Any]],
    first_rank: int = 1,
) -> BudgetedSelection:
    """Pack ``items`` into ``max_tokens``, stopping at the first that overflows.

    Order is preserved exactly: callers get a contiguous prefix of ``items``
    (top N by rank, or newest N), never a gapped subset with holes where a
    long entry was passed over. Fewer payloads than ``items`` means the
    budget ran out, and the caller is expected to say so -- silently
    returning a short list is what made a single long memory look like an
    empty store.

    An item too large for the whole budget is still returned when it lands
    first -- an over-budget answer beats an empty one.

    ``first_rank`` offsets the rank handed to ``payload``, so a paged read
    can number its results by absolute position rather than restarting at 1.
    """
    selected: list[dict[str, Any]] = []
    total_tokens = 0
    for item in items:
        content_tokens = token_count(content_of(item), encoding_name)
        if selected and total_tokens + content_tokens > max_tokens:
            break
        selected.append(payload(item, first_rank + len(selected), content_tokens))
        total_tokens += content_tokens
        if total_tokens > max_tokens:
            break
    return BudgetedSelection(selected, total_tokens)
