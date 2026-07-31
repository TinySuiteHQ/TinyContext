from __future__ import annotations

from functools import lru_cache
from typing import Any


DEFAULT_ENCODING_NAME = "o200k_base"


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
