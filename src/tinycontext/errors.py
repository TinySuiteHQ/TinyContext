from __future__ import annotations


class MemoryError(Exception):
    pass


class EmptyMemoryError(MemoryError):
    pass


class SessionNotFoundError(MemoryError):
    pass


class RecallBudgetError(MemoryError):
    pass


class MemoryNotFoundError(MemoryError):
    pass


MEMORY_ERROR_MAP: dict[type[Exception], tuple[str, int]] = {
    EmptyMemoryError: ("empty_memory", 400),
    SessionNotFoundError: ("session_not_found", 404),
    RecallBudgetError: ("recall_budget", 400),
    MemoryNotFoundError: ("memory_not_found", 404),
}
