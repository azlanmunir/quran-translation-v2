"""Provider failure classification shared by Urdu pipeline stages."""

from __future__ import annotations


TERMINAL_PROVIDER_MARKERS = (
    "billing",
    "credit_balance_exhausted",
    "insufficient_quota",
    "no credits remaining",
    "authentication",
    "invalid api key",
    "invalid_api_key",
    "permission_denied",
    "unauthorized",
)


def is_terminal_provider_failure(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in TERMINAL_PROVIDER_MARKERS)
