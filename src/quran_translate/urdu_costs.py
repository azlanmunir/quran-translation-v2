"""Frozen provider-aware cost accounting for Urdu production."""

from __future__ import annotations

import json
from typing import Any

from .config import DATA_DIR


PRICING_PATH = DATA_DIR / "evidence" / "urdu-production-pricing-v1.json"


class UrduPricingError(RuntimeError):
    """The frozen pricing record is missing or incompatible with provider usage."""


def pricing() -> dict[str, Any]:
    document = json.loads(PRICING_PATH.read_text(encoding="utf-8"))
    if document.get("version") != "urdu-production-pricing-v1":
        raise UrduPricingError("Unexpected Urdu pricing snapshot")
    return document


def usage_cost(model_id: str, usage: dict[str, Any]) -> float:
    rates = pricing()["per_million_tokens"].get(model_id)
    if not isinstance(rates, dict):
        raise UrduPricingError(f"Pricing snapshot lacks {model_id}")
    if (
        rates.get("transport") == "openrouter-standard"
        and usage.get("cost") is not None
    ):
        return round(float(usage["cost"]), 8)

    input_tokens = float(
        usage.get(
            "input_tokens",
            usage.get("prompt_tokens", usage.get("prompt_token_count", 0)),
        )
        or 0
    )
    output_tokens = float(
        usage.get(
            "output_tokens",
            usage.get("completion_tokens", usage.get("candidates_token_count", 0)),
        )
        or 0
    )
    if rates.get("transport") == "google-standard":
        output_tokens += float(usage.get("thoughts_token_count", 0) or 0)
    cache_creation = float(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read = float(
        usage.get("cache_read_input_tokens", 0)
        or usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        or usage.get("cached_content_token_count", 0)
        or 0
    )
    if rates.get("transport") == "anthropic-batch":
        cost = input_tokens * float(rates["input"])
        cost += cache_creation * float(
            rates.get("cache_creation_input", rates["input"])
        )
        cost += cache_read * float(rates.get("cached_input", rates["input"]))
    else:
        # OpenAI and Google include cached input inside their total input count.
        uncached_input = max(0.0, input_tokens - cache_read)
        cost = uncached_input * float(rates["input"])
        cost += cache_read * float(rates.get("cached_input", rates["input"]))
    cost += output_tokens * float(rates["output"])
    return round(cost / 1_000_000, 8)


def estimate_request_ceiling(
    model_id: str,
    system: str,
    user: str,
    max_output: int,
) -> float:
    rates = pricing()["per_million_tokens"].get(model_id)
    if not isinstance(rates, dict):
        raise UrduPricingError(f"Pricing snapshot lacks {model_id}")
    conservative_input_tokens = max(1, len((system + user).encode("utf-8")) // 2)
    conservative_input_rate = max(
        float(rates["input"]),
        float(rates.get("cache_creation_input", rates["input"])),
    )
    return round(
        (
            conservative_input_tokens * conservative_input_rate
            + max_output * float(rates["output"])
        )
        / 1_000_000,
        8,
    )
