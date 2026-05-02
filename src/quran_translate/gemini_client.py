"""Small Gemini client wrapper with retry/backoff."""

from __future__ import annotations

import os
import random
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType
from typing import Protocol

from .config import DEFAULT_MODEL, load_dotenv


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "rate",
    "quota",
    "timeout",
    "timed out",
    "deadline",
    "temporarily",
    "unavailable",
    "overloaded",
    "connection reset",
    "aborted",
)


def is_retryable_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in RETRYABLE_MARKERS)


@dataclass
class RetryConfig:
    max_attempts: int = 8
    initial_delay_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 120.0
    jitter_seconds: float = 1.0
    request_timeout_seconds: float = 180.0


class GenerationTimeoutError(TimeoutError):
    """Raised when a single model request exceeds the configured timeout."""


@contextmanager
def request_timeout(seconds: float):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def timeout_handler(signum: int, frame: FrameType | None) -> None:
        raise GenerationTimeoutError(f"Gemini call timed out after {seconds:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


class GeminiGenerator:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        api_key: str | None = None,
    ) -> None:
        load_dotenv()
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env or the environment")

        self._mode = "google-genai"
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            self._client = genai.Client(api_key=self.api_key)
            self._types = types
        except Exception:
            import google.generativeai as legacy_genai  # type: ignore

            self._mode = "google-generativeai"
            legacy_genai.configure(api_key=self.api_key)
            self._client = legacy_genai.GenerativeModel(
                model,
                generation_config={
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                },
            )
            self._types = None

    def generate(self, prompt: str) -> str:
        if self._mode == "google-genai":
            config = self._types.GenerateContentConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return response.text or ""

        response = self._client.generate_content(prompt)
        return response.text or ""


def generate_with_retry(
    generator: TextGenerator,
    prompt: str,
    retry_config: RetryConfig | None = None,
) -> tuple[str, int]:
    config = retry_config or RetryConfig()
    delay = config.initial_delay_seconds
    last_error: BaseException | None = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            with request_timeout(config.request_timeout_seconds):
                return generator.generate(prompt), attempt
        except Exception as exc:  # noqa: BLE001 - SDKs raise varied exception types.
            last_error = exc
            if attempt >= config.max_attempts or not is_retryable_error(exc):
                raise
            sleep_for = min(delay, config.max_delay_seconds)
            sleep_for += random.uniform(0, config.jitter_seconds)
            time.sleep(sleep_for)
            delay *= config.backoff_multiplier

    raise RuntimeError("Retry loop exhausted") from last_error
