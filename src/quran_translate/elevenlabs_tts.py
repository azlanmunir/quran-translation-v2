"""Small ElevenLabs Text to Speech client for audio smoke tests."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .config import OUTPUT_DIR, load_dotenv


ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_AUDIO_DIR = OUTPUT_DIR / "audio"


class ElevenLabsError(RuntimeError):
    """Raised when ElevenLabs rejects a request or returns malformed data."""


def env_value(name: str) -> str | None:
    load_dotenv()
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def require_api_key() -> str:
    load_dotenv()
    api_key = env_value("ELEVENLABS_API_KEY")
    if not api_key:
        raise ElevenLabsError(
            "Missing ELEVENLABS_API_KEY. Add it to .env or export it in your shell."
        )
    return api_key


def request_json(url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "xi-api-key": api_key,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ElevenLabsError(read_error(exc)) from exc
    except urllib.error.URLError as exc:
        raise ElevenLabsError(str(exc)) from exc


def list_voices(api_key: str | None = None) -> list[dict[str, Any]]:
    payload = request_json(f"{ELEVENLABS_API_BASE}/voices", api_key or require_api_key())
    voices = payload.get("voices")
    if not isinstance(voices, list):
        raise ElevenLabsError("ElevenLabs voices response did not include a voices list.")
    return voices


def read_error(exc: urllib.error.HTTPError) -> str:
    raw = exc.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = raw
    return f"{exc.code} {exc.reason}: {payload}"


def synthesize_text(
    *,
    text: str,
    voice_id: str,
    output_path: Path,
    api_key: str | None = None,
    model_id: str = DEFAULT_ELEVENLABS_MODEL,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    previous_text: str | None = None,
    next_text: str | None = None,
    seed: int | None = None,
    voice_settings: dict[str, Any] | None = None,
    apply_text_normalization: str | None = None,
    request_timeout_seconds: int = 120,
) -> Path:
    if not text.strip():
        raise ElevenLabsError("Text to synthesize is empty.")
    if not voice_id.strip():
        raise ElevenLabsError("Missing voice_id. Pass --voice-id or set ELEVENLABS_VOICE_ID.")

    query = urllib.parse.urlencode({"output_format": output_format})
    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id.strip()}?{query}"
    body: dict[str, Any] = {
        "text": text.strip(),
        "model_id": model_id,
    }
    if previous_text:
        body["previous_text"] = previous_text
    if next_text:
        body["next_text"] = next_text
    if seed is not None:
        body["seed"] = seed
    if voice_settings:
        body["voice_settings"] = voice_settings
    if apply_text_normalization:
        body["apply_text_normalization"] = apply_text_normalization

    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key or require_api_key(),
        },
        method="POST",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
            output_path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise ElevenLabsError(read_error(exc)) from exc
    except urllib.error.URLError as exc:
        raise ElevenLabsError(str(exc)) from exc
    return output_path
