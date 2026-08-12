"""Compact, blind audition of confirmed male Gemini TTS voices."""

from __future__ import annotations

import html
import json
import os
import secrets
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .audio_bakeoff import (
    CandidateSpec,
    EXPECTED_FINAL_TEXT_SHA256,
    NARRATION_DIRECTION,
    RELEASE_VERSION,
    _ffprobe,
    _generate_gemini,
    _normalize_audio,
    passage_payloads,
)
from .config import OUTPUT_DIR, file_sha256, text_sha256
from .db import utc_now
from .production_packets import atomic_json, atomic_text


AUDITION_ID = "quran-v2.4.1-google-male-voices-20260811"
DEFAULT_ROOT = OUTPUT_DIR / "audio" / "bakeoffs" / AUDITION_ID
MODEL_ID = "gemini-2.5-pro-preview-tts"
PASSAGE_ID = "consolation"

# Gender labels are from Google's Gemini-TTS voice catalog. The descriptors are
# Google's own voice labels and give this small set useful tonal contrast.
VOICES = (
    CandidateSpec(
        "gemini-pro-charon",
        "gemini",
        MODEL_ID,
        "Charon",
        "Charon - Informative (male)",
        ".wav",
        "pcm_s16le_24000_mono",
    ),
    CandidateSpec(
        "gemini-pro-algenib",
        "gemini",
        MODEL_ID,
        "Algenib",
        "Algenib - Gravelly (male)",
        ".wav",
        "pcm_s16le_24000_mono",
    ),
    CandidateSpec(
        "gemini-pro-algieba",
        "gemini",
        MODEL_ID,
        "Algieba",
        "Algieba - Smooth (male)",
        ".wav",
        "pcm_s16le_24000_mono",
    ),
    CandidateSpec(
        "gemini-pro-schedar",
        "gemini",
        MODEL_ID,
        "Schedar",
        "Schedar - Even (male)",
        ".wav",
        "pcm_s16le_24000_mono",
    ),
)


class AuditionError(RuntimeError):
    """Raised when the audition cannot be generated without mixing inputs."""


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditionError(f"Expected a JSON object: {path}")
    return payload


def _passage() -> dict[str, Any]:
    for passage in passage_payloads():
        if passage["passage_id"] == PASSAGE_ID:
            return passage
    raise AuditionError(f"Frozen release lacks passage {PASSAGE_ID}")


def _fingerprint(passage: dict[str, Any]) -> str:
    payload = {
        "version": "google-male-voice-audition-v1",
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "narration_direction": NARRATION_DIRECTION,
        "passage": passage,
        "voices": [asdict(voice) for voice in VOICES],
    }
    return text_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def prepare(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    passage = _passage()
    fingerprint = _fingerprint(passage)
    state_path = root / "RUN.json"
    if state_path.exists():
        state = _read_object(state_path)
        if state.get("input_fingerprint") != fingerprint:
            raise AuditionError("Audition inputs changed; refusing a mixed resume")
        return state

    state = {
        "version": "google-male-voice-audition-run-v1",
        "audition_id": AUDITION_ID,
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "input_fingerprint": fingerprint,
        "narration_direction_sha256": text_sha256(NARRATION_DIRECTION),
        "prepared_at": utc_now(),
        "updated_at": utc_now(),
        "status": "prepared",
        "passage": passage,
        "voices": [asdict(voice) for voice in VOICES],
        "jobs": [
            {
                "candidate_id": voice.candidate_id,
                "raw_path": str(root / "raw" / f"{voice.candidate_id}.wav"),
                "status": "pending",
                "attempts": 0,
                "last_error": None,
            }
            for voice in VOICES
        ],
    }
    atomic_json(state_path, state)
    atomic_text(root / "NARRATION_DIRECTION.txt", NARRATION_DIRECTION + "\n")
    return state


def _blind_mapping(root: Path) -> dict[str, str]:
    path = root / "PRIVATE_BLIND_KEY.json"
    candidate_ids = [voice.candidate_id for voice in VOICES]
    if path.exists():
        mapping = _read_object(path).get("candidate_to_code")
        if not isinstance(mapping, dict) or set(mapping) != set(candidate_ids):
            raise AuditionError("Blind key does not match the audition candidate set")
        return {str(key): str(value) for key, value in mapping.items()}

    codes = [f"Voice {chr(65 + index)}" for index in range(len(candidate_ids))]
    secrets.SystemRandom().shuffle(codes)
    mapping = dict(zip(candidate_ids, codes, strict=True))
    atomic_json(
        path,
        {
            "version": "google-male-voice-audition-key-v1",
            "created_at": utc_now(),
            "candidate_to_code": mapping,
        },
    )
    os.chmod(path, 0o600)
    return mapping


def _listening_html(manifest: dict[str, Any]) -> str:
    clips = json.dumps(manifest["clips"], ensure_ascii=False)
    passage = manifest["passage"]
    label = html.escape(str(passage["label"]))
    start_ref = html.escape(str(passage["start_ref"]))
    end_ref = html.escape(str(passage["end_ref"]))
    transcript = html.escape(str(passage["text"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Google Male Voice Audition</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;color:#171717;font:16px/1.5 Arial,sans-serif}}
header,main{{width:min(900px,calc(100% - 40px));margin:auto}} header{{padding:34px 0 22px;border-bottom:1px solid #ddd}}
h1{{font:600 31px Georgia,serif;margin:0 0 8px}} p{{margin:0}} .muted{{color:#666}}
main{{padding:26px 0 60px}} .transcript{{font:18px/1.6 Georgia,serif;margin:18px 0 28px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} .voice{{border:1px solid #d8d8d8;border-radius:6px;padding:16px}}
h2{{font-size:17px;margin:0 0 12px}} audio{{width:100%}} textarea{{width:100%;min-height:62px;margin-top:12px;padding:8px}}
@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header><h1>Google Male Voice Audition</h1><p class="muted">One passage, one model, four confirmed male voices. Listen for naturalness, restraint, depth, and fatigue over time.</p></header>
<main><p><strong>{label}</strong> · {start_ref}-{end_ref}</p>
<p class="transcript">{transcript}</p><div class="grid" id="grid"></div></main>
<script>
const clips={clips}; const grid=document.getElementById('grid');
for(const clip of clips){{const box=document.createElement('section');box.className='voice';box.innerHTML=`<h2>${{clip.voice_code}}</h2><audio controls preload="metadata" src="${{encodeURI(clip.file)}}"></audio><textarea placeholder="Naturalness, depth, pacing, pronunciation"></textarea>`;grid.appendChild(box)}}
</script>
</body></html>
"""


def package(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    state = _read_object(root / "RUN.json")
    if any(job["status"] != "complete" for job in state["jobs"]):
        raise AuditionError("Cannot package an incomplete audition")
    mapping = _blind_mapping(root)
    blind_dir = root / "blind"
    clips: list[dict[str, Any]] = []
    for job in state["jobs"]:
        code = mapping[job["candidate_id"]]
        destination = blind_dir / "clips" / f"{code.lower().replace(' ', '-')}.mp3"
        _normalize_audio(Path(job["raw_path"]), destination)
        clips.append(
            {
                "voice_code": code,
                "file": str(destination.relative_to(blind_dir)),
                "sha256": file_sha256(destination),
                "probe": _ffprobe(destination),
            }
        )
    manifest = {
        "version": "google-male-voice-audition-package-v1",
        "audition_id": AUDITION_ID,
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "created_at": utc_now(),
        "passage": state["passage"],
        "voice_codes": sorted(mapping.values()),
        "clips": sorted(clips, key=lambda item: item["voice_code"]),
        "normalization": "mono MP3, 44.1 kHz, 192 kbps, -18 LUFS target, -1.5 dBTP",
    }
    atomic_json(blind_dir / "BLIND_MANIFEST.json", manifest)
    atomic_text(blind_dir / "listen.html", _listening_html(manifest))
    state["status"] = "ready_for_blind_review"
    state["updated_at"] = utc_now()
    state["blind_package"] = str(blind_dir)
    atomic_json(root / "RUN.json", state)
    return manifest


def run(root: Path = DEFAULT_ROOT, *, max_attempts: int = 2) -> dict[str, Any]:
    state = prepare(root)
    voices = {
        item["candidate_id"]: CandidateSpec(**item) for item in state["voices"]
    }
    state["status"] = "running"
    atomic_json(root / "RUN.json", state)
    for job in state["jobs"]:
        if job["status"] == "complete":
            continue
        voice = voices[job["candidate_id"]]
        output_path = Path(job["raw_path"])
        for _ in range(max_attempts - int(job["attempts"])):
            job["attempts"] += 1
            started = time.monotonic()
            try:
                usage = _generate_gemini(voice, state["passage"]["text"], output_path)
                job.update(
                    {
                        "status": "complete",
                        "last_error": None,
                        "completed_at": utc_now(),
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "raw_sha256": file_sha256(output_path),
                        "provider_usage": usage,
                        "probe": _ffprobe(output_path),
                    }
                )
                break
            except Exception as exc:  # Provider SDKs expose several error families.
                job["status"] = "failed"
                job["last_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                state["updated_at"] = utc_now()
                atomic_json(root / "RUN.json", state)
        if job["status"] != "complete":
            state["status"] = "blocked"
            atomic_json(root / "RUN.json", state)
            raise AuditionError(f"Generation failed: {job['last_error']}")
    state["status"] = "generated"
    atomic_json(root / "RUN.json", state)
    package(root)
    return _read_object(root / "RUN.json")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
