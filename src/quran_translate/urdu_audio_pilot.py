"""Prepare and synthesize the bounded final Urdu Charon narration pilot."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .config import OUTPUT_DIR, PROJECT_ROOT, file_sha256, load_dotenv, text_sha256
from .db import utc_now
from .production_packets import atomic_json, atomic_text
from .urdu_release import RELEASE_ID, RELEASE_ROOT


PILOT_ID = "quran-urdu-charon-final-pilot-20260819-v1"
PILOT_ROOT = OUTPUT_DIR / "audio" / "pilots" / PILOT_ID
MODEL_ID = "gemini-2.5-pro-preview-tts"
VOICE_ID = "Charon"
MAX_ESTIMATED_STANDARD_COST_USD = 0.75
STANDARD_INPUT_USD_PER_MILLION = 1.0
STANDARD_AUDIO_USD_PER_MILLION = 20.0
BATCH_INPUT_USD_PER_MILLION = 0.5
BATCH_AUDIO_USD_PER_MILLION = 10.0
AUDIO_TOKENS_PER_SECOND = 25.0
BASELINE_TEXT_CHARS = 849
BASELINE_DIRECTION_CHARS = 303
BASELINE_PROMPT_TOKENS = 508
BASELINE_AUDIO_TOKENS = 2114
BASELINE_DURATION_SECONDS = 84.593

DIRECTION = (
    "Read only the Urdu text inside <text>. Do not read these instructions or the tags. "
    "Use natural, mature Pakistani Urdu audiobook narration with measured pacing and quiet "
    "authority. Preserve Urdu pronunciation, izafat, and Quranic Arabic names. Do not drift "
    "into Hindi diction, theatrical recitation, or a promotional voice. Read every word exactly. "
    "Pause briefly at each line break. Pronounce Quranic disconnected letters separately by "
    "their Arabic letter names rather than as one word."
)


@dataclass(frozen=True)
class PassageSpec:
    passage_id: str
    label: str
    ranges: tuple[tuple[int, int, int], ...]
    purpose: str


PASSAGES = (
    PassageSpec("fatihah", "Al-Fatihah", ((1, 1, 7),), "opening, basmala, and devotional restraint"),
    PassageSpec("taqwa-law", "Taqwa and fasting law", ((2, 177, 187),), "long legal syntax and terminology consistency"),
    PassageSpec("yusuf-dialogue", "Yusuf dialogue", ((12, 23, 30),), "dialogue, intimacy, and narrative pacing"),
    PassageSpec("maryam-names", "Maryam and Zachariah", ((19, 1, 15),), "disconnected letters, names, tenderness, and metaphor"),
    PassageSpec(
        "final-adjudications",
        "Final adjudication audit",
        ((17, 7, 7), (45, 11, 11), (53, 19, 23), (66, 1, 5)),
        "all final human-adjudicated passages",
    ),
    PassageSpec("force-and-consolation", "At-Takwir and Ad-Duha", ((81, 1, 14), (93, 1, 11)), "apocalyptic force followed by consolation"),
)

SPLIT_REGISTER_PASSAGES = PASSAGES[:-1] + (
    PassageSpec("force-takwir", "At-Takwir", ((81, 1, 14),), "apocalyptic force and compressed cadence"),
    PassageSpec("consolation-duha", "Ad-Duha", ((93, 1, 11),), "consolation, warmth, and restraint"),
)

PRONUNCIATION_OVERRIDES = {
    "19:1": {
        "source": "کٓھٰیٰعٓصٓ۔",
        "spoken": "کاف، ہا، یا، عین، صاد۔",
        "rationale": "Speak the Quranic disconnected letters by their Arabic letter names.",
    }
}


class UrduAudioPilotError(RuntimeError):
    """Raised when pilot inputs, budget, provider, or audio integrity fail."""


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UrduAudioPilotError(f"Expected JSON object: {path}")
    return payload


def _read_rows() -> list[dict[str, Any]]:
    path = RELEASE_ROOT / "quran-urdu.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 6236:
        raise UrduAudioPilotError("Frozen Urdu release does not contain 6,236 ayahs")
    return [dict(row) for row in payload]


def _passage_payloads(
    rows: list[dict[str, Any]], specs: tuple[PassageSpec, ...] = PASSAGES
) -> list[dict[str, Any]]:
    by_ref = {(int(row["surah"]), int(row["ayah"])): str(row["urdu"]) for row in rows}
    passages = []
    for spec in specs:
        refs: list[str] = []
        lines: list[str] = []
        speech_lines: list[str] = []
        for surah, first, last in spec.ranges:
            for ayah in range(first, last + 1):
                key = (surah, ayah)
                if key not in by_ref:
                    raise UrduAudioPilotError(f"Pilot source is missing {surah}:{ayah}")
                ref = f"{surah}:{ayah}"
                refs.append(ref)
                lines.append(by_ref[key])
                override = PRONUNCIATION_OVERRIDES.get(ref)
                if override:
                    if by_ref[key] != override["source"]:
                        raise UrduAudioPilotError(
                            f"Pronunciation override source guard failed at {ref}"
                        )
                    speech_lines.append(override["spoken"])
                else:
                    speech_lines.append(by_ref[key])
        text = "\n".join(lines)
        speech_text = "\n".join(speech_lines)
        passages.append(
            {
                **asdict(spec),
                "ranges": [list(item) for item in spec.ranges],
                "refs": refs,
                "text": text,
                "speech_text": speech_text,
                "characters": len(text),
                "text_sha256": text_sha256(text),
                "speech_text_sha256": text_sha256(speech_text),
            }
        )
    return passages


def _estimate(passages: list[dict[str, Any]], full_chars: int) -> dict[str, Any]:
    pilot_chars = sum(int(item["characters"]) for item in passages)
    seconds_per_char = BASELINE_DURATION_SECONDS / BASELINE_TEXT_CHARS
    pilot_seconds = pilot_chars * seconds_per_char
    full_seconds = full_chars * seconds_per_char
    prompt_material = pilot_chars + len(passages) * len(DIRECTION)
    baseline_material = BASELINE_TEXT_CHARS + 3 * BASELINE_DIRECTION_CHARS
    prompt_tokens = math.ceil(BASELINE_PROMPT_TOKENS * prompt_material / baseline_material)

    def costs(duration_seconds: float, input_tokens: float) -> dict[str, float]:
        output_tokens = duration_seconds * AUDIO_TOKENS_PER_SECOND
        return {
            "input_tokens": round(input_tokens, 2),
            "audio_tokens": round(output_tokens, 2),
            "standard_usd": round(
                input_tokens / 1_000_000 * STANDARD_INPUT_USD_PER_MILLION
                + output_tokens / 1_000_000 * STANDARD_AUDIO_USD_PER_MILLION,
                6,
            ),
            "batch_usd": round(
                input_tokens / 1_000_000 * BATCH_INPUT_USD_PER_MILLION
                + output_tokens / 1_000_000 * BATCH_AUDIO_USD_PER_MILLION,
                6,
            ),
        }

    full_input_tokens = BASELINE_PROMPT_TOKENS * full_chars / BASELINE_TEXT_CHARS
    return {
        "version": "quran-urdu-charon-cost-estimate-v1",
        "pricing_checked_on": "2026-08-19",
        "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
        "model_id": MODEL_ID,
        "voice_id": VOICE_ID,
        "rates_per_million_tokens_usd": {
            "standard_input": STANDARD_INPUT_USD_PER_MILLION,
            "standard_audio": STANDARD_AUDIO_USD_PER_MILLION,
            "batch_input": BATCH_INPUT_USD_PER_MILLION,
            "batch_audio": BATCH_AUDIO_USD_PER_MILLION,
            "audio_tokens_per_second": AUDIO_TOKENS_PER_SECOND,
        },
        "empirical_baseline": {
            "text_characters": BASELINE_TEXT_CHARS,
            "duration_seconds": BASELINE_DURATION_SECONDS,
            "prompt_tokens": BASELINE_PROMPT_TOKENS,
            "audio_tokens": BASELINE_AUDIO_TOKENS,
        },
        "pilot": {
            "clips": len(passages),
            "text_characters": pilot_chars,
            "estimated_duration_seconds": round(pilot_seconds, 2),
            "estimated_duration_minutes": round(pilot_seconds / 60, 2),
            **costs(pilot_seconds, prompt_tokens),
            "hard_estimated_standard_cost_ceiling_usd": MAX_ESTIMATED_STANDARD_COST_USD,
        },
        "full_book": {
            "text_characters": full_chars,
            "estimated_duration_seconds": round(full_seconds, 2),
            "estimated_duration_hours": round(full_seconds / 3600, 2),
            **costs(full_seconds, full_input_tokens),
            "uncertainty_note": "Actual duration and cost will be recalibrated from this final-text pilot before full synthesis.",
        },
    }


def _fingerprint(passages: list[dict[str, Any]]) -> str:
    selection = _read_object(RELEASE_ROOT / "tts-selection.json")
    release = _read_object(RELEASE_ROOT / "MANIFEST.json")
    payload = {
        "version": "quran-urdu-charon-final-pilot-v1",
        "release_id": RELEASE_ID,
        "release_text_sha256": release["artifact_sha256"]["quran-urdu.json"],
        "selection": selection,
        "model_id": MODEL_ID,
        "voice_id": VOICE_ID,
        "direction": DIRECTION,
        "passages": passages,
    }
    return text_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def prepare(root: Path = PILOT_ROOT) -> dict[str, Any]:
    if not (RELEASE_ROOT / "MANIFEST.json").exists():
        raise UrduAudioPilotError("Create the frozen Urdu release before preparing audio")
    state_path = root / "RUN.json"
    existing_state = _read_object(state_path) if state_path.exists() else None
    rows = _read_rows()
    specs = (
        SPLIT_REGISTER_PASSAGES
        if existing_state and existing_state.get("register_split")
        else PASSAGES
    )
    passages = _passage_payloads(rows, specs)
    estimate = _estimate(passages, sum(len(str(row["urdu"])) for row in rows))
    if estimate["pilot"]["standard_usd"] > MAX_ESTIMATED_STANDARD_COST_USD:
        raise UrduAudioPilotError("Pilot estimate exceeds the hard cost ceiling")
    for passage in passages:
        estimated_seconds = passage["characters"] * BASELINE_DURATION_SECONDS / BASELINE_TEXT_CHARS
        if estimated_seconds * AUDIO_TOKENS_PER_SECOND >= 14_000:
            raise UrduAudioPilotError(f"Pilot clip is too close to the model output limit: {passage['passage_id']}")
    fingerprint = _fingerprint(passages)
    if existing_state is not None:
        state = existing_state
        if state.get("input_fingerprint") != fingerprint:
            pristine = all(
                job.get("status") == "pending" and int(job.get("attempts", 0)) == 0
                for job in state.get("jobs", [])
            )
            if not pristine:
                raise UrduAudioPilotError("Pilot inputs changed; refusing a mixed resume")
            state["input_fingerprint"] = fingerprint
            state["updated_at"] = utc_now()
            atomic_json(root / "PASSAGES.json", passages)
            atomic_json(root / "COST_ESTIMATE.json", estimate)
            atomic_json(state_path, state)
        return state
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "PASSAGES.json", passages)
    atomic_json(root / "COST_ESTIMATE.json", estimate)
    state = {
        "version": "quran-urdu-charon-pilot-run-v1",
        "pilot_id": PILOT_ID,
        "release_id": RELEASE_ID,
        "input_fingerprint": fingerprint,
        "model_id": MODEL_ID,
        "voice_id": VOICE_ID,
        "direction": DIRECTION,
        "status": "prepared",
        "prepared_at": utc_now(),
        "updated_at": utc_now(),
        "hard_estimated_standard_cost_ceiling_usd": MAX_ESTIMATED_STANDARD_COST_USD,
        "jobs": [
            {
                "job_id": passage["passage_id"],
                "status": "pending",
                "attempts": 0,
                "raw_path": str(root / "raw" / f"{passage['passage_id']}.wav"),
                "normalized_path": str(root / "clips" / f"{passage['passage_id']}.mp3"),
                "last_error": None,
            }
            for passage in passages
        ],
    }
    atomic_json(state_path, state)
    return state


def _gemini_audio(response: Any) -> bytes:
    try:
        data = response.candidates[0].content.parts[0].inline_data.data
    except (AttributeError, IndexError, TypeError) as exc:
        raise UrduAudioPilotError("Gemini response did not contain audio") from exc
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    raise UrduAudioPilotError("Gemini returned unsupported audio data")


def _write_wav(path: Path, audio: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with wave.open(str(temp), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(audio)
    os.replace(temp, path)


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=codec_name,sample_rate,channels",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = (payload.get("streams") or [{}])[0]
    format_data = payload.get("format") or {}
    return {
        "duration_seconds": round(float(format_data.get("duration") or 0), 3),
        "bytes": int(format_data.get("size") or path.stat().st_size),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def _normalize(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp.mp3")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-af", "loudnorm=I=-18:TP=-1.5:LRA=11", "-ar", "44100", "-ac", "1",
            "-codec:a", "libmp3lame", "-b:a", "192k", str(temp),
        ],
        check=True,
        capture_output=True,
    )
    os.replace(temp, destination)


def _review_html(passages: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> str:
    job_map = {job["job_id"]: job for job in jobs}
    sections = []
    for passage in passages:
        job = job_map[passage["passage_id"]]
        relative = Path(job["normalized_path"]).relative_to(PILOT_ROOT)
        sections.append(
            f'<section><p class="purpose">{passage["purpose"]}</p>'
            f'<h2>{passage["label"]}</h2><p class="refs">{passage["refs"][0]} to {passage["refs"][-1]}</p>'
            f'<audio controls preload="metadata" src="{relative.as_posix()}"></audio>'
            f'<p class="urdu" lang="ur">{passage["text"].replace(chr(10), "<br>")}</p></section>'
        )
    return """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Final Urdu Charon Pilot</title>
<style>body{margin:0;color:#17201c;font-family:Arial,sans-serif}header,main{width:min(960px,calc(100% - 32px));margin:auto}header{padding:28px 0 20px;border-bottom:1px solid #cfd7d1}h1{font:600 30px Georgia,serif;margin:0 0 8px}header p,.purpose,.refs{color:#5e6963}section{padding:25px 0;border-bottom:1px solid #cfd7d1}h2{font:600 22px Georgia,serif;margin:3px 0}.purpose{text-transform:uppercase;font-size:11px;font-weight:700;color:#983f2e}.refs{font-size:12px}audio{width:100%;margin:9px 0}.urdu{direction:rtl;text-align:right;font-family:"Noto Nastaliq Urdu",serif;font-size:21px;line-height:2.15}</style></head><body><header><h1>Final Urdu Charon Pilot</h1><p>Judge exact wording first, then Urdu pronunciation, pacing, restraint, and long-form comfort.</p></header><main>""" + "\n".join(sections) + "</main></body></html>\n"


def _recalibrated_full_estimate(
    state: dict[str, Any], passages: list[dict[str, Any]]
) -> dict[str, Any]:
    release = _read_object(RELEASE_ROOT / "MANIFEST.json")
    full_chars = int(release["urdu_characters"])
    pilot_chars = sum(int(item["characters"]) for item in passages)
    duration = float(state["generated_duration_seconds"])
    full_duration = duration / pilot_chars * full_chars
    prompt_tokens = sum(
        int(job.get("provider_usage", {}).get("prompt_tokens", 0))
        for job in state["jobs"]
    )
    prompt_material = pilot_chars + len(passages) * len(DIRECTION)
    prompt_tokens_per_material_char = prompt_tokens / prompt_material
    assumed_jobs = 313
    full_prompt_tokens = prompt_tokens_per_material_char * (
        full_chars + assumed_jobs * len(DIRECTION)
    )
    full_audio_tokens = full_duration * AUDIO_TOKENS_PER_SECOND
    standard = (
        full_prompt_tokens / 1_000_000 * STANDARD_INPUT_USD_PER_MILLION
        + full_audio_tokens / 1_000_000 * STANDARD_AUDIO_USD_PER_MILLION
    )
    batch = (
        full_prompt_tokens / 1_000_000 * BATCH_INPUT_USD_PER_MILLION
        + full_audio_tokens / 1_000_000 * BATCH_AUDIO_USD_PER_MILLION
    )
    return {
        "version": "quran-urdu-charon-recalibrated-full-estimate-v1",
        "pilot_id": PILOT_ID,
        "model_id": MODEL_ID,
        "voice_id": VOICE_ID,
        "calibrated_at": utc_now(),
        "successful_pilot": {
            "clips": len(passages),
            "text_characters": pilot_chars,
            "duration_seconds": duration,
            "recorded_standard_cost_usd": state["actual_standard_cost_usd"],
            "prompt_tokens": prompt_tokens,
            "audio_tokens": sum(
                int(job.get("provider_usage", {}).get("output_tokens", 0))
                for job in state["jobs"]
            ),
        },
        "full_book": {
            "text_characters": full_chars,
            "assumed_synthesis_jobs": assumed_jobs,
            "estimated_duration_seconds": round(full_duration, 2),
            "estimated_duration_hours": round(full_duration / 3600, 2),
            "estimated_prompt_tokens": round(full_prompt_tokens),
            "estimated_audio_tokens": round(full_audio_tokens),
            "estimated_standard_cost_usd": round(standard, 2),
            "estimated_batch_cost_usd": round(batch, 2),
            "planning_range_standard_usd": [round(standard * 0.9, 2), round(standard * 1.15, 2)],
            "planning_range_batch_usd": [round(batch * 0.9, 2), round(batch * 1.15, 2)],
        },
        "billing_note": "Three empty-audio responses have no reported usage in the local receipt; any input-only provider charge for them is not included.",
    }


def _write_final_sidecars(
    root: Path, state: dict[str, Any], passages: list[dict[str, Any]]
) -> None:
    atomic_json(root / "RECALIBRATED_FULL_PRODUCTION_ESTIMATE.json", _recalibrated_full_estimate(state, passages))
    atomic_json(
        root / "QA_REPORT.json",
        {
            "version": "quran-urdu-charon-pilot-qa-v1",
            "pilot_id": PILOT_ID,
            "technical_passed": True,
            "human_listening_status": "pending",
            "release_gate": "PENDING_HUMAN_LISTENING",
            "clips": len(state["jobs"]),
            "duration_seconds": state["generated_duration_seconds"],
            "actual_standard_cost_usd": state["actual_standard_cost_usd"],
            "audio_contract": "MP3, 44.1 kHz, mono, 192 kbps, -18 LUFS target",
            "speech_normalization": state.get("speech_normalization"),
            "superseded_jobs": state.get("superseded_jobs", []),
            "jobs": state["jobs"],
        },
    )
    atomic_text(root / "review.html", _review_html(passages, state["jobs"]))


def run(root: Path = PILOT_ROOT, *, allow_recovery: bool = False) -> dict[str, Any]:
    state = prepare(root)
    if state.get("status") == "complete":
        passages = json.loads((root / "PASSAGES.json").read_text(encoding="utf-8"))
        _write_final_sidecars(root, state, passages)
        return state
    if any(job["status"] == "failed" for job in state["jobs"]) and not allow_recovery:
        raise UrduAudioPilotError("Pilot has a preserved failed job; refusing an automatic retry")
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise UrduAudioPilotError("Missing GOOGLE_API_KEY")
    passages = json.loads((root / "PASSAGES.json").read_text(encoding="utf-8"))
    by_id = {passage["passage_id"]: passage for passage in passages}
    estimate = _read_object(root / "COST_ESTIMATE.json")
    if float(estimate["pilot"]["standard_usd"]) > MAX_ESTIMATED_STANDARD_COST_USD:
        raise UrduAudioPilotError("Pilot estimate exceeds the hard cost ceiling")
    client = genai.Client(api_key=api_key)
    state["status"] = "running"
    atomic_json(root / "RUN.json", state)
    for job in state["jobs"]:
        if job["status"] == "complete":
            continue
        passage = by_id[job["job_id"]]
        spent = sum(float(item.get("actual_standard_cost_usd", 0)) for item in state["jobs"])
        estimated_clip_cost = (
            float(estimate["pilot"]["standard_usd"])
            * int(passage["characters"])
            / int(estimate["pilot"]["text_characters"])
        )
        if spent + estimated_clip_cost * 1.25 > MAX_ESTIMATED_STANDARD_COST_USD:
            raise UrduAudioPilotError(
                f"Next clip would exceed the guarded pilot cost ceiling: {job['job_id']}"
            )
        job["attempts"] += 1
        started = time.monotonic()
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=f"{DIRECTION}\n\n<text>{passage['speech_text']}</text>",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_ID)
                        )
                    ),
                ),
            )
            raw = Path(job["raw_path"])
            normalized = Path(job["normalized_path"])
            _write_wav(raw, _gemini_audio(response))
            _normalize(raw, normalized)
            probe = _probe(normalized)
            empirical = probe["duration_seconds"] / max(1, int(passage["characters"]))
            if probe["duration_seconds"] < 10 or not 0.04 <= empirical <= 0.22:
                raise UrduAudioPilotError(
                    f"Implausible generated duration for {job['job_id']}: {probe['duration_seconds']}s"
                )
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            if output_tokens <= 0:
                output_tokens = math.ceil(
                    probe["duration_seconds"] * AUDIO_TOKENS_PER_SECOND
                )
            job.update(
                {
                    "status": "complete",
                    "provider_usage": {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens},
                    "actual_standard_cost_usd": round(
                        prompt_tokens / 1_000_000 * STANDARD_INPUT_USD_PER_MILLION
                        + output_tokens / 1_000_000 * STANDARD_AUDIO_USD_PER_MILLION,
                        6,
                    ),
                    "probe": probe,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "raw_sha256": file_sha256(raw),
                    "normalized_sha256": file_sha256(normalized),
                    "last_error": None,
                }
            )
        except Exception as exc:
            job["status"] = "failed"
            job["last_error"] = f"{type(exc).__name__}: {exc}"
            state["status"] = "blocked"
        finally:
            state["updated_at"] = utc_now()
            atomic_json(root / "RUN.json", state)
        if job["status"] != "complete":
            raise UrduAudioPilotError(f"Pilot stopped at {job['job_id']}: {job['last_error']}")
    state["status"] = "complete"
    state["completed_at"] = utc_now()
    state["actual_standard_cost_usd"] = round(
        sum(float(job.get("actual_standard_cost_usd", 0)) for job in state["jobs"]), 6
    )
    state["generated_duration_seconds"] = round(
        sum(float(job["probe"]["duration_seconds"]) for job in state["jobs"]), 3
    )
    state["generated_bytes"] = sum(int(job["probe"]["bytes"]) for job in state["jobs"])
    atomic_json(root / "RUN.json", state)
    _write_final_sidecars(root, state, passages)
    return state


def recover_empty_audio(root: Path = PILOT_ROOT) -> dict[str, Any]:
    state = _read_object(root / "RUN.json")
    failed = [job for job in state.get("jobs", []) if job.get("status") == "failed"]
    if len(failed) != 1:
        raise UrduAudioPilotError("Recovery requires exactly one failed pilot job")
    job = failed[0]
    if int(job.get("attempts", 0)) != 1 or "did not contain audio" not in str(
        job.get("last_error", "")
    ):
        raise UrduAudioPilotError("Failed job is not eligible for empty-audio recovery")
    if Path(job["raw_path"]).exists() or Path(job["normalized_path"]).exists():
        raise UrduAudioPilotError("Failed job unexpectedly has audio; refusing recovery")
    job.setdefault("failure_history", []).append(
        {
            "attempt": 1,
            "error": job["last_error"],
            "preserved_at": utc_now(),
        }
    )
    job["status"] = "pending"
    job["last_error"] = None
    state["status"] = "recovery_prepared"
    state["updated_at"] = utc_now()
    atomic_json(root / "RUN.json", state)
    return run(root, allow_recovery=True)


def recover_disconnected_letters(root: Path = PILOT_ROOT) -> dict[str, Any]:
    state = _read_object(root / "RUN.json")
    failed = [job for job in state.get("jobs", []) if job.get("status") == "failed"]
    if len(failed) != 1 or failed[0].get("job_id") != "maryam-names":
        raise UrduAudioPilotError("Disconnected-letter recovery requires failed maryam-names")
    job = failed[0]
    if int(job.get("attempts", 0)) != 2 or "did not contain audio" not in str(
        job.get("last_error", "")
    ):
        raise UrduAudioPilotError("Maryam failure is not eligible for pronunciation recovery")
    if Path(job["raw_path"]).exists() or Path(job["normalized_path"]).exists():
        raise UrduAudioPilotError("Failed Maryam job unexpectedly has audio")
    rows = _read_rows()
    passages = _passage_payloads(rows)
    estimate = _estimate(passages, sum(len(str(row["urdu"])) for row in rows))
    job.setdefault("failure_history", []).append(
        {
            "attempt": 2,
            "error": job["last_error"],
            "preserved_at": utc_now(),
        }
    )
    job["status"] = "pending"
    job["last_error"] = None
    state["input_fingerprint"] = _fingerprint(passages)
    state["speech_normalization"] = {
        "version": "quran-urdu-tts-pronunciation-v1",
        "overrides": PRONUNCIATION_OVERRIDES,
    }
    state["status"] = "pronunciation_recovery_prepared"
    state["updated_at"] = utc_now()
    atomic_json(root / "PASSAGES.json", passages)
    atomic_json(root / "COST_ESTIMATE.json", estimate)
    atomic_json(root / "RUN.json", state)
    return run(root, allow_recovery=True)


def recover_split_registers(root: Path = PILOT_ROOT) -> dict[str, Any]:
    state = _read_object(root / "RUN.json")
    failed = [job for job in state.get("jobs", []) if job.get("status") == "failed"]
    if len(failed) != 1 or failed[0].get("job_id") != "force-and-consolation":
        raise UrduAudioPilotError("Register split requires failed force-and-consolation")
    parent = failed[0]
    if int(parent.get("attempts", 0)) != 1 or "did not contain audio" not in str(
        parent.get("last_error", "")
    ):
        raise UrduAudioPilotError("Combined register failure is not eligible for splitting")
    if Path(parent["raw_path"]).exists() or Path(parent["normalized_path"]).exists():
        raise UrduAudioPilotError("Failed combined register job unexpectedly has audio")
    rows = _read_rows()
    passages = _passage_payloads(rows, SPLIT_REGISTER_PASSAGES)
    estimate = _estimate(passages, sum(len(str(row["urdu"])) for row in rows))
    state.setdefault("superseded_jobs", []).append(
        {
            **parent,
            "superseded_at": utc_now(),
            "superseded_by": ["force-takwir", "consolation-duha"],
        }
    )
    state["jobs"] = [
        job for job in state["jobs"] if job.get("job_id") != "force-and-consolation"
    ]
    for passage_id in ("force-takwir", "consolation-duha"):
        state["jobs"].append(
            {
                "job_id": passage_id,
                "status": "pending",
                "attempts": 0,
                "raw_path": str(root / "raw" / f"{passage_id}.wav"),
                "normalized_path": str(root / "clips" / f"{passage_id}.mp3"),
                "last_error": None,
            }
        )
    state["input_fingerprint"] = _fingerprint(passages)
    state["register_split"] = {
        "version": "quran-urdu-pilot-register-split-v1",
        "source_job": "force-and-consolation",
        "replacement_jobs": ["force-takwir", "consolation-duha"],
    }
    state["status"] = "register_split_recovery_prepared"
    state["updated_at"] = utc_now()
    atomic_json(root / "PASSAGES.json", passages)
    atomic_json(root / "COST_ESTIMATE.json", estimate)
    atomic_json(root / "RUN.json", state)
    return run(root, allow_recovery=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "run",
            "recover-empty-audio",
            "recover-disconnected-letters",
            "recover-split-registers",
        ),
        nargs="?",
        default="prepare",
    )
    args = parser.parse_args()
    if args.action == "prepare":
        state = prepare()
    elif args.action == "recover-empty-audio":
        state = recover_empty_audio()
    elif args.action == "recover-disconnected-letters":
        state = recover_disconnected_letters()
    elif args.action == "recover-split-registers":
        state = recover_split_registers()
    else:
        state = run()
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
