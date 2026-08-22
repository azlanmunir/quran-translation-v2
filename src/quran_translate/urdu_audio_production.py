"""Produce the frozen Urdu Quran audiobook with Gemini Batch TTS."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .config import DATA_DIR, OUTPUT_DIR, file_sha256, load_dotenv, text_sha256
from .db import utc_now
from .production_packets import atomic_json, atomic_text
from .urdu_audio_pilot import (
    AUDIO_TOKENS_PER_SECOND,
    BATCH_AUDIO_USD_PER_MILLION,
    BATCH_INPUT_USD_PER_MILLION,
    DIRECTION,
    MODEL_ID,
    PILOT_ROOT,
    VOICE_ID,
)
from .urdu_release import RELEASE_ID, RELEASE_ROOT


PRODUCTION_ID = "quran-urdu-charon-production-v1"
PRODUCTION_ROOT = DATA_DIR / "work" / "audio-urdu-v1" / PRODUCTION_ID
PRE_BOUNDARY_RUN_ROOT = PRODUCTION_ROOT.with_name(
    f"{PRODUCTION_ID}.pre-para-boundary-correction-20260819"
)
RELEASE_AUDIO_ROOT = OUTPUT_DIR / "audio" / "releases" / PRODUCTION_ID
TRANSLATION_RUN_ROOT = (
    DATA_DIR / "work" / "urdu-production-v1" / "quran-urdu-production-v1-20260818"
)
JUZ_BOUNDARIES = DATA_DIR / "evidence" / "juz-boundaries-v1.json"
HARD_ESTIMATED_BATCH_COST_USD = 20.0
SHARD_TARGET_CHARACTERS = 30_000
FRAGMENT_TARGET_CHARACTERS = 900
FRAGMENT_MAX_AYAHS = 6
POLL_SECONDS = 30
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


# These substitutions affect speech only. Every source string is exact-guarded.
PRONUNCIATION_OVERRIDES = {
    "2:1": ("الم۔", "الف، لام، میم۔"),
    "3:1": ("الم۔", "الف، لام، میم۔"),
    "7:1": ("المص", "الف، لام، میم، صاد۔"),
    "10:1": ("الر،", "الف، لام، را۔"),
    "11:1": ("الر —", "الف، لام، را۔"),
    "13:1": ("المر۔", "الف، لام، میم، را۔"),
    "14:1": ("الر —", "الف، لام، را۔"),
    "19:1": ("کٓھٰیٰعٓصٓ۔", "کاف، ہا، یا، عین، صاد۔"),
    "20:1": ("طہ", "طا، ہا۔"),
    "26:1": ("طسم۔", "طا، سین، میم۔"),
    "27:1": ("طٰسٓ،", "طا، سین۔"),
    "28:1": ("طسم۔", "طا، سین، میم۔"),
    "29:1": ("الم۔", "الف، لام، میم۔"),
    "30:1": ("الم۔", "الف، لام، میم۔"),
    "31:1": ("الم۔", "الف، لام، میم۔"),
    "32:1": ("الم۔", "الف، لام، میم۔"),
    "36:1": ("یٰسٓ۔", "یا، سین۔"),
    "38:1": ("ص،", "صاد۔"),
    "40:1": ("حٰم۔", "حا، میم۔"),
    "41:1": ("حٰم۔", "حا، میم۔"),
    "42:1": ("حٰم۔", "حا، میم۔"),
    "42:2": ("عسق۔", "عین، سین، قاف۔"),
    "43:1": ("حٰم۔", "حا، میم۔"),
    "44:1": ("حٰم۔", "حا، میم۔"),
    "45:1": ("حٰم۔", "حا، میم۔"),
    "46:1": ("حٰم۔", "حا، میم۔"),
    "50:1": ("ق،", "قاف۔"),
    "68:1": ("ن،", "نون۔"),
}


class UrduAudioProductionError(RuntimeError):
    """Raised when production would violate a cost or integrity guard."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise UrduAudioProductionError(f"Expected JSON object: {path}")
    return value


def _speech_ayah(ref: str, source: str) -> str:
    override = PRONUNCIATION_OVERRIDES.get(ref)
    if not override:
        return source
    prefix, spoken = override
    if not source.startswith(prefix):
        raise UrduAudioProductionError(
            f"Pronunciation override source guard failed at {ref}: {source!r}"
        )
    return spoken + source[len(prefix) :]


def _source_rows() -> list[dict[str, Any]]:
    rows = _read_json(RELEASE_ROOT / "quran-urdu.json")
    if not isinstance(rows, list) or len(rows) != 6_236:
        raise UrduAudioProductionError("Frozen Urdu release must contain 6,236 ayahs")
    return [dict(row) for row in rows]


def _build_units() -> list[dict[str, Any]]:
    rows = _source_rows()
    source_units = _read_json(TRANSLATION_RUN_ROOT / "UNITS.json")
    if not isinstance(source_units, list) or len(source_units) != 323:
        raise UrduAudioProductionError("Frozen translation run must contain 323 units")
    by_ref = {
        (int(row["surah"]), int(row["ayah"])): str(row["urdu"])
        for row in rows
    }
    ordered_refs = [(int(row["surah"]), int(row["ayah"])) for row in rows]
    positions = {ref: index for index, ref in enumerate(ordered_refs)}
    boundary_payload = _read_object(JUZ_BOUNDARIES)
    raw_juzs = boundary_payload.get("juzs")
    if not isinstance(raw_juzs, list) or len(raw_juzs) != 30:
        raise UrduAudioProductionError("Canonical Para metadata must contain 30 ranges")
    juz_by_ref: dict[tuple[int, int], int] = {}
    previous_end = -1
    for expected_juz, raw in enumerate(raw_juzs, start=1):
        start = tuple(int(value) for value in str(raw["start_ref"]).split(":"))
        end = tuple(int(value) for value in str(raw["end_ref"]).split(":"))
        if start not in positions or end not in positions:
            raise UrduAudioProductionError(f"Para {expected_juz} boundary is absent")
        first_position = positions[start]
        last_position = positions[end]
        if first_position != previous_end + 1 or last_position < first_position:
            raise UrduAudioProductionError(f"Para {expected_juz} is not contiguous")
        for ref in ordered_refs[first_position : last_position + 1]:
            juz_by_ref[ref] = expected_juz
        previous_end = last_position
    if previous_end != len(ordered_refs) - 1 or len(juz_by_ref) != 6_236:
        raise UrduAudioProductionError("Canonical Para ranges do not cover the Quran")

    units: list[dict[str, Any]] = []
    covered: list[tuple[int, int]] = []
    for source_unit in source_units:
        surah = int(source_unit["surah"])
        first = int(source_unit["first_ayah"])
        last = int(source_unit["last_ayah"])
        source_refs = [(surah, ayah) for ayah in range(first, last + 1)]
        segments: list[list[tuple[int, int]]] = []
        for ref in source_refs:
            if ref not in by_ref:
                raise UrduAudioProductionError(f"Missing frozen Urdu ayah {ref}")
            if not segments or juz_by_ref[ref] != juz_by_ref[segments[-1][-1]]:
                segments.append([])
            segments[-1].append(ref)
        for refs in segments:
            lines = [by_ref[ref] for ref in refs]
            speech_lines = [
                _speech_ayah(f"{s}:{a}", by_ref[(s, a)]) for s, a in refs
            ]
            text = "\n".join(lines)
            speech_text = "\n".join(speech_lines)
            unit_id = f"s{surah:03d}_{refs[0][1]:03d}_{refs[-1][1]:03d}"
            units.append(
                {
                    "unit_index": len(units) + 1,
                    "unit_id": unit_id,
                    "source_unit_id": str(source_unit["unit_id"]),
                    "surah": surah,
                    "juz": juz_by_ref[refs[0]],
                    "first_ayah": refs[0][1],
                    "last_ayah": refs[-1][1],
                    "refs": [f"{s}:{a}" for s, a in refs],
                    "text": text,
                    "speech_text": speech_text,
                    "characters": len(text),
                    "speech_characters": len(speech_text),
                    "text_sha256": text_sha256(text),
                    "speech_text_sha256": text_sha256(speech_text),
                }
            )
            covered.extend(refs)
    expected_refs = [(int(row["surah"]), int(row["ayah"])) for row in rows]
    if covered != expected_refs or len(set(covered)) != 6_236:
        raise UrduAudioProductionError("Audio units do not exactly cover the frozen release")
    return units


def _make_shards(
    units: list[dict[str, Any]], *, seeded_units: int = 1
) -> list[list[dict[str, Any]]]:
    # Fatihah is seeded from the approved pilot. The next unit is isolated as the
    # real Batch transport canary before the remaining jobs are submitted.
    pending = units[seeded_units:]
    if not pending:
        return []
    shards = [[pending[0]]] if seeded_units == 1 else []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for unit in (pending[1:] if seeded_units == 1 else pending):
        characters = int(unit["speech_characters"])
        if current and current_chars + characters > SHARD_TARGET_CHARACTERS:
            shards.append(current)
            current = []
            current_chars = 0
        current.append(unit)
        current_chars += characters
    if current:
        shards.append(current)
    return shards


def _generation_config() -> dict[str, Any]:
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            language_code="ur",
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_ID)
            ),
        ),
    )
    return config.model_dump(by_alias=True, exclude_none=True)


def _request_line(unit: dict[str, Any]) -> str:
    request = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"{DIRECTION}\n\n<text>{unit['speech_text']}</text>"
                    }
                ],
            }
        ],
        "generationConfig": _generation_config(),
    }
    return json.dumps(
        {"key": unit["unit_id"], "request": request},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _fingerprint(units: list[dict[str, Any]]) -> str:
    release = _read_object(RELEASE_ROOT / "MANIFEST.json")
    pilot_approval = _read_object(PILOT_ROOT / "PILOT_APPROVED.json")
    payload = {
        "version": "quran-urdu-charon-production-input-v2",
        "production_id": PRODUCTION_ID,
        "release_id": RELEASE_ID,
        "release_sha256": release["artifact_sha256"]["quran-urdu.json"],
        "pilot_approval": pilot_approval,
        "model_id": MODEL_ID,
        "voice_id": VOICE_ID,
        "direction": DIRECTION,
        "generation_config": _generation_config(),
        "pronunciation_overrides": PRONUNCIATION_OVERRIDES,
        "units": units,
    }
    return text_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _estimated_cost(units: list[dict[str, Any]]) -> dict[str, Any]:
    calibration = _read_object(PILOT_ROOT / "RECALIBRATED_FULL_PRODUCTION_ESTIMATE.json")
    full = calibration["full_book"]
    estimate = float(full["estimated_batch_cost_usd"])
    return {
        "version": "quran-urdu-charon-production-cost-guard-v1",
        "pricing_checked_on": "2026-08-19",
        "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
        "model_id": MODEL_ID,
        "unit_count": len(units),
        "estimated_duration_hours": full["estimated_duration_hours"],
        "estimated_batch_cost_usd": estimate,
        "planning_range_batch_usd": full["planning_range_batch_usd"],
        "hard_estimated_batch_cost_ceiling_usd": HARD_ESTIMATED_BATCH_COST_USD,
    }


def _pilot_seed(unit: dict[str, Any], root: Path) -> dict[str, Any]:
    passages = _read_json(PILOT_ROOT / "PASSAGES.json")
    fatihah = next(item for item in passages if item["passage_id"] == "fatihah")
    if fatihah["text_sha256"] != unit["text_sha256"]:
        raise UrduAudioProductionError("Approved Fatihah pilot does not match production")
    pilot_state = _read_object(PILOT_ROOT / "RUN.json")
    pilot_job = next(job for job in pilot_state["jobs"] if job["job_id"] == "fatihah")
    source = Path(pilot_job["raw_path"])
    normalized = Path(pilot_job["normalized_path"])
    raw_path = root / "raw" / f"{unit['unit_id']}.wav"
    mp3_path = root / "clips" / f"{unit['unit_id']}.mp3"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        shutil.copy2(source, raw_path)
    if not mp3_path.exists():
        shutil.copy2(normalized, mp3_path)
    return {
        "status": "complete",
        "source": "approved_pilot_reuse",
        "raw_path": str(raw_path),
        "normalized_path": str(mp3_path),
        "raw_sha256": file_sha256(raw_path),
        "normalized_sha256": file_sha256(mp3_path),
        "probe": pilot_job["probe"],
        "provider_usage": pilot_job["provider_usage"],
        "actual_batch_cost_usd": 0.0,
        "historical_standard_cost_usd": pilot_job["actual_standard_cost_usd"],
    }


def _preserved_canary_seed(unit: dict[str, Any], root: Path) -> dict[str, Any] | None:
    state_path = PRE_BOUNDARY_RUN_ROOT / "RUN.json"
    units_path = PRE_BOUNDARY_RUN_ROOT / "UNITS.json"
    if not state_path.exists() or not units_path.exists():
        return None
    prior_state = _read_object(state_path)
    prior_units = {item["unit_id"]: item for item in _read_json(units_path)}
    prior_jobs = {item["unit_id"]: item for item in prior_state.get("jobs", [])}
    prior_unit = prior_units.get(unit["unit_id"])
    prior_job = prior_jobs.get(unit["unit_id"])
    if not prior_unit or not prior_job or prior_job.get("status") != "complete":
        return None
    if prior_unit.get("text_sha256") != unit["text_sha256"]:
        raise UrduAudioProductionError("Preserved Batch canary text does not match")
    source = PRE_BOUNDARY_RUN_ROOT / "raw" / f"{unit['unit_id']}.wav"
    normalized = PRE_BOUNDARY_RUN_ROOT / "clips" / f"{unit['unit_id']}.mp3"
    if not source.exists() or not normalized.exists():
        raise UrduAudioProductionError("Preserved Batch canary audio is missing")
    raw_path = root / "raw" / f"{unit['unit_id']}.wav"
    mp3_path = root / "clips" / f"{unit['unit_id']}.mp3"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, raw_path)
    shutil.copy2(normalized, mp3_path)
    return {
        **{key: value for key, value in prior_job.items() if key not in {"raw_path", "normalized_path"}},
        "status": "complete",
        "source": "preserved_batch_canary",
        "raw_path": str(raw_path),
        "normalized_path": str(mp3_path),
        "raw_sha256": file_sha256(raw_path),
        "normalized_sha256": file_sha256(mp3_path),
    }


def prepare(root: Path = PRODUCTION_ROOT) -> dict[str, Any]:
    approval = PILOT_ROOT / "PILOT_APPROVED.json"
    if not approval.exists():
        raise UrduAudioProductionError("Approved pilot marker is missing")
    units = _build_units()
    estimate = _estimated_cost(units)
    if estimate["estimated_batch_cost_usd"] > HARD_ESTIMATED_BATCH_COST_USD:
        raise UrduAudioProductionError("Estimated Batch spend exceeds the hard ceiling")
    max_unit = max(units, key=lambda item: int(item["speech_characters"]))
    estimated_max_seconds = (
        int(max_unit["speech_characters"])
        * float(estimate["estimated_duration_hours"])
        * 3600
        / sum(int(unit["speech_characters"]) for unit in units)
    )
    if estimated_max_seconds * AUDIO_TOKENS_PER_SECOND >= 15_000:
        raise UrduAudioProductionError("Longest unit is too close to the output limit")
    fingerprint = _fingerprint(units)
    state_path = root / "RUN.json"
    if state_path.exists():
        state = _read_object(state_path)
        if state.get("input_fingerprint") != fingerprint:
            raise UrduAudioProductionError("Production inputs changed; refusing a mixed resume")
        return state

    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "UNITS.json", units)
    atomic_json(root / "COST_GUARD.json", estimate)
    preserved_canary = _preserved_canary_seed(units[1], root)
    seeded_units = 2 if preserved_canary else 1
    shards = _make_shards(units, seeded_units=seeded_units)
    batches: list[dict[str, Any]] = []
    for index, shard in enumerate(shards, start=1):
        text = "\n".join(_request_line(unit) for unit in shard) + "\n"
        path = root / "jobs" / f"shard-{index:03d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(path, text)
        batches.append(
            {
                "shard": index,
                "canary": index == 1 and preserved_canary is None,
                "status": "prepared",
                "input_path": str(path),
                "input_sha256": text_sha256(text),
                "unit_ids": [unit["unit_id"] for unit in shard],
                "characters": sum(int(unit["characters"]) for unit in shard),
                "uploaded_file_name": None,
                "batch_id": None,
                "last_error": None,
            }
        )
    jobs = []
    for unit in units:
        job = {
            "unit_id": unit["unit_id"],
            "status": "pending",
            "raw_path": str(root / "raw" / f"{unit['unit_id']}.wav"),
            "normalized_path": str(root / "clips" / f"{unit['unit_id']}.mp3"),
            "last_error": None,
        }
        if unit["unit_index"] == 1:
            job.update(_pilot_seed(unit, root))
        elif unit["unit_index"] == 2 and preserved_canary:
            job.update(preserved_canary)
        jobs.append(job)
    state = {
        "version": "quran-urdu-charon-production-run-v1",
        "production_id": PRODUCTION_ID,
        "release_id": RELEASE_ID,
        "input_fingerprint": fingerprint,
        "model_id": MODEL_ID,
        "voice_id": VOICE_ID,
        "direction": DIRECTION,
        "hard_estimated_batch_cost_ceiling_usd": HARD_ESTIMATED_BATCH_COST_USD,
        "status": "prepared",
        "prepared_at": utc_now(),
        "updated_at": utc_now(),
        "canary_status": "passed_imported" if preserved_canary else "pending",
        "jobs": jobs,
        "batches": batches,
    }
    atomic_json(state_path, state)
    atomic_json(
        root / "MANIFEST.json",
        {
            "version": "quran-urdu-charon-production-manifest-v2",
            "production_id": PRODUCTION_ID,
            "input_fingerprint": fingerprint,
            "units": len(units),
            "ayahs": 6_236,
            "shards": len(shards),
            "seeded_units": seeded_units,
            "pronunciation_overrides": {
                ref: {"source_prefix": source, "spoken_prefix": spoken}
                for ref, (source, spoken) in PRONUNCIATION_OVERRIDES.items()
            },
            "source_sha256": file_sha256(RELEASE_ROOT / "quran-urdu.json"),
            "units_sha256": file_sha256(root / "UNITS.json"),
        },
    )
    return state


def _state_name(job: Any) -> str:
    state = getattr(job, "state", None)
    return str(getattr(state, "name", state) or "JOB_STATE_UNSPECIFIED")


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json(root / "RUN.json", state)


def _submit_batch(client: Any, root: Path, state: dict[str, Any], batch: dict[str, Any]) -> None:
    if batch["status"] != "prepared":
        raise UrduAudioProductionError(f"Shard {batch['shard']} is not prepared")
    batch["status"] = "uploading"
    batch["submission_intent_at"] = utc_now()
    _save_state(root, state)
    try:
        uploaded = client.files.upload(
            file=Path(batch["input_path"]),
            config=types.UploadFileConfig(
                display_name=f"{PRODUCTION_ID}-shard-{batch['shard']:03d}",
                mime_type="application/jsonl",
            ),
        )
        batch["uploaded_file_name"] = uploaded.name
        batch["status"] = "uploaded"
        _save_state(root, state)
        batch["status"] = "creating_batch"
        batch["batch_create_intent_at"] = utc_now()
        _save_state(root, state)
        provider_job = client.batches.create(
            model=MODEL_ID,
            src=uploaded.name,
            config={"display_name": f"{PRODUCTION_ID}-shard-{batch['shard']:03d}"},
        )
        batch["batch_id"] = provider_job.name
        batch["status"] = _state_name(provider_job)
        batch["submitted_at"] = utc_now()
        batch["provider_job"] = provider_job.model_dump(
            by_alias=True, exclude_none=True, mode="json"
        )
        _save_state(root, state)
    except Exception as exc:
        batch["last_error"] = f"{type(exc).__name__}: {exc}"
        # An upload failure is retryable only by explicit adjudication. A create
        # failure may be ambiguous, so neither path is retried automatically.
        batch["status"] = "submission_blocked"
        state["status"] = "blocked"
        _save_state(root, state)
        raise UrduAudioProductionError(
            f"Shard {batch['shard']} submission blocked: {batch['last_error']}"
        ) from exc


def _poll_batch(client: Any, root: Path, state: dict[str, Any], batch: dict[str, Any]) -> str:
    if not batch.get("batch_id"):
        raise UrduAudioProductionError(f"Shard {batch['shard']} has no provider batch id")
    provider_job = client.batches.get(name=batch["batch_id"])
    name = _state_name(provider_job)
    batch["status"] = name
    batch["last_polled_at"] = utc_now()
    batch["provider_job"] = provider_job.model_dump(
        by_alias=True, exclude_none=True, mode="json"
    )
    _save_state(root, state)
    return name


def _write_wav(path: Path, audio: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with wave.open(str(temp), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(audio)
    os.replace(temp, path)


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


def _probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=codec_name,sample_rate,channels",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    format_data = payload.get("format") or {}
    return {
        "duration_seconds": round(float(format_data.get("duration") or 0), 3),
        "bytes": int(format_data.get("size") or path.stat().st_size),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def _response_audio(response: dict[str, Any]) -> bytes:
    try:
        part = response["candidates"][0]["content"]["parts"][0]
        inline = part.get("inlineData") or part.get("inline_data")
        data = inline["data"]
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise UrduAudioProductionError("Batch response did not contain audio") from exc
    return base64.b64decode(data)


def _usage(response: dict[str, Any], duration: float) -> dict[str, int]:
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    prompt = int(usage.get("promptTokenCount") or usage.get("prompt_token_count") or 0)
    output = int(
        usage.get("candidatesTokenCount") or usage.get("candidates_token_count") or 0
    )
    if output <= 0:
        output = math.ceil(duration * AUDIO_TOKENS_PER_SECOND)
    return {"prompt_tokens": prompt, "output_tokens": output}


def _ffconcat_line(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _surah_names() -> dict[int, str]:
    listening = OUTPUT_DIR / "release" / "quran-translation-v2.4.1" / "quran-listening-edition.json"
    payload = _read_object(listening)
    rows = payload.get("ayahs")
    if not isinstance(rows, list) or len(rows) != 6_236:
        raise UrduAudioProductionError("English listening edition is missing Surah names")
    names: dict[int, str] = {}
    for row in rows:
        surah = int(row["surah"])
        names.setdefault(surah, str(row["surah_name_en"]))
    if len(names) != 114:
        raise UrduAudioProductionError("Expected names for 114 Surahs")
    return names


def _concat_mp3(inputs: list[Path], output: Path, *, title: str, track: int | None = None) -> None:
    if not inputs:
        raise UrduAudioProductionError(f"No source clips for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_path = output.with_name(f".{output.name}.concat.txt")
    temp = output.with_name(f".{output.name}.tmp.mp3")
    atomic_text(concat_path, "\n".join(_ffconcat_line(path) for path in inputs) + "\n")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
        "-safe", "0", "-i", str(concat_path), "-map_metadata", "-1", "-c", "copy",
        "-id3v2_version", "3", "-metadata", f"title={title}",
        "-metadata", "album=The Quran - Urdu Translation",
        "-metadata", "artist=Narrated with Gemini Charon",
    ]
    if track is not None:
        command.extend(["-metadata", f"track={track}"])
    command.append(str(temp))
    try:
        subprocess.run(command, check=True, capture_output=True)
        os.replace(temp, output)
    finally:
        concat_path.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)


def _concat_wav(inputs: list[Path], output: Path) -> None:
    if not inputs:
        raise UrduAudioProductionError(f"No source fragments for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_path = output.with_name(f".{output.name}.concat.txt")
    temp = output.with_name(f".{output.name}.tmp.wav")
    atomic_text(concat_path, "\n".join(_ffconcat_line(path) for path in inputs) + "\n")
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_path),
                "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(temp),
            ],
            check=True,
            capture_output=True,
        )
        os.replace(temp, output)
    finally:
        concat_path.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)


def _decode_check(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )


def _assembled_record(
    *,
    output_type: str,
    output_id: str,
    label: str,
    path: Path,
    jobs: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    probe = _probe(path)
    expected = sum(float(job["probe"]["duration_seconds"]) for job in jobs)
    tolerance = max(3.0, expected * 0.005)
    if abs(probe["duration_seconds"] - expected) > tolerance:
        raise UrduAudioProductionError(
            f"Assembled duration mismatch for {path}: {probe['duration_seconds']} vs {expected}"
        )
    if probe["codec"] != "mp3" or probe["sample_rate"] != 44_100 or probe["channels"] != 1:
        raise UrduAudioProductionError(f"Assembled audio contract failed for {path}: {probe}")
    first = units[jobs[0]["unit_id"]]["refs"][0]
    last = units[jobs[-1]["unit_id"]]["refs"][-1]
    return {
        "type": output_type,
        "id": output_id,
        "label": label,
        "range": f"{first}-{last}",
        "path": str(path),
        "sha256": file_sha256(path),
        **probe,
        "source_units": [job["unit_id"] for job in jobs],
    }


def assemble(
    root: Path = PRODUCTION_ROOT,
    release_root: Path = RELEASE_AUDIO_ROOT,
    *,
    decode_check: bool = True,
) -> dict[str, Any]:
    state = _read_object(root / "RUN.json")
    if any(job.get("status") != "complete" for job in state.get("jobs", [])):
        raise UrduAudioProductionError("All synthesis units must complete before assembly")
    units_list = _read_json(root / "UNITS.json")
    units = {unit["unit_id"]: unit for unit in units_list}
    jobs = state["jobs"]
    for job in jobs:
        path = Path(job["normalized_path"])
        if not path.exists() or file_sha256(path) != job.get("normalized_sha256"):
            raise UrduAudioProductionError(f"Master clip integrity failure: {path}")

    assembly_path = root / "ASSEMBLY_STATE.json"
    if assembly_path.exists():
        assembly_state = _read_object(assembly_path)
        if assembly_state.get("input_fingerprint") != state["input_fingerprint"]:
            raise UrduAudioProductionError("Assembly input fingerprint changed")
    else:
        assembly_state = {
            "version": "quran-urdu-charon-assembly-state-v1",
            "input_fingerprint": state["input_fingerprint"],
            "status": "assembling",
            "outputs": [],
            "started_at": utc_now(),
        }
        atomic_json(assembly_path, assembly_state)
    completed = {record["path"]: record for record in assembly_state["outputs"]}

    by_surah: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_juz: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        unit = units[job["unit_id"]]
        by_surah[int(unit["surah"])].append(job)
        by_juz[int(unit["juz"])].append(job)
    if len(by_surah) != 114 or len(by_juz) != 30:
        raise UrduAudioProductionError("Assembly groups must contain 114 Surahs and 30 Paras")
    names = _surah_names()

    specifications: list[tuple[str, str, str, Path, list[dict[str, Any]], int | None]] = []
    for surah in range(1, 115):
        name = names[surah]
        specifications.append(
            (
                "surah", f"{surah:03d}", name,
                release_root / "by-surah" / f"{surah:03d}-{_slug(name)}.mp3",
                by_surah[surah], surah,
            )
        )
    for juz in range(1, 31):
        specifications.append(
            (
                "para", f"{juz:02d}", f"Para {juz} of 30",
                release_root / "by-para" / f"para-{juz:02d}-of-30.mp3",
                by_juz[juz], juz,
            )
        )
    specifications.append(
        (
            "full_book", "full", "The Quran - Urdu Translation",
            release_root / "full-book" / "quran-urdu-complete.mp3", jobs, None,
        )
    )

    for output_type, output_id, label, path, group, track in specifications:
        prior = completed.get(str(path))
        if prior:
            if not path.exists() or file_sha256(path) != prior["sha256"]:
                raise UrduAudioProductionError(f"Completed assembly output changed: {path}")
            continue
        _concat_mp3(
            [Path(job["normalized_path"]) for job in group],
            path,
            title=label,
            track=track,
        )
        record = _assembled_record(
            output_type=output_type,
            output_id=output_id,
            label=label,
            path=path,
            jobs=group,
            units=units,
        )
        assembly_state["outputs"].append(record)
        assembly_state["updated_at"] = utc_now()
        atomic_json(assembly_path, assembly_state)

    outputs = assembly_state["outputs"]
    if len(outputs) != 145:
        raise UrduAudioProductionError(f"Expected 145 release MP3s, found {len(outputs)}")
    if decode_check:
        for record in outputs:
            _decode_check(Path(record["path"]))

    manifest = {
        "version": "quran-urdu-charon-audio-release-v1",
        "production_id": PRODUCTION_ID,
        "release_id": RELEASE_ID,
        "created_at": utc_now(),
        "input_fingerprint": state["input_fingerprint"],
        "voice": {"provider": "google", "model_id": MODEL_ID, "voice_id": VOICE_ID},
        "counts": {"master_units": len(jobs), "surahs": 114, "paras": 30, "full_book": 1},
        "totals": {
            "master_duration_seconds": round(
                sum(float(job["probe"]["duration_seconds"]) for job in jobs), 3
            ),
            "release_bytes": sum(int(record["bytes"]) for record in outputs),
            "recorded_batch_spend_usd": round(
                sum(float(job.get("actual_batch_cost_usd", 0)) for job in jobs), 6
            ),
        },
        "qa": {
            "all_master_hashes_verified": True,
            "all_output_durations_verified": True,
            "audio_contract": "MP3, 44.1 kHz, mono, approximately 192 kbps",
            "full_decode_check": decode_check,
        },
        "outputs": outputs,
    }
    release_root.mkdir(parents=True, exist_ok=True)
    atomic_json(release_root / "RELEASE_MANIFEST.json", manifest)
    qa = {
        "version": "quran-urdu-charon-audio-qa-v1",
        "passed": True,
        "counts": manifest["counts"],
        "totals": manifest["totals"],
        "qa": manifest["qa"],
    }
    atomic_json(release_root / "QA_REPORT.json", qa)
    checksum_paths = [Path(record["path"]) for record in outputs] + [
        release_root / "RELEASE_MANIFEST.json",
        release_root / "QA_REPORT.json",
    ]
    atomic_text(
        release_root / "SHA256SUMS.txt",
        "\n".join(
            f"{file_sha256(path)}  {path.relative_to(release_root)}" for path in checksum_paths
        )
        + "\n",
    )
    assembly_state["status"] = "complete"
    assembly_state["completed_at"] = utc_now()
    atomic_json(assembly_path, assembly_state)
    state["status"] = "complete"
    state["release_root"] = str(release_root)
    state["completed_at"] = utc_now()
    _save_state(root, state)
    atomic_json(root / "PRODUCTION_COMPLETE.json", {"qa": qa, "release_root": str(release_root)})
    return manifest


def _collect_batch(client: Any, root: Path, state: dict[str, Any], batch: dict[str, Any]) -> None:
    if batch["status"] not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
        raise UrduAudioProductionError(f"Shard {batch['shard']} is not collectable")
    provider_job = client.batches.get(name=batch["batch_id"])
    destination = getattr(provider_job, "dest", None)
    file_name = getattr(destination, "file_name", None)
    if not file_name:
        raise UrduAudioProductionError(f"Shard {batch['shard']} has no result file")
    result_path = root / "results" / f"shard-{batch['shard']:03d}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not result_path.exists():
        payload = bytes(client.files.download(file=file_name))
        temp = result_path.with_name(f".{result_path.name}.tmp")
        temp.write_bytes(payload)
        os.replace(temp, result_path)
    rows = {}
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row.get("key"))] = row
    units = {unit["unit_id"]: unit for unit in _read_json(root / "UNITS.json")}
    jobs = {job["unit_id"]: job for job in state["jobs"]}
    failures = []
    for unit_id in batch["unit_ids"]:
        job = jobs[unit_id]
        if job["status"] == "complete":
            continue
        row = rows.get(unit_id)
        if row is None or row.get("error"):
            error = row.get("error") if row else "missing result row"
            job.update({"status": "failed", "last_error": str(error)})
            failures.append(unit_id)
            continue
        try:
            response = row.get("response") or {}
            raw = Path(job["raw_path"])
            normalized = Path(job["normalized_path"])
            _write_wav(raw, _response_audio(response))
            _normalize(raw, normalized)
            probe = _probe(normalized)
            ratio = probe["duration_seconds"] / max(1, int(units[unit_id]["characters"]))
            if probe["codec"] != "mp3" or probe["sample_rate"] != 44_100 or probe["channels"] != 1:
                raise UrduAudioProductionError(f"Audio contract failed: {probe}")
            if not 0.04 <= ratio <= 0.22:
                raise UrduAudioProductionError(
                    f"Implausible duration {probe['duration_seconds']}s for {unit_id}"
                )
            usage = _usage(response, probe["duration_seconds"])
            cost = (
                usage["prompt_tokens"] / 1_000_000 * BATCH_INPUT_USD_PER_MILLION
                + usage["output_tokens"] / 1_000_000 * BATCH_AUDIO_USD_PER_MILLION
            )
            job.update(
                {
                    "status": "complete",
                    "provider_batch_id": batch["batch_id"],
                    "provider_result_file": file_name,
                    "provider_usage": usage,
                    "actual_batch_cost_usd": round(cost, 6),
                    "raw_sha256": file_sha256(raw),
                    "normalized_sha256": file_sha256(normalized),
                    "probe": probe,
                    "last_error": None,
                }
            )
        except Exception as exc:
            job.update({"status": "failed", "last_error": f"{type(exc).__name__}: {exc}"})
            failures.append(unit_id)
        finally:
            _save_state(root, state)
    batch["result_file_name"] = file_name
    batch["result_path"] = str(result_path)
    batch["result_sha256"] = file_sha256(result_path)
    batch["status"] = "collected" if not failures else "collected_with_failures"
    batch["collected_at"] = utc_now()
    if failures:
        batch["last_error"] = f"Failed units: {', '.join(failures)}"
        state["status"] = "failed_unit_recovery_needed"
    _save_state(root, state)


def _fragment_unit(unit: dict[str, Any]) -> list[dict[str, Any]]:
    refs = list(unit["refs"])
    speech_lines = str(unit["speech_text"]).splitlines()
    text_lines = str(unit["text"]).splitlines()
    if not refs or len(refs) != len(speech_lines) or len(refs) != len(text_lines):
        raise UrduAudioProductionError(
            f"Ayah-line alignment failed for fragment recovery: {unit['unit_id']}"
        )
    chunks: list[list[int]] = []
    current: list[int] = []
    current_chars = 0
    for index, line in enumerate(speech_lines):
        line_chars = len(line) + (1 if current else 0)
        if current and (
            len(current) >= FRAGMENT_MAX_AYAHS
            or current_chars + line_chars > FRAGMENT_TARGET_CHARACTERS
        ):
            chunks.append(current)
            current = []
            current_chars = 0
            line_chars = len(line)
        current.append(index)
        current_chars += line_chars
    if current:
        chunks.append(current)
    if len(chunks) == 1 and len(refs) > 1:
        midpoint = math.ceil(len(refs) / 2)
        chunks = [list(range(0, midpoint)), list(range(midpoint, len(refs)))]

    fragments = []
    for fragment_index, indexes in enumerate(chunks, start=1):
        speech_text = "\n".join(speech_lines[index] for index in indexes)
        text = "\n".join(text_lines[index] for index in indexes)
        fragment_id = f"frag-{unit['unit_id']}-{fragment_index:02d}"
        fragments.append(
            {
                "fragment_id": fragment_id,
                "fragment_index": fragment_index,
                "original_unit_id": unit["unit_id"],
                "refs": [refs[index] for index in indexes],
                "text": text,
                "speech_text": speech_text,
                "characters": len(text),
                "speech_characters": len(speech_text),
                "text_sha256": text_sha256(text),
                "speech_text_sha256": text_sha256(speech_text),
                "status": "pending",
                "last_error": None,
            }
        )
    if [ref for fragment in fragments for ref in fragment["refs"]] != refs:
        raise UrduAudioProductionError(f"Fragment coverage changed for {unit['unit_id']}")
    return fragments


def _make_fragment_shards(
    fragments: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    shards: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for fragment in fragments:
        characters = int(fragment["speech_characters"])
        if current and current_chars + characters > SHARD_TARGET_CHARACTERS:
            shards.append(current)
            current = []
            current_chars = 0
        current.append(fragment)
        current_chars += characters
    if current:
        shards.append(current)
    return shards


def prepare_fragment_recovery(root: Path = PRODUCTION_ROOT) -> dict[str, Any]:
    """Freeze ayah-boundary fragments for units exhausted by full-unit retries."""
    state = _read_object(root / "RUN.json")
    recovery_path = root / "FRAGMENT_RECOVERY.json"
    existing_batches = [
        batch for batch in state["batches"] if batch.get("fragment_recovery")
    ]
    if recovery_path.exists() or existing_batches:
        if not recovery_path.exists() or not existing_batches:
            raise UrduAudioProductionError("Fragment recovery artifacts are incomplete")
        return status(root)

    failed_jobs = [job for job in state["jobs"] if job["status"] == "failed"]
    if not failed_jobs:
        raise UrduAudioProductionError("No failed synthesis units need fragment recovery")
    units = {unit["unit_id"]: unit for unit in _read_json(root / "UNITS.json")}
    fragments = [
        fragment
        for job in failed_jobs
        for fragment in _fragment_unit(units[job["unit_id"]])
    ]
    if any(
        len(fragment["refs"]) >= len(units[fragment["original_unit_id"]]["refs"])
        for fragment in fragments
    ):
        raise UrduAudioProductionError("Fragment recovery did not reduce every failed unit")

    fragment_root = root / "fragment-recovery"
    for fragment in fragments:
        fragment["raw_path"] = str(fragment_root / "raw" / f"{fragment['fragment_id']}.wav")
        fragment["normalized_path"] = str(
            fragment_root / "clips" / f"{fragment['fragment_id']}.mp3"
        )
    recovery = {
        "version": "quran-urdu-charon-fragment-recovery-v1",
        "created_at": utc_now(),
        "input_fingerprint": state["input_fingerprint"],
        "original_unit_ids": [job["unit_id"] for job in failed_jobs],
        "fragment_target_characters": FRAGMENT_TARGET_CHARACTERS,
        "fragment_max_ayahs": FRAGMENT_MAX_AYAHS,
        "fragments": fragments,
    }
    atomic_json(recovery_path, recovery)

    next_shard = max(int(batch["shard"]) for batch in state["batches"]) + 1
    batches = []
    for offset, shard in enumerate(_make_fragment_shards(fragments)):
        shard_number = next_shard + offset
        input_path = root / "jobs" / f"fragment-recovery-{shard_number:03d}.jsonl"
        text = "\n".join(
            _request_line(
                {"unit_id": item["fragment_id"], "speech_text": item["speech_text"]}
            )
            for item in shard
        ) + "\n"
        atomic_text(input_path, text)
        batches.append(
            {
                "shard": shard_number,
                "canary": False,
                "status": "prepared",
                "input_path": str(input_path),
                "input_sha256": text_sha256(text),
                "unit_ids": [item["fragment_id"] for item in shard],
                "original_unit_ids": list(
                    dict.fromkeys(item["original_unit_id"] for item in shard)
                ),
                "characters": sum(int(item["speech_characters"]) for item in shard),
                "uploaded_file_name": None,
                "batch_id": None,
                "last_error": None,
                "fragment_recovery": True,
            }
        )

    insertion = next(
        (
            index
            for index, batch in enumerate(state["batches"])
            if batch["status"] in {"submission_blocked", "prepared"}
        ),
        len(state["batches"]),
    )
    state["batches"][insertion:insertion] = batches
    for job in failed_jobs:
        job.setdefault("failure_history", []).append(
            {
                "status": "failed",
                "error": job.get("last_error"),
                "preserved_at": utc_now(),
                "recovery": "ayah_fragment_recovery",
            }
        )
        job["status"] = "pending"
        job["last_error"] = None
        job["fragment_recovery"] = True
    state["status"] = "fragment_recovery_prepared"
    _save_state(root, state)
    atomic_json(
        root / "FRAGMENT_RECOVERY_TARGETS.json",
        {
            "original_unit_ids": recovery["original_unit_ids"],
            "fragment_ids": [item["fragment_id"] for item in fragments],
            "batch_shards": [batch["shard"] for batch in batches],
        },
    )
    return status(root)


def _stitch_fragment_unit(
    root: Path,
    state: dict[str, Any],
    recovery: dict[str, Any],
    unit_id: str,
) -> None:
    fragments = sorted(
        [
            fragment
            for fragment in recovery["fragments"]
            if fragment["original_unit_id"] == unit_id
        ],
        key=lambda item: int(item["fragment_index"]),
    )
    job = next(job for job in state["jobs"] if job["unit_id"] == unit_id)
    failed = [item for item in fragments if item["status"] == "failed"]
    if failed:
        job["status"] = "failed"
        job["last_error"] = "Failed fragments: " + ", ".join(
            item["fragment_id"] for item in failed
        )
        return
    if not fragments or any(item["status"] != "complete" for item in fragments):
        job["status"] = "pending"
        return

    raw = Path(job["raw_path"])
    normalized = Path(job["normalized_path"])
    _concat_wav([Path(item["raw_path"]) for item in fragments], raw)
    _concat_mp3(
        [Path(item["normalized_path"]) for item in fragments],
        normalized,
        title=unit_id,
    )
    _decode_check(normalized)
    probe = _probe(normalized)
    units = {unit["unit_id"]: unit for unit in _read_json(root / "UNITS.json")}
    ratio = probe["duration_seconds"] / max(1, int(units[unit_id]["characters"]))
    if probe["codec"] != "mp3" or probe["sample_rate"] != 44_100 or probe["channels"] != 1:
        raise UrduAudioProductionError(f"Stitched audio contract failed: {probe}")
    if not 0.04 <= ratio <= 0.22:
        raise UrduAudioProductionError(
            f"Implausible stitched duration {probe['duration_seconds']}s for {unit_id}"
        )
    usage = {
        "prompt_tokens": sum(
            int(item["provider_usage"]["prompt_tokens"]) for item in fragments
        ),
        "output_tokens": sum(
            int(item["provider_usage"]["output_tokens"]) for item in fragments
        ),
    }
    job.update(
        {
            "status": "complete",
            "source": "ayah_fragment_recovery",
            "fragment_ids": [item["fragment_id"] for item in fragments],
            "provider_batch_ids": list(
                dict.fromkeys(str(item["provider_batch_id"]) for item in fragments)
            ),
            "provider_usage": usage,
            "actual_batch_cost_usd": round(
                sum(float(item["actual_batch_cost_usd"]) for item in fragments), 6
            ),
            "raw_sha256": file_sha256(raw),
            "normalized_sha256": file_sha256(normalized),
            "probe": probe,
            "last_error": None,
        }
    )


def _collect_fragment_batch(
    client: Any,
    root: Path,
    state: dict[str, Any],
    batch: dict[str, Any],
) -> None:
    if batch["status"] not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
        raise UrduAudioProductionError(f"Shard {batch['shard']} is not collectable")
    provider_job = client.batches.get(name=batch["batch_id"])
    destination = getattr(provider_job, "dest", None)
    file_name = getattr(destination, "file_name", None)
    if not file_name:
        raise UrduAudioProductionError(f"Shard {batch['shard']} has no result file")
    result_path = root / "results" / f"fragment-recovery-{batch['shard']:03d}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not result_path.exists():
        payload = bytes(client.files.download(file=file_name))
        temp = result_path.with_name(f".{result_path.name}.tmp")
        temp.write_bytes(payload)
        os.replace(temp, result_path)
    rows = {}
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row.get("key"))] = row
    recovery_path = root / "FRAGMENT_RECOVERY.json"
    recovery = _read_object(recovery_path)
    fragments = {item["fragment_id"]: item for item in recovery["fragments"]}
    failures = []
    for fragment_id in batch["unit_ids"]:
        fragment = fragments[fragment_id]
        if fragment["status"] == "complete":
            continue
        row = rows.get(fragment_id)
        if row is None or row.get("error"):
            error = row.get("error") if row else "missing result row"
            fragment.update({"status": "failed", "last_error": str(error)})
            failures.append(fragment_id)
            continue
        try:
            response = row.get("response") or {}
            raw = Path(fragment["raw_path"])
            normalized = Path(fragment["normalized_path"])
            _write_wav(raw, _response_audio(response))
            _normalize(raw, normalized)
            probe = _probe(normalized)
            ratio = probe["duration_seconds"] / max(1, int(fragment["characters"]))
            if (
                probe["codec"] != "mp3"
                or probe["sample_rate"] != 44_100
                or probe["channels"] != 1
            ):
                raise UrduAudioProductionError(f"Audio contract failed: {probe}")
            if not 0.04 <= ratio <= 0.22:
                raise UrduAudioProductionError(
                    f"Implausible duration {probe['duration_seconds']}s for {fragment_id}"
                )
            usage = _usage(response, probe["duration_seconds"])
            cost = (
                usage["prompt_tokens"] / 1_000_000 * BATCH_INPUT_USD_PER_MILLION
                + usage["output_tokens"] / 1_000_000 * BATCH_AUDIO_USD_PER_MILLION
            )
            fragment.update(
                {
                    "status": "complete",
                    "provider_batch_id": batch["batch_id"],
                    "provider_result_file": file_name,
                    "provider_usage": usage,
                    "actual_batch_cost_usd": round(cost, 6),
                    "raw_sha256": file_sha256(raw),
                    "normalized_sha256": file_sha256(normalized),
                    "probe": probe,
                    "last_error": None,
                }
            )
        except Exception as exc:
            fragment.update(
                {"status": "failed", "last_error": f"{type(exc).__name__}: {exc}"}
            )
            failures.append(fragment_id)
        finally:
            atomic_json(recovery_path, recovery)
    for unit_id in batch["original_unit_ids"]:
        try:
            _stitch_fragment_unit(root, state, recovery, unit_id)
        except Exception as exc:
            job = next(job for job in state["jobs"] if job["unit_id"] == unit_id)
            job.update(
                {"status": "failed", "last_error": f"{type(exc).__name__}: {exc}"}
            )
            failures.append(unit_id)
    batch["result_file_name"] = file_name
    batch["result_path"] = str(result_path)
    batch["result_sha256"] = file_sha256(result_path)
    batch["status"] = "collected" if not failures else "collected_with_failures"
    batch["collected_at"] = utc_now()
    batch["last_error"] = (
        None if not failures else "Failed items: " + ", ".join(failures)
    )
    if failures:
        state["status"] = "fragment_recovery_blocked"
    atomic_json(recovery_path, recovery)
    _save_state(root, state)


def _collect_ready_batch(
    client: Any,
    root: Path,
    state: dict[str, Any],
    batch: dict[str, Any],
) -> None:
    if batch.get("fragment_recovery"):
        _collect_fragment_batch(client, root, state, batch)
    else:
        _collect_batch(client, root, state, batch)


COLLECTED_BATCH_STATES = {"collected", "collected_with_failures"}


def prepare_failed_unit_recovery(
    root: Path = PRODUCTION_ROOT,
) -> dict[str, Any]:
    """Prepare one idempotent Batch retry containing only failed provider items."""
    state = _read_object(root / "RUN.json")
    existing = [batch for batch in state["batches"] if batch.get("failed_unit_recovery")]
    prepared_recoveries = [
        batch
        for batch in existing
        if batch["status"] == "prepared" and not batch.get("batch_id")
    ]
    if len(prepared_recoveries) > 1:
        raise UrduAudioProductionError("Multiple prepared failed-unit recoveries exist")
    recovery = prepared_recoveries[0] if prepared_recoveries else None

    jobs = {job["unit_id"]: job for job in state["jobs"]}
    failed_ids = [job["unit_id"] for job in state["jobs"] if job["status"] == "failed"]
    target_ids = list(dict.fromkeys((recovery or {}).get("unit_ids", []) + failed_ids))
    if not target_ids:
        raise UrduAudioProductionError("No failed synthesis units need recovery")
    exhausted = []
    for unit_id in failed_ids:
        job = jobs[unit_id]
        attempts = len(job.get("failure_history", [])) + 1
        if attempts >= 3:
            exhausted.append(unit_id)
    if exhausted:
        raise UrduAudioProductionError(
            "Failed-unit attempt ceiling reached: " + ", ".join(exhausted)
        )

    source_batches = []
    for batch in state["batches"]:
        if batch.get("failed_unit_recovery"):
            continue
        affected = [unit_id for unit_id in batch["unit_ids"] if unit_id in target_ids]
        if not affected:
            continue
        if batch["status"] == "collection_blocked":
            batch["status"] = "collected_with_failures"
        if batch["status"] != "collected_with_failures":
            raise UrduAudioProductionError(
                f"Failed units belong to uncollected shard {batch['shard']}"
            )
        source_batches.append(batch["shard"])

    units = {unit["unit_id"]: unit for unit in _read_json(root / "UNITS.json")}
    missing = [unit_id for unit_id in target_ids if unit_id not in units]
    if missing:
        raise UrduAudioProductionError(f"Recovery units missing from manifest: {missing}")

    attempt = int((recovery or {}).get("recovery_attempt") or 0)
    if not recovery:
        attempt = max(
            [int(batch.get("recovery_attempt") or 0) for batch in existing] or [0]
        ) + 1
    input_path = (
        Path(recovery["input_path"])
        if recovery
        else root / "jobs" / f"failed-unit-recovery-{attempt:03d}.jsonl"
    )
    text = "\n".join(_request_line(units[unit_id]) for unit_id in target_ids) + "\n"
    atomic_text(input_path, text)
    if recovery:
        recovery.update(
            {
                "input_sha256": text_sha256(text),
                "unit_ids": target_ids,
                "characters": sum(int(units[unit_id]["characters"]) for unit_id in target_ids),
                "source_shards": source_batches,
            }
        )
    else:
        shard = max(int(batch["shard"]) for batch in state["batches"]) + 1
        recovery = {
            "shard": shard,
            "canary": False,
            "status": "prepared",
            "input_path": str(input_path),
            "input_sha256": text_sha256(text),
            "unit_ids": target_ids,
            "characters": sum(int(units[unit_id]["characters"]) for unit_id in target_ids),
            "uploaded_file_name": None,
            "batch_id": None,
            "last_error": None,
            "failed_unit_recovery": True,
            "recovery_attempt": attempt,
            "source_shards": source_batches,
        }

    for unit_id in target_ids:
        job = jobs[unit_id]
        if job["status"] == "failed":
            job.setdefault("failure_history", []).append(
                {
                    "status": "failed",
                    "error": job.get("last_error"),
                    "preserved_at": utc_now(),
                    "source_shards": source_batches,
                }
            )
        job["status"] = "pending"
        job["last_error"] = None

    if recovery not in state["batches"]:
        insertion = next(
            (
                index
                for index, batch in enumerate(state["batches"])
                if batch["status"] in {"submission_blocked", "prepared"}
            ),
            len(state["batches"]),
        )
        state["batches"].insert(insertion, recovery)
    state["failed_unit_recovery_attempts"] = attempt
    state["status"] = "failed_unit_recovery_prepared"
    _save_state(root, state)
    target_record = {
        "attempt": attempt,
        "unit_ids": target_ids,
        "source_shards": source_batches,
        "input_path": str(input_path),
        "input_sha256": text_sha256(text),
    }
    latest_targets = root / "FAILED_UNIT_RECOVERY_TARGETS.json"
    first_targets = root / "FAILED_UNIT_RECOVERY_TARGETS-001.json"
    if attempt > 1 and latest_targets.exists() and not first_targets.exists():
        shutil.copy2(latest_targets, first_targets)
    atomic_json(root / f"FAILED_UNIT_RECOVERY_TARGETS-{attempt:03d}.json", target_record)
    atomic_json(latest_targets, target_record)
    return status(root)


def status(root: Path = PRODUCTION_ROOT) -> dict[str, Any]:
    state = prepare(root)
    counts: dict[str, int] = {}
    for job in state["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    batch_counts: dict[str, int] = {}
    for batch in state["batches"]:
        batch_counts[batch["status"]] = batch_counts.get(batch["status"], 0) + 1
    complete = [job for job in state["jobs"] if job["status"] == "complete"]
    return {
        "production_id": PRODUCTION_ID,
        "status": state["status"],
        "unit_counts": counts,
        "batch_counts": batch_counts,
        "complete_units": len(complete),
        "total_units": len(state["jobs"]),
        "generated_duration_seconds": round(
            sum(float(job.get("probe", {}).get("duration_seconds", 0)) for job in complete), 3
        ),
        "generated_bytes": sum(int(job.get("probe", {}).get("bytes", 0)) for job in complete),
        "recorded_batch_spend_usd": round(
            sum(float(job.get("actual_batch_cost_usd", 0)) for job in complete), 6
        ),
        "updated_at": state.get("updated_at"),
        "batches": [
            {
                "shard": batch["shard"],
                "canary": batch["canary"],
                "status": batch["status"],
                "units": len(batch["unit_ids"]),
                "batch_id": batch.get("batch_id"),
                "last_error": batch.get("last_error"),
            }
            for batch in state["batches"]
        ],
    }


def collect_submitted(
    root: Path = PRODUCTION_ROOT, *, poll_seconds: int = POLL_SECONDS
) -> dict[str, Any]:
    """Collect only already accepted provider jobs; never submit new work."""
    state = _read_object(root / "RUN.json")
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise UrduAudioProductionError("GOOGLE_API_KEY is missing")
    client = genai.Client(api_key=api_key)
    state["status"] = "quota_blocked_collecting_submitted"
    _save_state(root, state)
    while True:
        active = False
        for batch in state["batches"]:
            if not batch.get("batch_id") or batch["status"] in COLLECTED_BATCH_STATES:
                continue
            if batch["status"] in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
                _collect_ready_batch(client, root, state, batch)
                continue
            if batch["status"] in TERMINAL_STATES:
                batch["last_error"] = f"Provider batch ended in {batch['status']}"
                state["status"] = "blocked"
                _save_state(root, state)
                raise UrduAudioProductionError(batch["last_error"])
            provider_state = _poll_batch(client, root, state, batch)
            if provider_state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
                _collect_ready_batch(client, root, state, batch)
            elif provider_state in TERMINAL_STATES:
                batch["last_error"] = f"Provider batch ended in {provider_state}"
                state["status"] = "blocked"
                _save_state(root, state)
                raise UrduAudioProductionError(batch["last_error"])
            else:
                active = True
        if not active:
            break
        time.sleep(poll_seconds)
    state["status"] = "quota_blocked"
    _save_state(root, state)
    summary = status(root)
    atomic_json(root / "SUBMITTED_BATCHES_COLLECTED.json", summary)
    return summary


def resume_quota_wave(
    root: Path = PRODUCTION_ROOT, *, poll_seconds: int = POLL_SECONDS
) -> dict[str, Any]:
    """Start one later submission wave after every prior accepted job is collected."""
    state = _read_object(root / "RUN.json")
    active = [
        batch
        for batch in state["batches"]
        if batch.get("batch_id") and batch["status"] not in COLLECTED_BATCH_STATES
    ]
    if active:
        raise UrduAudioProductionError("Collect all accepted Batch jobs before quota recovery")
    blocked = [batch for batch in state["batches"] if batch["status"] == "submission_blocked"]
    if len(blocked) != 1:
        raise UrduAudioProductionError("Quota recovery requires exactly one blocked submission")
    batch = blocked[0]
    error = str(batch.get("last_error") or "")
    if "429" not in error or "RESOURCE_EXHAUSTED" not in error:
        raise UrduAudioProductionError("Blocked submission is not an eligible quota failure")
    recoveries = int(state.get("quota_recovery_waves", 0))
    if recoveries >= 8:
        raise UrduAudioProductionError("Quota recovery wave ceiling reached")
    batch.setdefault("failure_history", []).append(
        {
            "status": "submission_blocked",
            "error": error,
            "preserved_at": utc_now(),
            "uploaded_file_name": batch.get("uploaded_file_name"),
        }
    )
    batch["status"] = "prepared"
    batch["last_error"] = None
    batch["uploaded_file_name"] = None
    batch["submission_intent_at"] = None
    batch["batch_create_intent_at"] = None
    state["quota_recovery_waves"] = recoveries + 1
    state["status"] = "prepared_quota_recovery_wave"
    _save_state(root, state)
    return run(root, poll_seconds=poll_seconds)


def run(root: Path = PRODUCTION_ROOT, poll_seconds: int = POLL_SECONDS) -> dict[str, Any]:
    state = prepare(root)
    if state["status"] in {"blocked", "synthesis_complete"}:
        if state["status"] == "blocked":
            raise UrduAudioProductionError("Production is blocked; inspect RUN.json")
        return status(root)
    ambiguous = [
        batch
        for batch in state["batches"]
        if batch["status"] in {"uploading", "uploaded", "creating_batch", "submission_blocked"}
    ]
    if ambiguous:
        raise UrduAudioProductionError(
            f"Ambiguous prior submission state for shard {ambiguous[0]['shard']}; refusing duplicate"
        )
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise UrduAudioProductionError("GOOGLE_API_KEY is missing")
    client = genai.Client(api_key=api_key)
    first_full_batch = 0
    if state.get("canary_status") != "passed_imported":
        state["status"] = "running_canary"
        _save_state(root, state)
        canary = state["batches"][0]
        if canary["status"] == "prepared":
            _submit_batch(client, root, state, canary)
        while canary["status"] not in TERMINAL_STATES and canary["status"] != "collected":
            time.sleep(poll_seconds)
            _poll_batch(client, root, state, canary)
        if canary["status"] in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
            _collect_batch(client, root, state, canary)
        if canary["status"] != "collected":
            state["status"] = "blocked"
            canary["last_error"] = f"Canary ended in {canary['status']}"
            _save_state(root, state)
            raise UrduAudioProductionError(canary["last_error"])
        state["canary_status"] = "passed"
        first_full_batch = 1

    state["status"] = "submitting_full_batch"
    _save_state(root, state)
    for batch in state["batches"][first_full_batch:]:
        if batch["status"] == "prepared":
            _submit_batch(client, root, state, batch)
    state["status"] = "waiting_for_batches"
    _save_state(root, state)

    while True:
        active = False
        for batch in state["batches"][first_full_batch:]:
            if batch["status"] in COLLECTED_BATCH_STATES:
                continue
            if batch["status"] in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
                _collect_ready_batch(client, root, state, batch)
                continue
            if batch["status"] in TERMINAL_STATES:
                state["status"] = "blocked"
                batch["last_error"] = f"Provider batch ended in {batch['status']}"
                _save_state(root, state)
                raise UrduAudioProductionError(batch["last_error"])
            _poll_batch(client, root, state, batch)
            active = True
        if all(batch["status"] in COLLECTED_BATCH_STATES for batch in state["batches"]):
            break
        if active:
            time.sleep(poll_seconds)

    if any(job["status"] != "complete" for job in state["jobs"]):
        state["status"] = "blocked"
        _save_state(root, state)
        raise UrduAudioProductionError(
            f"Not all {len(state['jobs'])} synthesis units completed"
        )
    state["status"] = "synthesis_complete"
    state["completed_at"] = utc_now()
    _save_state(root, state)
    summary = status(root)
    atomic_json(root / "SYNTHESIS_COMPLETE.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "status",
            "run",
            "collect-submitted",
            "prepare-failed-unit-recovery",
            "prepare-fragment-recovery",
            "resume-quota-wave",
            "assemble",
        ),
        nargs="?",
        default="status",
    )
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument("--skip-decode-check", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare()
    elif args.action == "run":
        result = run(poll_seconds=args.poll_seconds)
    elif args.action == "assemble":
        result = assemble(decode_check=not args.skip_decode_check)
    elif args.action == "collect-submitted":
        result = collect_submitted(poll_seconds=args.poll_seconds)
    elif args.action == "prepare-failed-unit-recovery":
        result = prepare_failed_unit_recovery()
    elif args.action == "prepare-fragment-recovery":
        result = prepare_fragment_recovery()
    elif args.action == "resume-quota-wave":
        result = resume_quota_wave(poll_seconds=args.poll_seconds)
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
