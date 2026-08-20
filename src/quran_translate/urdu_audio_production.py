"""Produce the frozen Urdu Quran audiobook with Gemini Batch TTS."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import time
import wave
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
RELEASE_AUDIO_ROOT = OUTPUT_DIR / "audio" / "releases" / PRODUCTION_ID
TRANSLATION_RUN_ROOT = (
    DATA_DIR / "work" / "urdu-production-v1" / "quran-urdu-production-v1-20260818"
)
HARD_ESTIMATED_BATCH_COST_USD = 20.0
SHARD_TARGET_CHARACTERS = 30_000
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
    units: list[dict[str, Any]] = []
    covered: list[tuple[int, int]] = []
    for source_unit in source_units:
        surah = int(source_unit["surah"])
        first = int(source_unit["first_ayah"])
        last = int(source_unit["last_ayah"])
        refs = [(surah, ayah) for ayah in range(first, last + 1)]
        try:
            lines = [by_ref[ref] for ref in refs]
        except KeyError as exc:
            raise UrduAudioProductionError(f"Missing frozen Urdu ayah {exc.args[0]}") from exc
        speech_lines = [
            _speech_ayah(f"{s}:{a}", by_ref[(s, a)]) for s, a in refs
        ]
        text = "\n".join(lines)
        speech_text = "\n".join(speech_lines)
        unit_id = str(source_unit["unit_id"])
        units.append(
            {
                "unit_index": len(units) + 1,
                "unit_id": unit_id,
                "surah": surah,
                "first_ayah": first,
                "last_ayah": last,
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


def _make_shards(units: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    # Fatihah is seeded from the approved pilot. The next unit is isolated as the
    # real Batch transport canary before the remaining jobs are submitted.
    pending = units[1:]
    if not pending:
        return []
    shards = [[pending[0]]]
    current: list[dict[str, Any]] = []
    current_chars = 0
    for unit in pending[1:]:
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
        "version": "quran-urdu-charon-production-input-v1",
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
    shards = _make_shards(units)
    batches: list[dict[str, Any]] = []
    for index, shard in enumerate(shards, start=1):
        text = "\n".join(_request_line(unit) for unit in shard) + "\n"
        path = root / "jobs" / f"shard-{index:03d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(path, text)
        batches.append(
            {
                "shard": index,
                "canary": index == 1,
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
        "jobs": jobs,
        "batches": batches,
    }
    atomic_json(state_path, state)
    atomic_json(
        root / "MANIFEST.json",
        {
            "version": "quran-urdu-charon-production-manifest-v1",
            "production_id": PRODUCTION_ID,
            "input_fingerprint": fingerprint,
            "units": len(units),
            "ayahs": 6_236,
            "shards": len(shards),
            "seeded_units": 1,
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
    batch["status"] = "collected" if not failures else "collection_blocked"
    batch["collected_at"] = utc_now()
    if failures:
        batch["last_error"] = f"Failed units: {', '.join(failures)}"
        state["status"] = "blocked"
    _save_state(root, state)
    if failures:
        raise UrduAudioProductionError(batch["last_error"])


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

    state["status"] = "submitting_full_batch"
    _save_state(root, state)
    for batch in state["batches"][1:]:
        if batch["status"] == "prepared":
            _submit_batch(client, root, state, batch)
    state["status"] = "waiting_for_batches"
    _save_state(root, state)

    while True:
        active = False
        for batch in state["batches"][1:]:
            if batch["status"] in {"collected", "collection_blocked"}:
                continue
            if batch["status"] in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
                _collect_batch(client, root, state, batch)
                continue
            if batch["status"] in TERMINAL_STATES:
                state["status"] = "blocked"
                batch["last_error"] = f"Provider batch ended in {batch['status']}"
                _save_state(root, state)
                raise UrduAudioProductionError(batch["last_error"])
            _poll_batch(client, root, state, batch)
            active = True
        if all(batch["status"] == "collected" for batch in state["batches"]):
            break
        if active:
            time.sleep(poll_seconds)

    if any(job["status"] != "complete" for job in state["jobs"]):
        state["status"] = "blocked"
        _save_state(root, state)
        raise UrduAudioProductionError("Not all 323 synthesis units completed")
    state["status"] = "synthesis_complete"
    state["completed_at"] = utc_now()
    _save_state(root, state)
    summary = status(root)
    atomic_json(root / "SYNTHESIS_COMPLETE.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "status", "run"), nargs="?", default="status")
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare()
    elif args.action == "run":
        result = run(poll_seconds=args.poll_seconds)
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
