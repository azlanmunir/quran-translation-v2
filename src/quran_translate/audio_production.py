"""Release-pinned, resumable production audiobook pipeline."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import OUTPUT_DIR, PROJECT_ROOT, file_sha256, text_sha256
from .db import utc_now
from .elevenlabs_tts import ElevenLabsError, synthesize_text
from .production_packets import atomic_json, atomic_text


RELEASE_VERSION = "quran-translation-v2.4.1"
EXPECTED_FINAL_TEXT_SHA256 = (
    "09e302204ea5ebf90e5e1fcd13ac924495dd7f6bd8585105f1f1933c0557291b"
)
DEFAULT_AUDIO_RUN_ID = "quran-v2.4.1-nathan-v3-production-v1"
DEFAULT_CHUNK_TARGET_CHARS = 3200
MAX_ELEVEN_V3_CHARS = 5000
DEFAULT_CONTEXT_CHARS = 450
DEFAULT_VOICE_ID = "lWDDHwXsJXJM7nv2YgHY"
DEFAULT_MODEL_ID = "eleven_v3"
DEFAULT_SOURCE_FORMAT = "pcm_44100"
DEFAULT_FINAL_BITRATE = "192k"
DEFAULT_PITCH_SEMITONES = -1.25
DEFAULT_COST_PER_THOUSAND_USD = 0.20

RELEASE_ROOT = OUTPUT_DIR / "release" / RELEASE_VERSION
RELEASE_MANIFEST = RELEASE_ROOT / "MANIFEST.json"
LISTENING_EDITION = RELEASE_ROOT / "quran-listening-edition.json"
VOICE_SELECTION = PROJECT_ROOT / "releases" / f"{RELEASE_VERSION}-audio-voice.json"
JUZ_BOUNDARIES = PROJECT_ROOT / "data" / "evidence" / "juz-boundaries-v1.json"


class AudioProductionError(RuntimeError):
    """Raised when production audio inputs or artifacts fail an integrity check."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioProductionError(f"Cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AudioProductionError(f"Expected a JSON object: {path}")
    return payload


def _ref_key(ref: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{1,3}):(\d{1,3})", ref)
    if not match:
        raise AudioProductionError(f"Malformed Quran reference: {ref}")
    return int(match.group(1)), int(match.group(2))


def _release_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not RELEASE_MANIFEST.exists() or not LISTENING_EDITION.exists():
        raise AudioProductionError(f"Missing frozen release artifacts under {RELEASE_ROOT}")
    manifest = _read_object(RELEASE_MANIFEST)
    if manifest.get("qa_passed") is not True:
        raise AudioProductionError("Refusing audio generation from a release that failed QA")
    if manifest.get("final_text_sha256") != EXPECTED_FINAL_TEXT_SHA256:
        raise AudioProductionError("Release text hash is not the approved v2.4.1 hash")
    expected_hash = manifest.get("artifacts", {}).get(LISTENING_EDITION.name)
    if expected_hash != file_sha256(LISTENING_EDITION):
        raise AudioProductionError("Listening edition does not match its release manifest")

    payload = _read_object(LISTENING_EDITION)
    raw_rows = payload.get("ayahs")
    if not isinstance(raw_rows, list) or len(raw_rows) != 6236:
        raise AudioProductionError("Listening edition must contain exactly 6,236 ayahs")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    hash_material: list[str] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise AudioProductionError("Listening edition contains a malformed ayah")
        ref = raw.get("ref")
        translation = raw.get("translation")
        if not isinstance(ref, str) or ref in seen:
            raise AudioProductionError(f"Malformed or duplicate ayah reference: {ref}")
        if not isinstance(translation, str) or not translation.strip():
            raise AudioProductionError(f"Empty translation at {ref}")
        surah, ayah = _ref_key(ref)
        if raw.get("surah") != surah or raw.get("ayah") != ayah:
            raise AudioProductionError(f"Reference fields disagree at {ref}")
        rows.append(raw)
        seen.add(ref)
        hash_material.append(f"{ref}\t{translation}")
    if text_sha256("\n".join(hash_material)) != EXPECTED_FINAL_TEXT_SHA256:
        raise AudioProductionError("Listening edition does not reproduce the approved text hash")
    if [_ref_key(str(row["ref"])) for row in rows] != sorted(
        _ref_key(str(row["ref"])) for row in rows
    ):
        raise AudioProductionError("Listening edition is not in canonical ayah order")
    return rows, manifest


def _juz_assignment(rows: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    payload = _read_object(JUZ_BOUNDARIES)
    boundaries = payload.get("juzs")
    if not isinstance(boundaries, list) or len(boundaries) != 30:
        raise AudioProductionError("Juz metadata must contain exactly 30 ranges")
    refs = [str(row["ref"]) for row in rows]
    positions = {ref: index for index, ref in enumerate(refs)}
    assignment: dict[str, int] = {}
    prior_end = -1
    normalized: list[dict[str, Any]] = []
    for expected_juz, raw in enumerate(boundaries, start=1):
        if not isinstance(raw, dict) or raw.get("juz") != expected_juz:
            raise AudioProductionError(f"Malformed juz metadata at position {expected_juz}")
        start_ref = str(raw.get("start_ref"))
        end_ref = str(raw.get("end_ref"))
        if start_ref not in positions or end_ref not in positions:
            raise AudioProductionError(f"Juz {expected_juz} references are absent from release")
        start = positions[start_ref]
        end = positions[end_ref]
        expected_count = int(raw.get("verse_count", 0))
        if start != prior_end + 1 or end < start or end - start + 1 != expected_count:
            raise AudioProductionError(f"Juz {expected_juz} is not contiguous or has a bad count")
        for ref in refs[start : end + 1]:
            assignment[ref] = expected_juz
        normalized.append(
            {
                "juz": expected_juz,
                "start_ref": start_ref,
                "end_ref": end_ref,
                "verse_count": expected_count,
            }
        )
        prior_end = end
    if prior_end != len(rows) - 1 or len(assignment) != 6236:
        raise AudioProductionError("Juz ranges do not cover the full Quran exactly once")
    return assignment, normalized


def _surah_title(row: dict[str, Any]) -> str:
    return f"Surah {row['surah']}. {str(row['surah_name_en']).strip()}.\n\n"


def plan_chunks(target_chars: int = DEFAULT_CHUNK_TARGET_CHARS) -> dict[str, Any]:
    if target_chars <= 0 or target_chars > MAX_ELEVEN_V3_CHARS:
        raise AudioProductionError(
            f"Chunk target must be between 1 and {MAX_ELEVEN_V3_CHARS} characters"
        )
    rows, release_manifest = _release_rows()
    juz_by_ref, juzs = _juz_assignment(rows)
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_parts: list[str] = []

    def flush() -> None:
        if not current:
            return
        text = "\n".join(current_parts)
        if len(text) > MAX_ELEVEN_V3_CHARS:
            raise AudioProductionError(
                f"Chunk {current[0]['ref']}-{current[-1]['ref']} exceeds Eleven v3 limit"
            )
        first = current[0]
        last = current[-1]
        chunks.append(
            {
                "chunk_index": len(chunks) + 1,
                "surah_number": int(first["surah"]),
                "surah_name": str(first["surah_name_en"]),
                "juz_number": juz_by_ref[str(first["ref"])],
                "start_ref": str(first["ref"]),
                "end_ref": str(last["ref"]),
                "ayah_count": len(current),
                "text": text,
                "text_sha256": text_sha256(text),
                "char_count": len(text),
            }
        )
        current.clear()
        current_parts.clear()

    for row in rows:
        ref = str(row["ref"])
        part = (
            (_surah_title(row) if int(row["ayah"]) == 1 else "")
            + str(row["translation"]).strip()
        )
        if len(part) > MAX_ELEVEN_V3_CHARS:
            raise AudioProductionError(f"Single ayah plus title exceeds model limit at {ref}")
        hard_boundary = bool(current) and (
            int(row["surah"]) != int(current[0]["surah"])
            or juz_by_ref[ref] != juz_by_ref[str(current[0]["ref"])]
        )
        candidate_chars = sum(len(value) for value in current_parts) + len(part)
        if current_parts:
            candidate_chars += len(current_parts)
        if current and (hard_boundary or candidate_chars > target_chars):
            flush()
        current.append(row)
        current_parts.append(part)
    flush()

    covered_refs: list[str] = []
    row_position = {str(row["ref"]): index for index, row in enumerate(rows)}
    for chunk in chunks:
        start = row_position[str(chunk["start_ref"])]
        end = row_position[str(chunk["end_ref"])]
        covered_refs.extend(str(row["ref"]) for row in rows[start : end + 1])
    if covered_refs != [str(row["ref"]) for row in rows]:
        raise AudioProductionError("Chunk plan does not cover the release exactly once")

    return {
        "release_manifest": release_manifest,
        "juzs": juzs,
        "chunks": chunks,
        "ayah_count": len(rows),
        "billable_characters": sum(int(chunk["char_count"]) for chunk in chunks),
    }


def _tool_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise AudioProductionError(f"Required audio tool failed: {' '.join(command)}") from exc
    return (result.stdout.strip() or result.stderr.strip()).splitlines()[0]


def _run_root(audio_run_id: str) -> Path:
    return OUTPUT_DIR / "audio" / "runs" / audio_run_id


def _source_extension(source_format: str) -> str:
    if source_format.startswith("pcm_"):
        return ".pcm"
    if source_format.startswith("mp3_"):
        return ".mp3"
    raise AudioProductionError(f"Unsupported ElevenLabs source format: {source_format}")


def prepare_audio_production(
    *,
    audio_run_id: str = DEFAULT_AUDIO_RUN_ID,
    target_chars: int = DEFAULT_CHUNK_TARGET_CHARS,
    source_format: str = DEFAULT_SOURCE_FORMAT,
    cost_per_thousand_usd: float = DEFAULT_COST_PER_THOUSAND_USD,
) -> dict[str, Any]:
    root = _run_root(audio_run_id)
    state_path = root / "RUN.json"
    plan = plan_chunks(target_chars)
    selection = _read_object(VOICE_SELECTION)
    if selection.get("final_text_sha256") != EXPECTED_FINAL_TEXT_SHA256:
        raise AudioProductionError("Voice selection is not pinned to the approved text")
    narration = selection.get("narration", {})
    pitch = selection.get("pitch_processing", {})
    if narration.get("voice_id") != DEFAULT_VOICE_ID or narration.get("model_id") != DEFAULT_MODEL_ID:
        raise AudioProductionError("Voice selection does not match the approved Nathan v3 narrator")
    if float(pitch.get("pitch_semitones")) != DEFAULT_PITCH_SEMITONES:
        raise AudioProductionError("Voice selection does not match the approved pitch treatment")

    rubberband_version = _tool_version(["rubberband", "--version"])
    ffmpeg_version = _tool_version(["ffmpeg", "-version"])
    if not rubberband_version.startswith("4."):
        raise AudioProductionError(f"Unreviewed Rubber Band version: {rubberband_version}")

    extension = _source_extension(source_format)
    job_specs: list[dict[str, Any]] = []
    input_payloads: list[tuple[Path, str, str]] = []
    for chunk in plan["chunks"]:
        index = int(chunk["chunk_index"])
        stem = f"{index:04d}-{chunk['start_ref'].replace(':', '_')}-{chunk['end_ref'].replace(':', '_')}"
        input_path = root / "inputs" / f"{stem}.txt"
        raw_path = root / "raw" / f"{stem}{extension}"
        source_wav_path = root / "work" / f"{stem}-source.wav"
        master_wav_path = root / "masters-wav" / f"{stem}.wav"
        master_mp3_path = root / "masters-mp3" / f"{stem}.mp3"
        input_payloads.append((input_path, str(chunk["text"]), str(chunk["text_sha256"])))
        job_specs.append(
            {
                key: value for key, value in chunk.items() if key != "text"
            }
            | {
                "job_id": f"{audio_run_id}-{index:04d}",
                "input_path": str(input_path),
                "raw_path": str(raw_path),
                "source_wav_path": str(source_wav_path),
                "master_wav_path": str(master_wav_path),
                "master_mp3_path": str(master_mp3_path),
            }
        )

    fingerprint_material = {
        "version": "quran-audio-production-v1",
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "release_manifest_sha256": file_sha256(RELEASE_MANIFEST),
        "listening_edition_sha256": file_sha256(LISTENING_EDITION),
        "voice_selection_sha256": file_sha256(VOICE_SELECTION),
        "juz_boundaries_sha256": file_sha256(JUZ_BOUNDARIES),
        "production_runner_sha256": file_sha256(Path(__file__)),
        "elevenlabs_client_sha256": file_sha256(Path(__file__).with_name("elevenlabs_tts.py")),
        "voice_id": DEFAULT_VOICE_ID,
        "model_id": DEFAULT_MODEL_ID,
        "source_format": source_format,
        "required_subscription_tier": "Pro or above",
        "pitch_semitones": DEFAULT_PITCH_SEMITONES,
        "target_chars": target_chars,
        "jobs": [
            {
                "job_id": job["job_id"],
                "start_ref": job["start_ref"],
                "end_ref": job["end_ref"],
                "text_sha256": job["text_sha256"],
                "char_count": job["char_count"],
            }
            for job in job_specs
        ],
    }
    fingerprint = text_sha256(
        json.dumps(fingerprint_material, ensure_ascii=False, sort_keys=True)
    )
    if state_path.exists():
        state = _read_object(state_path)
        if state.get("input_fingerprint") != fingerprint:
            raise AudioProductionError("Audio inputs changed; refusing a mixed-version resume")
        migrated = False
        for job in state.get("jobs", []):
            input_path = Path(str(job["input_path"]))
            if not input_path.exists() or text_sha256(input_path.read_text(encoding="utf-8")) != job["text_sha256"]:
                raise AudioProductionError(f"Audio input integrity failure: {input_path}")
            if job.get("status") == "quota_paused" and job.get("last_error"):
                corrected = _provider_blocker_status(str(job["last_error"]))
                if corrected and corrected != "quota_paused":
                    job["status"] = corrected
                    state["status"] = corrected
                    migrated = True
        if migrated:
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
        return state

    for input_path, text, expected_hash in input_payloads:
        if input_path.exists() and text_sha256(input_path.read_text(encoding="utf-8")) != expected_hash:
            raise AudioProductionError(f"Existing chunk input changed: {input_path}")
        atomic_text(input_path, text)

    billable = int(plan["billable_characters"])
    jobs = [
        job
        | {
            "status": "pending",
            "attempts": 0,
            "processing_attempts": 0,
            "raw_sha256": None,
            "master_wav_sha256": None,
            "master_mp3_sha256": None,
            "source_duration_seconds": None,
            "duration_seconds": None,
            "bytes": None,
            "last_error": None,
        }
        for job in job_specs
    ]
    state = {
        "version": "quran-audio-production-run-v1",
        "audio_run_id": audio_run_id,
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "input_fingerprint": fingerprint,
        "prepared_at": utc_now(),
        "updated_at": utc_now(),
        "status": "prepared",
        "provider": "ElevenLabs",
        "voice_id": DEFAULT_VOICE_ID,
        "voice_label": narration.get("voice_label"),
        "model_id": DEFAULT_MODEL_ID,
        "source_format": source_format,
        "final_format": "mp3_44100_192_local_encode",
        "chunk_target_chars": target_chars,
        "context_chars": DEFAULT_CONTEXT_CHARS,
        "pitch_processing": {
            "tool": "rubberband",
            "version": rubberband_version,
            "engine": "R3 fine",
            "formant_preservation": True,
            "pitch_semitones": DEFAULT_PITCH_SEMITONES,
            "global_time_stretch": 1.0,
        },
        "ffmpeg_version": ffmpeg_version,
        "cost": {
            "billable_characters": billable,
            "credits_required": billable,
            "cost_per_thousand_usd": cost_per_thousand_usd,
            "one_pass_estimate_usd": round(billable / 1000 * cost_per_thousand_usd, 2),
            "retry_reserve_10_percent_credits": round(billable * 0.10),
        },
        "juzs": plan["juzs"],
        "jobs": jobs,
    }
    atomic_json(state_path, state)
    return state


def production_audio_status(audio_run_id: str = DEFAULT_AUDIO_RUN_ID) -> dict[str, Any]:
    state = _read_object(_run_root(audio_run_id) / "RUN.json")
    counts: dict[str, int] = defaultdict(int)
    chars: dict[str, int] = defaultdict(int)
    duration = 0.0
    for job in state.get("jobs", []):
        status = str(job.get("status", "unknown"))
        counts[status] += 1
        chars[status] += int(job.get("char_count", 0))
        duration += float(job.get("duration_seconds") or 0.0)
    return {
        "audio_run_id": audio_run_id,
        "status": state.get("status"),
        "release_version": state.get("release_version"),
        "final_text_sha256": state.get("final_text_sha256"),
        "jobs": len(state.get("jobs", [])),
        "billable_characters": state.get("cost", {}).get("billable_characters"),
        "one_pass_estimate_usd": state.get("cost", {}).get("one_pass_estimate_usd"),
        "job_status": dict(sorted(counts.items())),
        "characters_by_status": dict(sorted(chars.items())),
        "generated_duration_seconds": round(duration, 3),
    }


def _ffprobe(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,bit_rate:format=duration,size,bit_rate",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise AudioProductionError(f"ffprobe failed for {path}") from exc
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise AudioProductionError(f"No audio stream in {path}")
    stream = streams[0]
    fmt = payload.get("format", {})
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "bit_rate": int(stream.get("bit_rate") or fmt.get("bit_rate") or 0),
        "duration_seconds": float(fmt.get("duration") or 0.0),
        "bytes": int(fmt.get("size") or path.stat().st_size),
    }


def _run_audio_command(command: list[str], error: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace")[-1000:]
        raise AudioProductionError(f"{error}: {detail}") from exc


def _decode_source(raw: Path, destination: Path, source_format: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp.wav")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if source_format == "pcm_44100":
        command.extend(["-f", "s16le", "-ar", "44100", "-ac", "1"])
    command.extend(
        [
            "-i",
            str(raw),
            "-ar",
            "44100",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(temp),
        ]
    )
    _run_audio_command(command, f"Could not decode ElevenLabs source {raw}")
    os.replace(temp, destination)


def _pitch_and_encode(job: dict[str, Any], source_format: str) -> dict[str, Any]:
    raw = Path(str(job["raw_path"]))
    source_wav = Path(str(job["source_wav_path"]))
    master_wav = Path(str(job["master_wav_path"]))
    master_mp3 = Path(str(job["master_mp3_path"]))
    _decode_source(raw, source_wav, source_format)
    source_info = _ffprobe(source_wav)

    master_wav.parent.mkdir(parents=True, exist_ok=True)
    wav_temp = master_wav.with_name(f".{master_wav.name}.tmp.wav")
    _run_audio_command(
        [
            "rubberband",
            "--quiet",
            "--fine",
            "--formant",
            "--pitch",
            str(DEFAULT_PITCH_SEMITONES),
            str(source_wav),
            str(wav_temp),
        ],
        f"Pitch processing failed for {job['job_id']}",
    )
    os.replace(wav_temp, master_wav)
    master_wav_info = _ffprobe(master_wav)
    duration_ratio = master_wav_info["duration_seconds"] / source_info["duration_seconds"]
    if not 0.98 <= duration_ratio <= 1.02:
        raise AudioProductionError(
            f"Pitch processing changed duration unexpectedly for {job['job_id']}: {duration_ratio:.4f}"
        )

    master_mp3.parent.mkdir(parents=True, exist_ok=True)
    mp3_temp = master_mp3.with_name(f".{master_mp3.name}.tmp.mp3")
    _run_audio_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(master_wav),
            "-ar",
            "44100",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            DEFAULT_FINAL_BITRATE,
            str(mp3_temp),
        ],
        f"MP3 encoding failed for {job['job_id']}",
    )
    os.replace(mp3_temp, master_mp3)
    mp3_info = _ffprobe(master_mp3)
    if (
        mp3_info["codec"] != "mp3"
        or mp3_info["sample_rate"] != 44100
        or mp3_info["channels"] != 1
        or mp3_info["duration_seconds"] <= 0
    ):
        raise AudioProductionError(f"Master MP3 failed format QA: {master_mp3}")
    return {
        "raw_sha256": file_sha256(raw),
        "master_wav_sha256": file_sha256(master_wav),
        "master_mp3_sha256": file_sha256(master_mp3),
        "source_duration_seconds": source_info["duration_seconds"],
        "duration_seconds": mp3_info["duration_seconds"],
        "bytes": master_mp3.stat().st_size,
    }


def _provider_blocker_status(message: str) -> str | None:
    lowered = message.lower()
    if any(token in lowered for token in ("subscription_required", "output_format_not_allowed")):
        return "provider_blocked"
    if any(token in lowered for token in ("unauthorized", "authentication", "invalid api key", "401 ")):
        return "authentication_blocked"
    if any(token in lowered for token in ("billing", "payment", "402 ")):
        return "billing_blocked"
    if any(token in lowered for token in ("quota", "credit", "limit exceeded")):
        return "quota_paused"
    if any(token in lowered for token in ("permission", "403 ")):
        return "provider_blocked"
    return None


def _context_for_job(jobs: list[dict[str, Any]], index: int, context_chars: int) -> tuple[str | None, str | None]:
    current = jobs[index]
    previous: str | None = None
    following: str | None = None
    if index > 0 and jobs[index - 1]["surah_number"] == current["surah_number"]:
        previous = Path(str(jobs[index - 1]["input_path"])).read_text(encoding="utf-8")[-context_chars:]
    if index + 1 < len(jobs) and jobs[index + 1]["surah_number"] == current["surah_number"]:
        following = Path(str(jobs[index + 1]["input_path"])).read_text(encoding="utf-8")[:context_chars]
    return previous, following


def synthesize_audio_production(
    *,
    audio_run_id: str = DEFAULT_AUDIO_RUN_ID,
    limit: int | None = None,
    max_attempts: int = 2,
    request_timeout_seconds: int = 600,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    state_path = _run_root(audio_run_id) / "RUN.json"
    state = _read_object(state_path)
    if state.get("final_text_sha256") != EXPECTED_FINAL_TEXT_SHA256:
        raise AudioProductionError("Audio run is not pinned to the approved release")
    jobs = state.get("jobs")
    if not isinstance(jobs, list):
        raise AudioProductionError("Audio run has no job list")
    processed = 0
    state["status"] = "generating"
    state["updated_at"] = utc_now()
    atomic_json(state_path, state)

    for index, job in enumerate(jobs):
        if job.get("status") == "complete":
            continue
        if limit is not None and processed >= limit:
            break
        input_path = Path(str(job["input_path"]))
        text = input_path.read_text(encoding="utf-8")
        if text_sha256(text) != job["text_sha256"] or len(text) != job["char_count"]:
            raise AudioProductionError(f"Chunk input integrity failure: {input_path}")
        master_mp3 = Path(str(job["master_mp3_path"]))
        raw_path = Path(str(job["raw_path"]))
        try:
            if master_mp3.exists() and master_mp3.stat().st_size > 1024:
                recovered = _ffprobe(master_mp3)
                if recovered["codec"] != "mp3" or recovered["duration_seconds"] <= 0:
                    raise AudioProductionError(f"Existing master is invalid: {master_mp3}")
                job.update(
                    {
                        "status": "complete",
                        "master_mp3_sha256": file_sha256(master_mp3),
                        "duration_seconds": recovered["duration_seconds"],
                        "bytes": master_mp3.stat().st_size,
                        "last_error": None,
                    }
                )
                processed += 1
                state["updated_at"] = utc_now()
                atomic_json(state_path, state)
                continue

            if not raw_path.exists() or raw_path.stat().st_size <= 1024:
                previous, following = _context_for_job(
                    jobs, index, int(state.get("context_chars", DEFAULT_CONTEXT_CHARS))
                )
                last_error = ""
                for attempt in range(max_attempts):
                    job["attempts"] = int(job.get("attempts", 0)) + 1
                    job["status"] = "requesting"
                    state["updated_at"] = utc_now()
                    atomic_json(state_path, state)
                    temp = raw_path.with_name(f".{raw_path.name}.attempt-{attempt + 1}.tmp")
                    temp.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        synthesize_text(
                            text=text,
                            voice_id=str(state["voice_id"]),
                            output_path=temp,
                            model_id=str(state["model_id"]),
                            output_format=str(state["source_format"]),
                            previous_text=previous,
                            next_text=following,
                            seed=int(str(job["text_sha256"])[:8], 16),
                            request_timeout_seconds=request_timeout_seconds,
                        )
                        if temp.stat().st_size <= 1024:
                            raise AudioProductionError("ElevenLabs returned implausibly small audio")
                        os.replace(temp, raw_path)
                        job.update(
                            {
                                "status": "generated",
                                "raw_sha256": file_sha256(raw_path),
                                "last_error": None,
                            }
                        )
                        state["updated_at"] = utc_now()
                        atomic_json(state_path, state)
                        break
                    except (ElevenLabsError, AudioProductionError) as exc:
                        temp.unlink(missing_ok=True)
                        last_error = str(exc)
                        job["last_error"] = last_error[:4000]
                        state["updated_at"] = utc_now()
                        atomic_json(state_path, state)
                        if _provider_blocker_status(last_error):
                            raise
                        if attempt + 1 < max_attempts:
                            time.sleep(2)
                else:
                    raise AudioProductionError(last_error or "ElevenLabs generation failed")

            job["processing_attempts"] = int(job.get("processing_attempts", 0)) + 1
            job["status"] = "processing"
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            result = _pitch_and_encode(job, str(state["source_format"]))
            job.update(result)
            job["status"] = "complete"
            job["last_error"] = None
            processed += 1
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            print(
                f"complete {sum(j.get('status') == 'complete' for j in jobs)}/{len(jobs)} "
                f"{job['start_ref']}-{job['end_ref']} chars={job['char_count']}",
                flush=True,
            )
            if sleep_seconds:
                time.sleep(sleep_seconds)
        except (ElevenLabsError, AudioProductionError, OSError) as exc:
            job["last_error"] = str(exc)[:4000]
            blocker_status = _provider_blocker_status(str(exc))
            if blocker_status:
                job["status"] = blocker_status
                state["status"] = blocker_status
                state["updated_at"] = utc_now()
                atomic_json(state_path, state)
                break
            job["status"] = "failed"
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)

    statuses = {str(job.get("status")) for job in jobs}
    if statuses == {"complete"}:
        state["status"] = "generated"
    elif statuses & {
        "quota_paused",
        "billing_blocked",
        "authentication_blocked",
        "provider_blocked",
    }:
        state["status"] = sorted(
            statuses
            & {
                "quota_paused",
                "billing_blocked",
                "authentication_blocked",
                "provider_blocked",
            }
        )[0]
    elif "failed" in statuses:
        state["status"] = "failed"
    else:
        state["status"] = "generating"
    state["updated_at"] = utc_now()
    atomic_json(state_path, state)
    return production_audio_status(audio_run_id)


def _ffconcat_line(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


def _concat_mp3(
    inputs: list[Path],
    output: Path,
    *,
    title: str,
    track: int | None = None,
) -> None:
    if not inputs:
        raise AudioProductionError(f"No sources for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    list_path = output.with_name(f".{output.name}.concat.txt")
    temp = output.with_name(f".{output.name}.tmp.mp3")
    atomic_text(list_path, "\n".join(_ffconcat_line(path) for path in inputs) + "\n")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-map_metadata",
        "-1",
        "-c",
        "copy",
        "-id3v2_version",
        "3",
        "-metadata",
        f"title={title}",
        "-metadata",
        "album=The Quran - English Listening Edition",
        "-metadata",
        "artist=Narrated with ElevenLabs Nathan",
    ]
    if track is not None:
        command.extend(["-metadata", f"track={track}"])
    command.append(str(temp))
    try:
        _run_audio_command(command, f"Could not assemble {output}")
        os.replace(temp, output)
    finally:
        list_path.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _output_record(
    *,
    output_type: str,
    output_id: str,
    label: str,
    start_ref: str,
    end_ref: str,
    path: Path,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    info = _ffprobe(path)
    expected_duration = sum(float(job["duration_seconds"]) for job in jobs)
    tolerance = max(2.0, expected_duration * 0.003)
    if abs(info["duration_seconds"] - expected_duration) > tolerance:
        raise AudioProductionError(
            f"Assembled duration mismatch for {path}: {info['duration_seconds']:.3f} vs {expected_duration:.3f}"
        )
    return {
        "type": output_type,
        "id": output_id,
        "label": label,
        "range": f"{start_ref}-{end_ref}",
        "path": str(path),
        "sha256": file_sha256(path),
        **info,
        "source_jobs": [str(job["job_id"]) for job in jobs],
    }


def _fixed_duration_groups(
    jobs: list[dict[str, Any]], target_seconds: float
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    duration = 0.0
    for job in jobs:
        job_duration = float(job["duration_seconds"])
        if current and duration + job_duration > target_seconds:
            without = abs(target_seconds - duration)
            with_next = abs(target_seconds - (duration + job_duration))
            if without <= with_next:
                groups.append(current)
                current = []
                duration = 0.0
        current.append(job)
        duration += job_duration
        if duration >= target_seconds:
            groups.append(current)
            current = []
            duration = 0.0
    if current:
        groups.append(current)
    return groups


def _decode_check(path: Path) -> None:
    _run_audio_command(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-v", "error", "-i", str(path), "-f", "null", "-"],
        f"Decode check failed for {path}",
    )


def assemble_audio_release(
    *,
    audio_run_id: str = DEFAULT_AUDIO_RUN_ID,
    include_fixed_tracks: bool = False,
    fixed_track_minutes: float = 40.0,
    decode_check: bool = False,
) -> dict[str, Any]:
    state = _read_object(_run_root(audio_run_id) / "RUN.json")
    jobs = state.get("jobs")
    if not isinstance(jobs, list) or not jobs or any(job.get("status") != "complete" for job in jobs):
        raise AudioProductionError("All production chunks must be complete before assembly")
    for job in jobs:
        path = Path(str(job["master_mp3_path"]))
        if not path.exists() or file_sha256(path) != job.get("master_mp3_sha256"):
            raise AudioProductionError(f"Master chunk integrity failure: {path}")

    release_root = OUTPUT_DIR / "audio" / "releases" / audio_run_id
    outputs: list[dict[str, Any]] = []
    by_surah: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_juz: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        by_surah[int(job["surah_number"])].append(job)
        by_juz[int(job["juz_number"])].append(job)

    for surah in range(1, 115):
        group = by_surah[surah]
        name = str(group[0]["surah_name"])
        output = release_root / "by-surah" / f"{surah:03d}-{_slug(name)}.mp3"
        _concat_mp3(
            [Path(str(job["master_mp3_path"])) for job in group],
            output,
            title=f"Surah {surah}: {name}",
            track=surah,
        )
        outputs.append(
            _output_record(
                output_type="surah",
                output_id=f"{surah:03d}",
                label=name,
                start_ref=str(group[0]["start_ref"]),
                end_ref=str(group[-1]["end_ref"]),
                path=output,
                jobs=group,
            )
        )

    for juz in range(1, 31):
        group = by_juz[juz]
        output = release_root / "by-juz" / f"juz-{juz:02d}.mp3"
        _concat_mp3(
            [Path(str(job["master_mp3_path"])) for job in group],
            output,
            title=f"Juz {juz}",
            track=juz,
        )
        outputs.append(
            _output_record(
                output_type="juz",
                output_id=f"{juz:02d}",
                label=f"Juz {juz}",
                start_ref=str(group[0]["start_ref"]),
                end_ref=str(group[-1]["end_ref"]),
                path=output,
                jobs=group,
            )
        )

    full_output = release_root / "full-book" / "quran-english-listening-edition.mp3"
    _concat_mp3(
        [Path(str(job["master_mp3_path"])) for job in jobs],
        full_output,
        title="The Quran - English Listening Edition",
    )
    outputs.append(
        _output_record(
            output_type="full_book",
            output_id="full",
            label="The Quran - English Listening Edition",
            start_ref=str(jobs[0]["start_ref"]),
            end_ref=str(jobs[-1]["end_ref"]),
            path=full_output,
            jobs=jobs,
        )
    )

    if include_fixed_tracks:
        groups = _fixed_duration_groups(jobs, fixed_track_minutes * 60)
        for index, group in enumerate(groups, start=1):
            output = release_root / "listening-tracks" / f"track-{index:02d}.mp3"
            _concat_mp3(
                [Path(str(job["master_mp3_path"])) for job in group],
                output,
                title=f"Listening Track {index}",
                track=index,
            )
            outputs.append(
                _output_record(
                    output_type="fixed_track",
                    output_id=f"{index:02d}",
                    label=f"Listening Track {index}",
                    start_ref=str(group[0]["start_ref"]),
                    end_ref=str(group[-1]["end_ref"]),
                    path=output,
                    jobs=group,
                )
            )

    if decode_check:
        for output in outputs:
            _decode_check(Path(str(output["path"])))

    payload = {
        "version": "quran-audio-release-v1",
        "created_at": utc_now(),
        "audio_run_id": audio_run_id,
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "input_fingerprint": state["input_fingerprint"],
        "voice": {
            "provider": state["provider"],
            "model_id": state["model_id"],
            "voice_id": state["voice_id"],
            "voice_label": state["voice_label"],
            "pitch_processing": state["pitch_processing"],
        },
        "counts": {
            "master_chunks": len(jobs),
            "surahs": 114,
            "juzs": 30,
            "full_book": 1,
            "fixed_tracks": sum(output["type"] == "fixed_track" for output in outputs),
        },
        "qa": {
            "all_master_hashes_verified": True,
            "all_output_durations_verified": True,
            "full_decode_check": decode_check,
        },
        "outputs": outputs,
    }
    atomic_json(release_root / "RELEASE_MANIFEST.json", payload)
    return payload
