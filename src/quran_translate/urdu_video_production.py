"""Resumable local production for synchronized Urdu Quran YouTube videos."""

from __future__ import annotations

from .catalog_cache import catalog_contract, validate_cached_catalog

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import mlx_whisper

from .config import OUTPUT_DIR, PROJECT_ROOT, file_sha256, text_sha256
from .db import utc_now
from .production_packets import atomic_json, atomic_text
from .urdu_video_render import build_urdu_display_events, render_urdu_frames
from .video_alignment import AlignmentError
from .video_production import _probe_video, write_catalog_srt
from .video_render import (
    render_video,
    validate_encoded_timeline,
    validate_srt_identity,
    validate_video,
    write_mobile_review,
    write_srt,
)


DEFAULT_RUN_ID = "quran-urdu-youtube-production-v1"
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get("QURAN_SOURCE_ROOT", str(PROJECT_ROOT))
).expanduser()
AUDIO_RUN_RELATIVE = Path("data/work/audio-urdu-v1/quran-urdu-charon-production-v1")
PUBLICATION_RELATIVE = Path("output/publication/quran-publication.json")
POLICY_PATH = PROJECT_ROOT / "configs" / "urdu_video_release_v1.json"
MIN_MAPPED_WORD_COVERAGE = 0.95
MIN_AYAH_WORD_COVERAGE = 0.90
MAX_INTERPOLATED_WORDS = 2
MAX_INTERPOLATED_GAP_SECONDS = 1.0
URDU_LETTER_FORMS = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "ة": "ہ",
        "ه": "ہ",
        "ۀ": "ہ",
        "ؤ": "و",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
    }
)


class UrduVideoProductionError(RuntimeError):
    """Raised when Urdu video production would violate an integrity contract."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UrduVideoProductionError(f"Cannot read JSON {path}: {exc}") from exc


def _run_root(run_id: str) -> Path:
    return OUTPUT_DIR / "video" / "runs" / run_id


def _implementation_hashes() -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "src/quran_translate/video_alignment.py",
        PROJECT_ROOT / "src/quran_translate/video_render.py",
        PROJECT_ROOT / "src/quran_translate/urdu_video_render.py",
        PROJECT_ROOT / "src/quran_translate/urdu_video_production.py",
        PROJECT_ROOT / "src/quran_translate/catalog_cache.py",
        POLICY_PATH,
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise UrduVideoProductionError(f"Missing implementation inputs: {missing}")
    return {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in paths}


def _source_paths(source_root: Path) -> tuple[Path, Path, Path]:
    audio_root = source_root / AUDIO_RUN_RELATIVE
    return audio_root, audio_root / "UNITS.json", audio_root / "RUN.json"


def _load_sources(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    audio_root, units_path, audio_run_path = _source_paths(source_root)
    units = _read_json(units_path)
    audio_run = _read_json(audio_run_path)
    publication = _read_json(source_root / PUBLICATION_RELATIVE)
    if not isinstance(units, list) or len(units) != 343:
        raise UrduVideoProductionError("Urdu source must contain exactly 343 canonical units")
    if not isinstance(audio_run, dict) or len(audio_run.get("jobs", [])) != 343:
        raise UrduVideoProductionError("Urdu audio run must contain exactly 343 jobs")
    rows = publication.get("ayahs") if isinstance(publication, dict) else None
    if not isinstance(rows, list) or len(rows) != 6236:
        raise UrduVideoProductionError("Publication metadata must contain 6,236 ayahs")

    audio_jobs = {str(row["unit_id"]): row for row in audio_run["jobs"]}
    failures: list[str] = []
    for expected, unit in enumerate(units, start=1):
        unit_id = str(unit.get("unit_id", ""))
        refs = unit.get("refs")
        text_lines = str(unit.get("text", "")).splitlines()
        speech_lines = str(unit.get("speech_text", "")).splitlines()
        audio_job = audio_jobs.get(unit_id)
        if int(unit.get("unit_index", 0)) != expected:
            failures.append(f"unit-index:{expected}")
        if not isinstance(refs, list) or len(refs) != len(text_lines) or len(refs) != len(speech_lines):
            failures.append(f"line-contract:{unit_id}")
        if audio_job is None or audio_job.get("status") != "complete":
            failures.append(f"audio-state:{unit_id}")
            continue
        audio_path = Path(str(audio_job.get("normalized_path", "")))
        if not audio_path.is_file() or file_sha256(audio_path) != audio_job.get("normalized_sha256"):
            failures.append(f"audio-hash:{unit_id}")
        if text_sha256(str(unit.get("speech_text", ""))) != unit.get("speech_text_sha256"):
            failures.append(f"speech-hash:{unit_id}")
    if failures:
        raise UrduVideoProductionError(
            f"Urdu source verification failed ({len(failures)}): {failures[:12]}"
        )
    return units, audio_run, publication


def _immutable(source_root: Path, model: str) -> dict[str, Any]:
    audio_root, units_path, audio_run_path = _source_paths(source_root)
    return {
        "source_root": str(source_root),
        "audio_root": str(audio_root),
        "units_path": str(units_path),
        "units_sha256": file_sha256(units_path),
        "audio_run_path": str(audio_run_path),
        "audio_run_sha256": file_sha256(audio_run_path),
        "publication_path": str(source_root / PUBLICATION_RELATIVE),
        "publication_sha256": file_sha256(source_root / PUBLICATION_RELATIVE),
        "policy_path": str(POLICY_PATH),
        "policy_sha256": file_sha256(POLICY_PATH),
        "alignment_model": model,
        "implementation_hashes": _implementation_hashes(),
    }


def prepare(
    *,
    run_id: str = DEFAULT_RUN_ID,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    units, _, _ = _load_sources(source_root)
    immutable = _immutable(source_root, model)
    root = _run_root(run_id)
    state_path = root / "RUN.json"
    if state_path.is_file():
        state = _read_json(state_path)
        if state.get("immutable") != immutable:
            raise UrduVideoProductionError(
                "Urdu video inputs changed; refusing a mixed-version resume"
            )
        return state
    root.mkdir(parents=True, exist_ok=True)
    state = {
        "version": "quran-urdu-video-production-run-v1",
        "run_id": run_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "prepared",
        "immutable": immutable,
        "jobs": [
            {
                "unit_index": int(unit["unit_index"]),
                "unit_id": str(unit["unit_id"]),
                "alignment": "pending",
                "render": "pending",
                "attempts": 0,
            }
            for unit in units
        ],
        "catalogs": {"paras": "pending", "surahs": "pending"},
        "provider_cost_usd": 0.0,
    }
    atomic_json(state_path, state)
    return state


def migrate_implementation(run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    """Adopt current renderer hashes without changing frozen source inputs."""
    root = _run_root(run_id)
    state = _read_json(root / "RUN.json")
    immutable = state.get("immutable")
    if not isinstance(immutable, dict):
        raise UrduVideoProductionError("RUN.json has no immutable input block")

    source_root = Path(str(immutable["source_root"]))
    expected = _immutable(source_root, str(immutable["alignment_model"]))
    frozen_inputs = dict(immutable)
    current_inputs = dict(expected)
    old_hashes = frozen_inputs.pop("implementation_hashes", None)
    new_hashes = current_inputs.pop("implementation_hashes", None)
    if frozen_inputs != current_inputs:
        raise UrduVideoProductionError(
            "Urdu video source inputs changed; implementation-only migration refused"
        )
    if not isinstance(old_hashes, dict) or not isinstance(new_hashes, dict):
        raise UrduVideoProductionError("Implementation hash block is invalid")
    if old_hashes == new_hashes:
        return status(run_id)

    marker = {
        "version": "quran-urdu-video-implementation-migration-v1",
        "run_id": run_id,
        "migrated_at": utc_now(),
        "old_implementation_hashes": old_hashes,
        "new_implementation_hashes": new_hashes,
        "source_inputs_unchanged": True,
    }
    atomic_json(root / "MIGRATION_IMPLEMENTATION_v1.json", marker)
    state["immutable"]["implementation_hashes"] = new_hashes
    _save(root, state)
    return status(run_id)


def adopt_audio_content_repair(
    *,
    run_id: str = DEFAULT_RUN_ID,
    unit_index: int,
) -> dict[str, Any]:
    """Adopt one audited source-audio repair without invalidating unrelated work."""
    root = _run_root(run_id)
    state = _read_json(root / "RUN.json")
    immutable = state.get("immutable")
    if not isinstance(immutable, dict):
        raise UrduVideoProductionError("RUN.json has no immutable input block")
    if not 1 <= unit_index <= len(state.get("jobs", [])):
        raise UrduVideoProductionError(f"Invalid unit index: {unit_index}")

    source_root = Path(str(immutable["source_root"]))
    expected = _immutable(source_root, str(immutable["alignment_model"]))
    frozen_inputs = dict(immutable)
    current_inputs = dict(expected)
    old_audio_run_sha256 = frozen_inputs.pop("audio_run_sha256", None)
    new_audio_run_sha256 = current_inputs.pop("audio_run_sha256", None)
    old_implementation_hashes = frozen_inputs.pop("implementation_hashes", None)
    new_implementation_hashes = current_inputs.pop("implementation_hashes", None)
    if frozen_inputs != current_inputs:
        raise UrduVideoProductionError(
            "Audio-repair adoption found unrelated immutable input changes"
        )
    if old_audio_run_sha256 == new_audio_run_sha256:
        raise UrduVideoProductionError("Audio RUN.json has not changed")

    units, audio_run, _ = _load_sources(source_root)
    unit = units[unit_index - 1]
    video_job = state["jobs"][unit_index - 1]
    if int(unit["unit_index"]) != unit_index or int(video_job["unit_index"]) != unit_index:
        raise UrduVideoProductionError(f"Unit index drift at {unit_index}")
    if video_job.get("alignment") == "complete":
        raise UrduVideoProductionError("Repaired source unit already has a complete alignment")

    audio_jobs = _audio_index(audio_run)
    audio_job = audio_jobs[str(unit["unit_id"])]
    repair_marker = (
        Path(str(immutable["audio_root"]))
        / "content-repairs"
        / str(unit["unit_id"])
        / "attempt-0001"
        / "CONTENT_REPAIR_COMPLETE.json"
    )
    if not repair_marker.is_file():
        raise UrduVideoProductionError(f"Audio content-repair marker is missing: {repair_marker}")
    repair = _read_json(repair_marker)
    if (
        repair.get("status") != "complete"
        or repair.get("unit_id") != unit["unit_id"]
        or repair.get("new_master_sha256") != audio_job.get("normalized_sha256")
    ):
        raise UrduVideoProductionError("Audio content-repair marker does not match the master")

    validation = repair.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise UrduVideoProductionError("Audio content repair has no passing alignment validation")
    validated_alignment_path = Path(str(validation.get("normalized_alignment", "")))
    if (
        not validated_alignment_path.is_file()
        or file_sha256(validated_alignment_path)
        != validation.get("normalized_alignment_sha256")
    ):
        raise UrduVideoProductionError("Validated repair alignment is missing or changed")
    validated_alignment = _read_json(validated_alignment_path)
    master_path = Path(str(audio_job["normalized_path"]))
    _validate_alignment(unit, validated_alignment, master_path)

    verified_alignments = 0
    for prior_unit, prior_job in zip(units, state["jobs"]):
        if prior_job.get("alignment") != "complete":
            continue
        alignment_path = root / "alignments" / f"{int(prior_unit['unit_index']):04d}" / "normalized.json"
        if not alignment_path.is_file():
            raise UrduVideoProductionError(
                f"Completed alignment is missing: {prior_unit['unit_id']}"
            )
        alignment = _read_json(alignment_path)
        current_audio = audio_jobs[str(prior_unit["unit_id"])]
        if alignment.get("audio_sha256") != current_audio.get("normalized_sha256"):
            raise UrduVideoProductionError(
                f"Unrelated source audio changed at {prior_unit['unit_id']}"
            )
        verified_alignments += 1

    marker = {
        "version": "quran-urdu-video-audio-content-repair-adoption-v1",
        "run_id": run_id,
        "adopted_at": utc_now(),
        "unit_index": unit_index,
        "unit_id": unit["unit_id"],
        "old_audio_run_sha256": old_audio_run_sha256,
        "new_audio_run_sha256": new_audio_run_sha256,
        "old_implementation_hashes": old_implementation_hashes,
        "new_implementation_hashes": new_implementation_hashes,
        "audio_content_repair_marker": str(repair_marker),
        "audio_content_repair_marker_sha256": file_sha256(repair_marker),
        "new_master_sha256": audio_job["normalized_sha256"],
        "unchanged_completed_alignments_verified": verified_alignments,
        "unrelated_frozen_inputs_unchanged": True,
    }

    work = root / "alignments" / f"{unit_index:04d}"
    archive = work / "pre-content-repair-adoption"
    archive.mkdir(parents=True, exist_ok=True)
    for source in (work / "raw.json", work / "normalized.json", work / "FAILURE.json"):
        if source.is_file() and not (archive / source.name).is_file():
            shutil.copy2(source, archive / source.name)
    source_raw_path = validated_alignment_path.parent / str(
        validated_alignment.get("raw_alignment", {}).get("path", "raw.json")
    )
    adopted_raw_path = work / "content-repair-validation-raw.json"
    if not source_raw_path.is_file():
        raise UrduVideoProductionError("Validated repair alignment has no raw transcript evidence")
    shutil.copy2(source_raw_path, adopted_raw_path)
    adopted_alignment = dict(validated_alignment)
    adopted_alignment["raw_alignment"] = {
        "path": adopted_raw_path.name,
        "sha256": file_sha256(adopted_raw_path),
    }
    adopted_alignment["audio_content_repair_validation"] = {
        "path": str(validated_alignment_path),
        "sha256": file_sha256(validated_alignment_path),
        "repair_marker": str(repair_marker),
        "repair_marker_sha256": file_sha256(repair_marker),
    }
    normalized_path = work / "normalized.json"
    atomic_json(normalized_path, adopted_alignment)
    _validate_alignment(unit, adopted_alignment, master_path)
    video_job["alignment"] = "complete"
    video_job["alignment_sha256"] = file_sha256(normalized_path)
    video_job["mapped_word_coverage"] = adopted_alignment["metrics"][
        "mapped_word_coverage"
    ]
    video_job.pop("alignment_error", None)
    marker["adopted_validated_alignment"] = str(normalized_path)
    marker["adopted_validated_alignment_sha256"] = file_sha256(normalized_path)
    atomic_json(root / f"AUDIO_CONTENT_REPAIR_ADOPTION_{unit_index:04d}.json", marker)
    state["immutable"]["audio_run_sha256"] = new_audio_run_sha256
    state["immutable"]["implementation_hashes"] = new_implementation_hashes
    state.setdefault("source_amendments", []).append(marker)
    state["status"] = "aligning"
    _save(root, state)
    return status(run_id)


def _load_run(run_id: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    root = _run_root(run_id)
    state = _read_json(root / "RUN.json")
    immutable = state.get("immutable")
    if not isinstance(immutable, dict):
        raise UrduVideoProductionError("RUN.json has no immutable input block")
    source_root = Path(str(immutable["source_root"]))
    if _immutable(source_root, str(immutable["alignment_model"])) != immutable:
        raise UrduVideoProductionError("Urdu video inputs changed; refusing resume")
    units, audio_run, publication = _load_sources(source_root)
    return root, state, units, audio_run, publication


def _save(root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json(root / "RUN.json", state)


def _audio_index(audio_run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["unit_id"]): row for row in audio_run["jobs"]}


def _publication_index(publication: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["ref"]): row for row in publication["ayahs"]}


def _spans(unit: dict[str, Any]) -> list[dict[str, Any]]:
    speech_lines = str(unit["speech_text"]).splitlines()
    display_lines = str(unit["text"]).splitlines()
    refs = [str(ref) for ref in unit["refs"]]
    transcript = str(unit["speech_text"])
    if len(speech_lines) != len(refs) or len(display_lines) != len(refs):
        raise UrduVideoProductionError("Spoken/display line counts differ from ayah references")
    cursor = 0
    spans: list[dict[str, Any]] = []
    for ref, speech, display in zip(refs, speech_lines, display_lines):
        start = transcript.find(speech, cursor)
        if start < 0:
            raise UrduVideoProductionError(f"Cannot locate spoken ayah {ref} in unit text")
        end = start + len(speech)
        surah, ayah = (int(value) for value in ref.split(":"))
        spans.append(
            {
                "kind": "ayah",
                "ref": ref,
                "surah": surah,
                "ayah": ayah,
                "start_char": start,
                "end_char": end,
                "text": display,
            }
        )
        cursor = end
    return spans


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalized_urdu_word(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(URDU_LETTER_FORMS)
    return "".join(
        character
        for character in normalized
        if character.isalnum() and unicodedata.category(character) != "Mn"
    ).casefold()


def _transcript_words(transcript: str) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for match in re.finditer(r"\w+", transcript, flags=re.UNICODE):
        normalized = _normalized_urdu_word(match.group(0))
        if normalized:
            words.append(
                {
                    "text": match.group(0),
                    "normalized": normalized,
                    "start_char": match.start(),
                    "end_char": match.end(),
                }
            )
    return words


def _observed_words(raw: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in raw.get("segments", []):
        if not isinstance(segment, dict):
            continue
        for item in segment.get("words", []):
            if not isinstance(item, dict):
                continue
            text = item.get("word")
            start = item.get("start")
            end = item.get("end")
            if not isinstance(text, str) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            normalized = _normalized_urdu_word(text)
            if normalized and float(end) >= float(start):
                words.append(
                    {
                        "text": text.strip(),
                        "normalized": normalized,
                        "start": float(start),
                        "end": float(end),
                        "probability": item.get("probability"),
                    }
                )
    if not words:
        raise AlignmentError("Whisper output has no timestamped Urdu words")
    return words


def _match_word_times(
    expected: list[dict[str, Any]], observed: list[dict[str, Any]]
) -> tuple[dict[int, tuple[float, float]], int]:
    matcher = SequenceMatcher(
        None,
        [row["normalized"] for row in expected],
        [row["normalized"] for row in observed],
        autojunk=False,
    )
    mapping: dict[int, tuple[float, float]] = {}
    exact = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for expected_index, observed_index in zip(range(i1, i2), range(j1, j2)):
                mapping[expected_index] = (
                    float(observed[observed_index]["start"]),
                    float(observed[observed_index]["end"]),
                )
                exact += 1
        elif tag == "replace" and i2 - i1 == j2 - j1:
            for expected_index, observed_index in zip(range(i1, i2), range(j1, j2)):
                ratio = SequenceMatcher(
                    None,
                    str(expected[expected_index]["normalized"]),
                    str(observed[observed_index]["normalized"]),
                    autojunk=False,
                ).ratio()
                if ratio >= 0.60:
                    mapping[expected_index] = (
                        float(observed[observed_index]["start"]),
                        float(observed[observed_index]["end"]),
                    )
    if not mapping:
        raise AlignmentError("Whisper transcript has no usable overlap with Urdu narration")
    return mapping, exact


def _interpolate_word_times(
    expected: list[dict[str, Any]], mapping: dict[int, tuple[float, float]]
) -> list[tuple[float, float]]:
    result: list[tuple[float, float] | None] = [mapping.get(index) for index in range(len(expected))]
    known = sorted(mapping)
    if not known or known[0] != 0 or known[-1] != len(expected) - 1:
        raise AlignmentError("Unsupported Urdu boundary words require alignment review")
    for left, right in zip(known, known[1:]):
        missing = right - left - 1
        gap = mapping[right][0] - mapping[left][1]
        if missing and (
            missing > MAX_INTERPOLATED_WORDS
            or not 0 < gap <= MAX_INTERPOLATED_GAP_SECONDS
        ):
            raise AlignmentError("Urdu alignment gap exceeds bounded interpolation")
        for offset in range(missing):
            slot = gap / missing
            window_start = mapping[left][1]
            index = left + offset + 1
            result[index] = (
                window_start + slot * offset,
                window_start + slot * (offset + 1),
            )
    return [row for row in result if row is not None]


def _normalize_urdu_alignment(
    *,
    raw: dict[str, Any],
    transcript: str,
    spans: list[dict[str, Any]],
    audio_path: Path,
    raw_path: Path,
    normalized_path: Path,
    model: str,
) -> dict[str, Any]:
    expected = _transcript_words(transcript)
    observed = _observed_words(raw)
    mapping, exact = _match_word_times(expected, observed)
    timings = _interpolate_word_times(expected, mapping)
    words = [
        {
            "text": row["text"],
            "start_char": int(row["start_char"]),
            "end_char": int(row["end_char"]),
            "start": round(float(timings[index][0]), 3),
            "end": round(float(timings[index][1]), 3),
            "timing_source": "mapped_word" if index in mapping else "interpolated_word",
        }
        for index, row in enumerate(expected)
    ]
    aligned_spans: list[dict[str, Any]] = []
    for span in spans:
        indices = [
            index
            for index, word in enumerate(expected)
            if int(word["start_char"]) < int(span["end_char"])
            and int(word["end_char"]) > int(span["start_char"])
        ]
        if not indices:
            raise AlignmentError(f"Urdu ayah span has no words: {span['ref']}")
        aligned_spans.append(
            dict(span)
            | {
                "start": round(float(timings[indices[0]][0]), 3),
                "end": round(float(timings[indices[-1]][1]), 3),
                "timing_source": (
                    "exact_words" if all(index in mapping for index in indices) else "interpolated_words"
                ),
            }
        )
    payload = {
        "version": "quran-urdu-video-alignment-v1",
        "engine": f"mlx-whisper:{model}",
        "audio_sha256": file_sha256(audio_path),
        "transcript_sha256": text_sha256(transcript),
        "raw_alignment": {"path": raw_path.name, "sha256": file_sha256(raw_path)},
        "metrics": {
            "expected_words": len(expected),
            "observed_words": len(observed),
            "exact_word_coverage": round(exact / len(expected), 6),
            "mapped_word_coverage": round(len(mapping) / len(expected), 6),
        },
        "words": words,
        "spans": aligned_spans,
    }
    atomic_json(normalized_path, payload)
    return payload


def _audio_duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise UrduVideoProductionError("Audio has no finite positive duration")
    return duration


def _validate_alignment(unit: dict[str, Any], payload: dict[str, Any], audio_path: Path) -> None:
    if not str(payload.get("engine", "")).startswith("mlx-whisper"):
        raise UrduVideoProductionError(f"Wrong alignment engine for {unit['unit_id']}")
    if payload.get("audio_sha256") != file_sha256(audio_path):
        raise UrduVideoProductionError(f"Alignment audio drift at {unit['unit_id']}")
    if payload.get("transcript_sha256") != unit.get("speech_text_sha256"):
        raise UrduVideoProductionError(f"Alignment transcript drift at {unit['unit_id']}")
    expected_refs = [str(ref) for ref in unit["refs"]]
    spans = payload.get("spans")
    actual_refs = [str(row.get("ref")) for row in spans or [] if row.get("kind") == "ayah"]
    if actual_refs != expected_refs or len(spans or []) != len(expected_refs):
        raise UrduVideoProductionError(f"Alignment ayah coverage failed at {unit['unit_id']}")
    expected_words = _transcript_words(unit["speech_text"])
    words = payload.get("words", [])
    if len(words) != len(expected_words) or not words:
        raise UrduVideoProductionError("Alignment word inventory differs from transcript")
    mapping = {}
    duration = _audio_duration(audio_path)
    previous_end = 0.0
    for index, (word, expected) in enumerate(zip(words, expected_words)):
        if any(word.get(key) != expected.get(key) for key in ("text", "start_char", "end_char")):
            raise UrduVideoProductionError("Alignment word offsets or text changed")
        start, end = float(word["start"]), float(word["end"])
        if not all(math.isfinite(value) for value in (start, end)) or (
            start < previous_end - 0.002 or end <= start or end > duration + 0.002
        ):
            raise UrduVideoProductionError("Alignment word timing is outside the audio contract")
        previous_end = end
        if word.get("timing_source") in {"exact_word", "mapped_word"}:
            mapping[index] = (start, end)
        elif word.get("timing_source") != "interpolated_word":
            raise UrduVideoProductionError("Unknown word timing evidence")
    timings = _interpolate_word_times(expected_words, mapping)
    for word, (start, end) in zip(words, timings):
        if abs(word["start"] - start) > 0.002 or abs(word["end"] - end) > 0.002:
            raise UrduVideoProductionError("Interpolated words differ from bounded evidence")
    coverage = len(mapping) / len(words)
    if coverage < MIN_MAPPED_WORD_COVERAGE:
        raise UrduVideoProductionError(
            f"Alignment word coverage {coverage:.3f} is below {MIN_MAPPED_WORD_COVERAGE:.2f} "
            f"at {unit['unit_id']}"
        )
    prior = -1.0
    for span, canonical in zip(spans, _spans(unit)):
        if any(span.get(key) != value for key, value in canonical.items()):
            raise UrduVideoProductionError("Alignment ayah text or offsets changed")
        indices = [index for index, word in enumerate(words)
                   if word["start_char"] < span["end_char"]
                   and word["end_char"] > span["start_char"]]
        if not indices or sum(index in mapping for index in indices) / len(indices) < MIN_AYAH_WORD_COVERAGE:
            raise UrduVideoProductionError(f"Insufficient measured words in ayah {span.get('ref')}")
        start = float(span.get("start", -1))
        end = float(span.get("end", -1))
        if not all(math.isfinite(value) for value in (start, end)) or (
            start < 0 or end <= start or start < prior or end > duration + 0.002
            or abs(start - words[indices[0]]["start"]) > 0.002
            or abs(end - words[indices[-1]]["end"]) > 0.002
        ):
            raise UrduVideoProductionError(f"Non-monotonic alignment at {unit['unit_id']}")
        prior = start


def align(
    *,
    run_id: str = DEFAULT_RUN_ID,
    limit: int | None = None,
    unit_index: int | None = None,
) -> dict[str, Any]:
    root, state, units, audio_run, _ = _load_run(run_id)
    audio_jobs = _audio_index(audio_run)
    model = str(state["immutable"]["alignment_model"])
    completed_this_call = 0
    state["status"] = "aligning"
    _save(root, state)
    for unit, job_state in zip(units, state["jobs"]):
        index = int(unit["unit_index"])
        if unit_index is not None and index != unit_index:
            continue
        work = root / "alignments" / f"{index:04d}"
        raw_path = work / "raw.json"
        normalized_path = work / "normalized.json"
        audio_path = Path(str(audio_jobs[str(unit["unit_id"])]["normalized_path"]))
        if normalized_path.is_file():
            payload = _read_json(normalized_path)
            _validate_alignment(unit, payload, audio_path)
            job_state["alignment"] = "complete"
            job_state["alignment_sha256"] = file_sha256(normalized_path)
            continue
        if limit is not None and completed_this_call >= limit:
            break
        work.mkdir(parents=True, exist_ok=True)
        job_state["alignment"] = "transcribing"
        job_state["attempts"] = int(job_state.get("attempts", 0)) + 1
        _save(root, state)
        try:
            if raw_path.is_file():
                raw = _read_json(raw_path)
            else:
                raw = _jsonable(
                    mlx_whisper.transcribe(
                        str(audio_path),
                        path_or_hf_repo=model,
                        language="ur",
                        task="transcribe",
                        word_timestamps=True,
                        verbose=False,
                        condition_on_previous_text=False,
                        initial_prompt=str(unit["speech_text"]),
                        temperature=0.0,
                    )
                )
                atomic_json(raw_path, raw)
            payload = _normalize_urdu_alignment(
                raw=raw,
                transcript=str(unit["speech_text"]),
                spans=_spans(unit),
                audio_path=audio_path,
                raw_path=raw_path,
                normalized_path=normalized_path,
                model=model,
            )
            _validate_alignment(unit, payload, audio_path)
            job_state["alignment"] = "complete"
            job_state["alignment_sha256"] = file_sha256(normalized_path)
            job_state["mapped_word_coverage"] = payload["metrics"]["mapped_word_coverage"]
            job_state.pop("alignment_error", None)
            completed_this_call += 1
            _save(root, state)
            print(f"alignment {index:04d}/0343 complete", flush=True)
        except Exception as exc:
            job_state["alignment"] = "failed"
            job_state["alignment_error"] = str(exc)
            atomic_json(work / "FAILURE.json", {"at": utc_now(), "error": str(exc)})
            _save(root, state)
            raise UrduVideoProductionError(
                f"Alignment failed at unit {index:04d}: {exc}"
            ) from exc
    if all(row.get("alignment") == "complete" for row in state["jobs"]):
        state["status"] = "aligned"
    _save(root, state)
    return status(run_id)


def _alignment_recovery_prompt(transcript: str) -> str:
    """Keep Whisper's retry prompt local enough to avoid repetition loops."""
    return " ".join(transcript.split())[:220]


def repair_alignment(
    *,
    run_id: str = DEFAULT_RUN_ID,
    unit_index: int,
) -> dict[str, Any]:
    """Retry one failed local alignment while preserving its original artifacts."""
    root, state, units, audio_run, _ = _load_run(run_id)
    if not 1 <= unit_index <= len(units):
        raise UrduVideoProductionError(f"Invalid unit index: {unit_index}")

    unit = units[unit_index - 1]
    job_state = state["jobs"][unit_index - 1]
    if int(unit["unit_index"]) != unit_index or int(job_state["unit_index"]) != unit_index:
        raise UrduVideoProductionError(f"Unit index drift at {unit_index}")
    if job_state.get("alignment") == "complete":
        return status(run_id)
    if job_state.get("alignment") != "failed":
        raise UrduVideoProductionError(
            f"Alignment recovery requires a failed unit, got {job_state.get('alignment')}"
        )

    work = root / "alignments" / f"{unit_index:04d}"
    raw_path = work / "raw.json"
    normalized_path = work / "normalized.json"
    failure_path = work / "FAILURE.json"
    if not raw_path.is_file() or not normalized_path.is_file() or not failure_path.is_file():
        raise UrduVideoProductionError(
            f"Failed alignment artifacts are incomplete at unit {unit_index:04d}"
        )

    prior_attempt = max(1, int(job_state.get("attempts", 1)))
    archive = work / f"failed-attempt-{prior_attempt:04d}"
    archive.mkdir(parents=True, exist_ok=True)
    for source in (raw_path, normalized_path, failure_path):
        archived = archive / source.name
        if not archived.is_file():
            shutil.copy2(source, archived)

    attempt = prior_attempt + 1
    recovery = work / f"recovery-attempt-{attempt:04d}"
    recovery.mkdir(parents=True, exist_ok=True)
    retry_raw_path = recovery / "raw.json"
    retry_normalized_path = recovery / "normalized.json"
    audio_path = Path(
        str(_audio_index(audio_run)[str(unit["unit_id"])]["normalized_path"])
    )
    model = str(state["immutable"]["alignment_model"])

    job_state["alignment"] = "recovering"
    job_state["attempts"] = attempt
    _save(root, state)
    try:
        if retry_raw_path.is_file():
            raw = _read_json(retry_raw_path)
        else:
            raw = _jsonable(
                mlx_whisper.transcribe(
                    str(audio_path),
                    path_or_hf_repo=model,
                    language="ur",
                    task="transcribe",
                    word_timestamps=True,
                    verbose=False,
                    condition_on_previous_text=False,
                    initial_prompt=_alignment_recovery_prompt(str(unit["speech_text"])),
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    hallucination_silence_threshold=2.0,
                )
            )
            atomic_json(retry_raw_path, raw)
        payload = _normalize_urdu_alignment(
            raw=raw,
            transcript=str(unit["speech_text"]),
            spans=_spans(unit),
            audio_path=audio_path,
            raw_path=retry_raw_path,
            normalized_path=retry_normalized_path,
            model=model,
        )
        _validate_alignment(unit, payload, audio_path)
        atomic_json(raw_path, raw)
        atomic_json(normalized_path, payload)
        job_state["alignment"] = "complete"
        job_state["alignment_sha256"] = file_sha256(normalized_path)
        job_state["mapped_word_coverage"] = payload["metrics"]["mapped_word_coverage"]
        job_state.pop("alignment_error", None)
        atomic_json(
            recovery / "RECOVERY_COMPLETE.json",
            {
                "at": utc_now(),
                "unit_id": unit["unit_id"],
                "unit_index": unit_index,
                "mapped_word_coverage": payload["metrics"]["mapped_word_coverage"],
                "raw_sha256": file_sha256(raw_path),
                "normalized_sha256": file_sha256(normalized_path),
                "preserved_failure": str(archive),
            },
        )
        if all(row.get("alignment") == "complete" for row in state["jobs"]):
            state["status"] = "aligned"
        else:
            state["status"] = "aligning"
        _save(root, state)
        return status(run_id)
    except Exception as exc:
        job_state["alignment"] = "failed"
        job_state["alignment_error"] = str(exc)
        atomic_json(recovery / "RECOVERY_FAILURE.json", {"at": utc_now(), "error": str(exc)})
        _save(root, state)
        raise UrduVideoProductionError(
            f"Alignment recovery failed at unit {unit_index:04d}: {exc}"
        ) from exc


def _valid_render(
    *,
    qa_path: Path,
    video_path: Path,
    alignment_path: Path,
    audio_path: Path,
) -> bool:
    if not qa_path.is_file() or not video_path.is_file():
        return False
    qa = _read_json(qa_path)
    return (
        qa.get("render", {}).get("sha256") == file_sha256(video_path)
        and qa.get("render", {}).get("source_audio_sha256") == file_sha256(audio_path)
        and qa.get("alignment", {}).get("sha256") == file_sha256(alignment_path)
        and qa.get("checks", {}).get("decode_passed") is True
        and qa.get("srt_qa", {}).get("timeline_identity") is True
        and qa.get("encoded_timeline", {}).get("passed") is True
        and qa.get("implementation_hashes") == _implementation_hashes()
    )


def render(
    *,
    run_id: str = DEFAULT_RUN_ID,
    limit: int | None = None,
    unit_index: int | None = None,
) -> dict[str, Any]:
    root, state, units, audio_run, publication = _load_run(run_id)
    audio_jobs = _audio_index(audio_run)
    metadata = _publication_index(publication)
    rendered_this_call = 0
    state["status"] = "rendering"
    _save(root, state)
    for unit, job_state in zip(units, state["jobs"]):
        index = int(unit["unit_index"])
        if unit_index is not None and index != unit_index:
            continue
        if job_state.get("alignment") != "complete":
            if unit_index is not None:
                raise UrduVideoProductionError(f"Unit {index:04d} is not aligned")
            continue
        alignment_path = root / "alignments" / f"{index:04d}" / "normalized.json"
        audio_path = Path(str(audio_jobs[str(unit["unit_id"])]["normalized_path"]))
        video_path = root / "segments" / f"{index:04d}-{unit['unit_id']}.mp4"
        work = root / "segment-work" / f"{index:04d}"
        qa_path = work / "QA.json"
        if _valid_render(
            qa_path=qa_path,
            video_path=video_path,
            alignment_path=alignment_path,
            audio_path=audio_path,
        ):
            job_state["render"] = "complete"
            job_state["render_sha256"] = file_sha256(video_path)
            continue
        if limit is not None and rendered_this_call >= limit:
            break
        alignment = _read_json(alignment_path)
        _validate_alignment(unit, alignment, audio_path)
        display = build_urdu_display_events(alignment)
        first = metadata[str(unit["refs"][0])]
        work.mkdir(parents=True, exist_ok=True)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(work / "DISPLAY_EVENTS.json", display)
        job_state["render"] = "rendering"
        _save(root, state)
        try:
            frames = render_urdu_frames(
                display=display,
                output_dir=work / "frames-2x",
                surah_name_ar=str(first["surah_name_ar"]),
                surah_name_en=str(first["surah_name_en"]),
                para_number=int(unit["juz"]),
            )
            mobile = write_mobile_review(
                frames=frames,
                output_dir=work / "mobile-360",
                max_samples=3,
            )
            srt_path = write_srt(display, work / "captions.ur.srt")
            srt_qa = validate_srt_identity(display, srt_path)
            segment_metadata = {
                "version": "quran-urdu-video-segment-metadata-v1",
                "unit_index": index,
                "unit_id": unit["unit_id"],
                "surah": int(unit["surah"]),
                "para": int(unit["juz"]),
                "start_ref": unit["refs"][0],
                "end_ref": unit["refs"][-1],
            }
            atomic_json(work / "SEGMENT_METADATA.json", segment_metadata)
            video_path.unlink(missing_ok=True)
            render_result = render_video(
                display=display,
                frames=frames,
                audio_path=audio_path,
                output_path=video_path,
            )
            minimum_content_duration = (
                max(float(event["source_end"]) for event in display["events"])
                - float(display["selection"]["clip_start"])
            )
            checks = validate_video(
                video_path,
                float(display["selection"]["duration"]),
                minimum_content_duration=minimum_content_duration,
            )
            encoded = validate_encoded_timeline(
                video_path=video_path,
                display=display,
                frames=frames,
                max_samples=6,
            )
            qa = {
                "version": "quran-urdu-video-segment-qa-v1",
                "unit_index": index,
                "unit_id": unit["unit_id"],
                "alignment": {
                    "path": str(alignment_path),
                    "sha256": file_sha256(alignment_path),
                    "engine": alignment.get("engine"),
                    "metrics": alignment.get("metrics"),
                },
                "selection": display["selection"],
                "display_events": len(display["events"]),
                "frames": len(frames),
                "mobile_review": mobile,
                "srt": str(srt_path),
                "srt_qa": srt_qa,
                "metadata": segment_metadata,
                "render": render_result,
                "checks": checks,
                "encoded_timeline": encoded,
                "implementation_hashes": _implementation_hashes(),
            }
            atomic_json(qa_path, qa)
            job_state["render"] = "complete"
            job_state["render_sha256"] = file_sha256(video_path)
            job_state.pop("render_error", None)
            rendered_this_call += 1
            _save(root, state)
            print(f"render {index:04d}/0343 complete", flush=True)
        except Exception as exc:
            job_state["render"] = "failed"
            job_state["render_error"] = str(exc)
            atomic_json(work / "FAILURE.json", {"at": utc_now(), "error": str(exc)})
            _save(root, state)
            raise UrduVideoProductionError(f"Render failed at unit {index:04d}: {exc}") from exc
    if all(row.get("render") == "complete" for row in state["jobs"]):
        state["status"] = "rendered"
    _save(root, state)
    return status(run_id)


def _slug(value: str) -> str:
    value = value.casefold().replace("'", "-")
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def _ffconcat_line(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


def _chapter_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _chapters(
    *,
    kind: str,
    units: list[dict[str, Any]],
    displays: list[dict[str, Any]],
    durations: list[float],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    ayah_points: list[tuple[float, str]] = []
    offset = 0.0
    previous_surah: int | None = None
    for unit, display, duration in zip(units, displays, durations):
        start_ref = str(unit["refs"][0])
        surah = int(unit["surah"])
        if previous_surah != surah or not points:
            name = str(metadata[start_ref]["surah_name_en"])
            points.append(
                {
                    "seconds": round(offset, 3),
                    "timestamp": _chapter_time(offset),
                    "label": f"Surah {surah}: {name}",
                    "ref": start_ref,
                }
            )
        previous_surah = surah
        seen: set[str] = set()
        for event in display.get("events", []):
            ref = event.get("active_ref")
            if isinstance(ref, str) and ref not in seen:
                ayah_points.append((offset + float(event["start"]), ref))
                seen.add(ref)
        offset += duration
    if kind == "surah" and offset >= 600:
        target = 600.0
        used = {str(row["ref"]) for row in points}
        while target < offset:
            candidates = [row for row in ayah_points if row[0] >= target and row[1] not in used]
            if not candidates:
                break
            seconds, ref = candidates[0]
            points.append(
                {
                    "seconds": round(seconds, 3),
                    "timestamp": _chapter_time(seconds),
                    "label": f"Ayah {ref}",
                    "ref": ref,
                }
            )
            used.add(ref)
            target += 600
    points.sort(key=lambda row: float(row["seconds"]))
    if points:
        points[0]["seconds"] = 0.0
        points[0]["timestamp"] = "0:00"
    return points if len(points) >= 3 else []


def _catalog_metadata(
    *,
    kind: str,
    number: int,
    units: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    first_ref = str(units[0]["refs"][0])
    last_ref = str(units[-1]["refs"][-1])
    first = metadata[first_ref]
    if kind == "para":
        title = f"Para {number}/30 | Quran Urdu Translation | پارہ {number}"
        playlist = "Quran Urdu Translation - 30 Paras"
        description = (
            f"Para {number} of 30, {first_ref}–{last_ref}.\n\n"
            "Urdu Quran translation with synchronized on-screen text."
        )
        thumbnail_label = f"PARA {number}/30"
    else:
        name = str(first["surah_name_en"])
        name_ar = str(first["surah_name_ar"])
        title = f"{name} ({number}/114) | Quran Urdu Translation | سورۃ {name_ar}"
        playlist = "Quran Urdu Translation - 114 Surahs"
        description = (
            f"Surah {name}, {first_ref}–{last_ref}.\n\n"
            "Urdu Quran translation with synchronized on-screen text."
        )
        thumbnail_label = f"{number}/114  {name}"
    if chapters:
        description += "\n\n" + "\n".join(
            f"{row['timestamp']} {row['label']}" for row in chapters
        )
    return {
        "version": "quran-urdu-youtube-metadata-v1",
        "title": title,
        "description": description + "\n",
        "thumbnail_label": thumbnail_label,
        "playlist": playlist,
        "language": "Urdu",
        "caption_language": "Urdu",
        "category": "Education",
        "made_for_kids": False,
        "visibility": "public",
        "chapters": chapters,
    }


def _assemble_catalog(
    *,
    root: Path,
    kind: str,
    number: int,
    units: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    decode_check: bool,
) -> dict[str, Any]:
    indices = [int(unit["unit_index"]) for unit in units]
    inputs = [root / "segments" / f"{index:04d}-{unit['unit_id']}.mp4" for index, unit in zip(indices, units)]
    for path in inputs:
        if not path.is_file():
            raise UrduVideoProductionError(f"Missing rendered segment: {path}")
    first = metadata[str(units[0]["refs"][0])]
    if kind == "para":
        output = root / "paras" / f"para-{number:02d}-of-30.mp4"
    else:
        output = root / "surahs" / f"surah-{number:03d}-{_slug(str(first['surah_name_en']))}-urdu.mp4"
    qa_path = output.with_suffix(".qa.json")
    contract = catalog_contract(
        inputs, [root / "segment-work" / f"{index:04d}" / "DISPLAY_EVENTS.json" for index in indices],
        {"kind": kind, "number": number, "units": units, "metadata": metadata,
         "implementation_hashes": _implementation_hashes()},
    )
    def restore_sidecars(captions_path: Path, metadata_path: Path) -> None:
        durations = [float(_probe_video(path)["duration_seconds"]) for path in inputs]
        displays = [_read_json(root / "segment-work" / f"{index:04d}" / "DISPLAY_EVENTS.json")
                    for index in indices]
        write_catalog_srt(displays=displays, segment_durations=durations, output_path=captions_path)
        chapters = _chapters(
            kind=kind, units=units, displays=displays, durations=durations, metadata=metadata
        )
        atomic_json(metadata_path, _catalog_metadata(
            kind=kind, number=number, units=units, chapters=chapters, metadata=metadata
        ))

    prior = validate_cached_catalog(output, qa_path, contract, "ur", restore_sidecars)
    if prior is not None:
        return prior

    output.parent.mkdir(parents=True, exist_ok=True)
    concat = output.with_suffix(".concat.txt")
    temp = output.with_name(f".{output.name}.tmp.mp4")
    atomic_text(concat, "\n".join(_ffconcat_line(path) for path in inputs) + "\n")
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat),
                "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
                "-movflags", "+faststart", str(temp),
            ],
            check=True,
        )
        os.replace(temp, output)
    finally:
        concat.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)

    probes = [_probe_video(path) for path in inputs]
    durations = [float(row["duration_seconds"]) for row in probes]
    displays = [
        _read_json(root / "segment-work" / f"{index:04d}" / "DISPLAY_EVENTS.json")
        for index in indices
    ]
    captions = output.with_suffix(".ur.srt")
    write_catalog_srt(displays=displays, segment_durations=durations, output_path=captions)
    chapters = _chapters(
        kind=kind,
        units=units,
        displays=displays,
        durations=durations,
        metadata=metadata,
    )
    metadata_payload = _catalog_metadata(
        kind=kind,
        number=number,
        units=units,
        chapters=chapters,
        metadata=metadata,
    )
    metadata_path = output.with_suffix(".metadata.json")
    atomic_json(metadata_path, metadata_payload)
    probe = _probe_video(output)
    expected = sum(durations)
    tolerance = max(1.0, expected * 0.001)
    if abs(float(probe["duration_seconds"]) - expected) > tolerance:
        raise UrduVideoProductionError(
            f"Catalog duration mismatch at {output}: {probe['duration_seconds']} vs {expected}"
        )
    if (
        (probe["width"], probe["height"]) != (1920, 1080)
        or probe["pixel_format"] != "yuv420p"
        or probe["video_codec"] != "h264"
        or probe["audio_codec"] != "aac"
        or probe["audio_sample_rate"] != 44100
        or probe["audio_channels"] != 1
    ):
        raise UrduVideoProductionError(f"Catalog media contract failed: {output}")
    if decode_check:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
            check=True,
        )
    qa = {
        "version": "quran-urdu-video-catalog-qa-v1",
        "input_contract": contract,
        "kind": kind,
        "number": number,
        "path": str(output),
        "sha256": file_sha256(output),
        "bytes": output.stat().st_size,
        "source_units": indices,
        "expected_duration_seconds": round(expected, 3),
        "probe": probe,
        "captions": str(captions),
        "captions_sha256": file_sha256(captions),
        "metadata": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "chapters": len(chapters),
        "stream_copy": True,
        "decode_passed": bool(decode_check),
    }
    atomic_json(qa_path, qa)
    return qa


def _write_checksums(root: Path) -> Path:
    paths = sorted(
        path
        for folder in (root / "paras", root / "surahs")
        for path in folder.glob("*")
        if path.is_file()
    )
    checksum = root / "SHA256SUMS.txt"
    atomic_text(
        checksum,
        "\n".join(f"{file_sha256(path)}  {path.relative_to(root)}" for path in paths) + "\n",
    )
    return checksum


def assemble(*, run_id: str = DEFAULT_RUN_ID, decode_check: bool = True) -> dict[str, Any]:
    root, state, units, _, publication = _load_run(run_id)
    if any(row.get("render") != "complete" for row in state["jobs"]):
        raise UrduVideoProductionError("All 343 rendered segments must pass before assembly")
    metadata = _publication_index(publication)
    by_para: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_surah: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_para[int(unit["juz"])].append(unit)
        by_surah[int(unit["surah"])].append(unit)
    if sorted(by_para) != list(range(1, 31)) or sorted(by_surah) != list(range(1, 115)):
        raise UrduVideoProductionError("Catalog groups do not cover all 30 Paras and 114 Surahs")
    state["status"] = "assembling"
    _save(root, state)
    results = {"paras": [], "surahs": []}
    for number in range(1, 31):
        results["paras"].append(
            _assemble_catalog(
                root=root,
                kind="para",
                number=number,
                units=by_para[number],
                metadata=metadata,
                decode_check=decode_check,
            )
        )
        print(f"assemble para {number:02d}/30 complete", flush=True)
    state["catalogs"]["paras"] = "complete"
    _save(root, state)
    for number in range(1, 115):
        results["surahs"].append(
            _assemble_catalog(
                root=root,
                kind="surah",
                number=number,
                units=by_surah[number],
                metadata=metadata,
                decode_check=decode_check,
            )
        )
        print(f"assemble surah {number:03d}/114 complete", flush=True)
    state["catalogs"]["surahs"] = "complete"
    state["status"] = "complete"
    checksum = _write_checksums(root)
    marker = {
        "version": "quran-urdu-video-production-complete-v1",
        "run_id": run_id,
        "completed_at": utc_now(),
        "segments": 343,
        "para_videos": 30,
        "surah_videos": 114,
        "checksums": str(checksum),
        "checksums_sha256": file_sha256(checksum),
        "decode_checked": bool(decode_check),
        "provider_cost_usd": 0.0,
    }
    atomic_json(root / "PRODUCTION_COMPLETE.json", marker)
    _save(root, state)
    return marker


def status(run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    root = _run_root(run_id)
    state = _read_json(root / "RUN.json")

    def count(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in state.get("jobs", []):
            value = str(row.get(field, "pending"))
            result[value] = result.get(value, 0) + 1
        return result

    return {
        "run_id": run_id,
        "status": state.get("status"),
        "alignment": count("alignment"),
        "render": count("render"),
        "catalogs": state.get("catalogs"),
        "provider_cost_usd": state.get("provider_cost_usd", 0.0),
        "production_complete": (root / "PRODUCTION_COMPLETE.json").is_file(),
    }


def pipeline(*, run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    align(run_id=run_id)
    render(run_id=run_id)
    return assemble(run_id=run_id, decode_check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "migrate",
            "adopt-audio-content-repair",
            "align",
            "repair-alignment",
            "render",
            "assemble",
            "status",
            "pipeline",
        ),
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--unit-index", type=int)
    parser.add_argument("--no-decode-check", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare(
            run_id=args.run_id,
            source_root=args.source_root,
            model=args.model,
        )
    elif args.command == "migrate":
        result = migrate_implementation(args.run_id)
    elif args.command == "adopt-audio-content-repair":
        if args.unit_index is None:
            raise UrduVideoProductionError(
                "adopt-audio-content-repair requires --unit-index"
            )
        result = adopt_audio_content_repair(
            run_id=args.run_id,
            unit_index=args.unit_index,
        )
    elif args.command == "align":
        result = align(run_id=args.run_id, limit=args.limit, unit_index=args.unit_index)
    elif args.command == "repair-alignment":
        if args.unit_index is None:
            raise UrduVideoProductionError("repair-alignment requires --unit-index")
        result = repair_alignment(run_id=args.run_id, unit_index=args.unit_index)
    elif args.command == "render":
        result = render(run_id=args.run_id, limit=args.limit, unit_index=args.unit_index)
    elif args.command == "assemble":
        result = assemble(run_id=args.run_id, decode_check=not args.no_decode_check)
    elif args.command == "pipeline":
        result = pipeline(run_id=args.run_id)
    else:
        result = status(args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
