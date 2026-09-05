"""Integrity-first inputs for the Quran video publication pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import file_sha256, text_sha256
from .production_packets import atomic_json


class VideoPipelineError(RuntimeError):
    """Raised when video inputs cannot be tied to the frozen audiobook release."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoPipelineError(f"Cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VideoPipelineError(f"Expected JSON object: {path}")
    return payload


def _ref_key(ref: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{1,3}):(\d{1,3})", ref)
    if not match:
        raise VideoPipelineError(f"Malformed Quran reference: {ref}")
    return int(match.group(1)), int(match.group(2))


def _release_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = _read_object(path)
    raw_rows = payload.get("ayahs")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise VideoPipelineError("Listening edition has no ayahs")
    rows: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    prior: tuple[int, int] | None = None
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise VideoPipelineError(f"Malformed ayah at position {index + 1}")
        ref = raw.get("ref")
        translation = raw.get("translation")
        if not isinstance(ref, str) or ref in positions:
            raise VideoPipelineError(f"Malformed or duplicate ayah reference: {ref}")
        if not isinstance(translation, str) or not translation.strip():
            raise VideoPipelineError(f"Empty translation at {ref}")
        key = _ref_key(ref)
        if prior is not None and key <= prior:
            raise VideoPipelineError("Listening edition is not in canonical ayah order")
        if raw.get("surah") != key[0] or raw.get("ayah") != key[1]:
            raise VideoPipelineError(f"Reference fields disagree at {ref}")
        positions[ref] = len(rows)
        rows.append(raw)
        prior = key
    return rows, positions


def _span_payload(
    rows: list[dict[str, Any]],
    start: int,
    end: int,
) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    for row_index, row in enumerate(rows[start : end + 1]):
        if row_index:
            parts.append("\n")
            cursor += 1
        if int(row["ayah"]) == 1:
            title = f"Surah {row['surah']}. {str(row['surah_name_en']).strip()}."
            parts.append(title)
            spans.append(
                {
                    "kind": "surah_announcement",
                    "surah": int(row["surah"]),
                    "start_char": cursor,
                    "end_char": cursor + len(title),
                    "text": title,
                }
            )
            cursor += len(title)
            parts.append("\n\n")
            cursor += 2
        translation = str(row["translation"]).strip()
        parts.append(translation)
        spans.append(
            {
                "kind": "ayah",
                "ref": str(row["ref"]),
                "surah": int(row["surah"]),
                "ayah": int(row["ayah"]),
                "start_char": cursor,
                "end_char": cursor + len(translation),
                "text": translation,
            }
        )
        cursor += len(translation)
    return "".join(parts), spans


def build_narration_manifest(
    *,
    run_path: Path,
    inputs_dir: Path,
    listening_edition_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate and package the exact 313 narration scripts used for production."""

    state = _read_object(run_path)
    jobs = state.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise VideoPipelineError("Audio run has no jobs")
    rows, positions = _release_rows(listening_edition_path)

    semantic_material = "\n".join(
        f"{row['ref']}\t{row['translation']}" for row in rows
    )
    semantic_hash = text_sha256(semantic_material)
    if state.get("final_text_sha256") != semantic_hash:
        raise VideoPipelineError("Audio run and listening edition text hashes disagree")

    expected_indices = list(range(1, len(jobs) + 1))
    actual_indices = [int(job.get("chunk_index", 0)) for job in jobs if isinstance(job, dict)]
    if actual_indices != expected_indices:
        raise VideoPipelineError("Audio jobs are not in complete sequential order")

    manifest_jobs: list[dict[str, Any]] = []
    input_hash_material: list[str] = []
    covered_refs: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            raise VideoPipelineError("Audio run contains a malformed job")
        index = int(job["chunk_index"])
        start_ref = str(job["start_ref"])
        end_ref = str(job["end_ref"])
        if start_ref not in positions or end_ref not in positions:
            raise VideoPipelineError(f"Job {index:04d} references are absent from release")
        start = positions[start_ref]
        end = positions[end_ref]
        if end < start:
            raise VideoPipelineError(f"Job {index:04d} has an inverted range")
        expected_text, spans = _span_payload(rows, start, end)
        filename = Path(str(job["input_path"])).name
        input_path = inputs_dir / filename
        if not input_path.exists():
            raise VideoPipelineError(f"Missing narration script: {input_path}")
        actual_text = input_path.read_text(encoding="utf-8")
        actual_hash = text_sha256(actual_text)
        if actual_text != expected_text:
            raise VideoPipelineError(
                f"Narration script does not reproduce release rows: {filename}"
            )
        if actual_hash != job.get("text_sha256"):
            raise VideoPipelineError(f"Narration script hash mismatch: {filename}")
        if len(actual_text) != int(job.get("char_count", -1)):
            raise VideoPipelineError(f"Narration script character count mismatch: {filename}")
        ayah_spans = [span for span in spans if span["kind"] == "ayah"]
        if len(ayah_spans) != int(job.get("ayah_count", -1)):
            raise VideoPipelineError(f"Narration script ayah count mismatch: {filename}")

        refs = [str(row["ref"]) for row in rows[start : end + 1]]
        covered_refs.extend(refs)
        input_hash_material.append(f"inputs/{filename}\t{actual_hash}")
        master_name = Path(str(job.get("master_mp3_path", ""))).name
        manifest_jobs.append(
            {
                "job_id": str(job["job_id"]),
                "chunk_index": index,
                "start_ref": start_ref,
                "end_ref": end_ref,
                "ayah_count": len(ayah_spans),
                "juz_number": int(job["juz_number"]),
                "surah_number": int(job["surah_number"]),
                "surah_name": str(job["surah_name"]),
                "input_path": f"inputs/{filename}",
                "text_sha256": actual_hash,
                "char_count": len(actual_text),
                "master_mp3_path": f"masters-mp3/{master_name}",
                "master_mp3_sha256": job.get("master_mp3_sha256"),
                "duration_seconds": job.get("duration_seconds"),
                "spans": spans,
            }
        )

    release_refs = [str(row["ref"]) for row in rows]
    if covered_refs != release_refs:
        raise VideoPipelineError("Audio jobs do not cover the listening edition exactly once")

    payload: dict[str, Any] = {
        "version": "quran-video-narration-manifest-v1",
        "audio_run_id": state.get("audio_run_id"),
        "audio_run_version": state.get("version"),
        "release_version": state.get("release_version"),
        "final_text_sha256": semantic_hash,
        "audio_input_fingerprint": state.get("input_fingerprint"),
        "source_artifacts": {
            "run_json": {
                "path": "RUN.json",
                "sha256": file_sha256(run_path),
            },
            "listening_edition": {
                "path": listening_edition_path.name,
                "sha256": file_sha256(listening_edition_path),
            },
            "narration_scripts_sha256": text_sha256("\n".join(input_hash_material)),
        },
        "totals": {
            "jobs": len(manifest_jobs),
            "ayahs": len(release_refs),
            "characters": sum(int(job["char_count"]) for job in manifest_jobs),
            "duration_seconds": round(
                sum(float(job["duration_seconds"] or 0.0) for job in manifest_jobs), 3
            ),
        },
        "jobs": manifest_jobs,
    }
    atomic_json(output_path, payload)
    return payload


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "surah"


def build_video_catalog_plan(
    *,
    narration_manifest: dict[str, Any],
    output_path: Path,
    expected_juz_count: int = 30,
    expected_surah_count: int = 114,
) -> dict[str, Any]:
    """Plan one canonical segment render reused by both release catalogs."""

    jobs = narration_manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise VideoPipelineError("Narration manifest has no jobs")
    expected_indices = list(range(1, len(jobs) + 1))
    actual_indices = [int(job.get("chunk_index", 0)) for job in jobs if isinstance(job, dict)]
    if actual_indices != expected_indices:
        raise VideoPipelineError("Catalog jobs are not in canonical chunk order")

    segments: list[dict[str, Any]] = []
    by_juz: dict[int, list[int]] = {}
    by_surah: dict[int, list[int]] = {}
    surah_names: dict[int, str] = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise VideoPipelineError("Narration manifest contains a malformed job")
        index = int(job["chunk_index"])
        juz = int(job["juz_number"])
        surah = int(job["surah_number"])
        span_surahs = {
            int(span["surah"])
            for span in job.get("spans", [])
            if isinstance(span, dict) and span.get("kind") == "ayah"
        }
        if span_surahs != {surah}:
            raise VideoPipelineError(
                f"Chunk {index:04d} crosses a Surah boundary or has inconsistent metadata"
            )
        by_juz.setdefault(juz, []).append(index)
        by_surah.setdefault(surah, []).append(index)
        surah_names[surah] = str(job["surah_name"])
        segments.append(
            {
                "chunk_index": index,
                "job_id": str(job["job_id"]),
                "juz_number": juz,
                "surah_number": surah,
                "start_ref": str(job["start_ref"]),
                "end_ref": str(job["end_ref"]),
                "source_audio_sha256": job.get("master_mp3_sha256"),
                "source_text_sha256": job.get("text_sha256"),
                "output": f"segments/segment-{index:04d}-{job['job_id']}.mp4",
            }
        )

    if expected_juz_count and sorted(by_juz) != list(range(1, expected_juz_count + 1)):
        raise VideoPipelineError("Catalog plan does not cover every expected Juz")
    if expected_surah_count and sorted(by_surah) != list(range(1, expected_surah_count + 1)):
        raise VideoPipelineError("Catalog plan does not cover every expected Surah")

    def require_contiguous(groups: dict[int, list[int]], label: str) -> None:
        for number, indices in groups.items():
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise VideoPipelineError(f"{label} {number} is not contiguous in the master timeline")

    require_contiguous(by_juz, "Juz")
    require_contiguous(by_surah, "Surah")
    if sorted(index for values in by_juz.values() for index in values) != expected_indices:
        raise VideoPipelineError("Juz catalog does not use every segment exactly once")
    if sorted(index for values in by_surah.values() for index in values) != expected_indices:
        raise VideoPipelineError("Surah catalog does not use every segment exactly once")

    payload = {
        "version": "quran-video-catalog-plan-v1",
        "narration_manifest_version": narration_manifest.get("version"),
        "audio_run_id": narration_manifest.get("audio_run_id"),
        "render_strategy": "render-each-canonical-chunk-once",
        "assembly_strategy": "ffmpeg-concat-stream-copy-no-realignment-no-reencoding",
        "totals": {
            "segments": len(segments),
            "juz_videos": len(by_juz),
            "surah_videos": len(by_surah),
        },
        "segments": segments,
        "juz": [
            {
                "number": number,
                "output": f"juz/juz-{number:02d}.mp4",
                "segment_indices": indices,
            }
            for number, indices in sorted(by_juz.items())
        ],
        "surahs": [
            {
                "number": number,
                "name": surah_names[number],
                "output": f"surahs/surah-{number:03d}-{_slug(surah_names[number])}.mp4",
                "segment_indices": indices,
            }
            for number, indices in sorted(by_surah.items())
        ],
        "invariants": {
            "each_segment_rendered_once": True,
            "each_segment_used_once_per_catalog": True,
            "chunks_cross_surah_boundaries": False,
            "catalogs_share_one_timeline": True,
        },
    }
    atomic_json(output_path, payload)
    return payload
