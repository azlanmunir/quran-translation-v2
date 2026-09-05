"""Resumable, integrity-locked Quran video fleet production."""

from __future__ import annotations

from .catalog_cache import catalog_contract, validate_cached_catalog

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import OUTPUT_DIR, PROJECT_ROOT, file_sha256, text_sha256
from .db import utc_now
from .production_packets import atomic_json, atomic_text
from .video_alignment import (
    AlignmentError,
    normalize_forced_alignment,
    request_forced_alignment,
)
from .video_render import (
    MAX_LOUDNESS_DELTA_LUFS,
    MAX_DURATION_DELTA_SECONDS,
    MIN_CONTENT_END_MARGIN_SECONDS,
    TARGET_LOUDNESS_LUFS,
    build_display_events,
    render_frames,
    render_video,
    validate_encoded_timeline,
    validate_srt_identity,
    validate_video,
    write_metadata_kit,
    write_mobile_review,
    write_srt,
)


DEFAULT_VIDEO_RUN_ID = "quran-v2.4.1-youtube-production-v1"
DEFAULT_AUDIO_RUN_ID = "quran-v2.4.1-nathan-v3-production-v1"
DEFAULT_PILOT_ROOT = (
    OUTPUT_DIR / "video" / "pilots" / "quran-v2.4.1-youtube-pilot-v1"
)
DEFAULT_NARRATION_MANIFEST = DEFAULT_PILOT_ROOT / "NARRATION_MANIFEST.json"
DEFAULT_CATALOG_PLAN = DEFAULT_PILOT_ROOT / "CATALOG_PLAN.json"
DEFAULT_LISTENING_EDITION = (
    OUTPUT_DIR
    / "release"
    / "quran-translation-v2.4.1"
    / "quran-listening-edition.json"
)


class VideoProductionError(RuntimeError):
    """Raised when production cannot continue without violating provenance or QA."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoProductionError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VideoProductionError(f"Expected a JSON object: {path}")
    return payload


def _run_root(run_id: str) -> Path:
    return OUTPUT_DIR / "video" / "runs" / run_id


def _audio_root(audio_run_id: str) -> Path:
    return OUTPUT_DIR / "audio" / "runs" / audio_run_id


def _source_hashes() -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "src" / "quran_translate" / "video_alignment.py",
        PROJECT_ROOT / "src" / "quran_translate" / "video_render.py",
        PROJECT_ROOT / "src" / "quran_translate" / "video_production.py",
        PROJECT_ROOT / "src" / "quran_translate" / "catalog_cache.py",
        PROJECT_ROOT / "configs" / "video_release_v1.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise VideoProductionError(f"Missing production source inputs: {missing}")
    return {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in paths}


def _immutable_inputs(
    *,
    narration_manifest: Path,
    catalog_plan: Path,
    listening_edition: Path,
    audio_run_id: str,
) -> dict[str, Any]:
    return {
        "narration_manifest": str(narration_manifest),
        "narration_manifest_sha256": file_sha256(narration_manifest),
        "catalog_plan": str(catalog_plan),
        "catalog_plan_sha256": file_sha256(catalog_plan),
        "listening_edition": str(listening_edition),
        "listening_edition_sha256": file_sha256(listening_edition),
        "audio_run_id": audio_run_id,
        "source_hashes": _source_hashes(),
    }


def _validate_inputs(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    *,
    audio_run_id: str,
) -> None:
    jobs = manifest.get("jobs")
    segments = plan.get("segments")
    if not isinstance(jobs, list) or len(jobs) != 313:
        raise VideoProductionError("Narration manifest must contain exactly 313 jobs")
    if not isinstance(segments, list) or len(segments) != len(jobs):
        raise VideoProductionError("Catalog plan does not match narration jobs")
    if manifest.get("audio_run_id") != audio_run_id or plan.get("audio_run_id") != audio_run_id:
        raise VideoProductionError("Audio run IDs disagree")
    if plan.get("invariants") != {
        "each_segment_rendered_once": True,
        "each_segment_used_once_per_catalog": True,
        "chunks_cross_surah_boundaries": False,
        "catalogs_share_one_timeline": True,
    }:
        raise VideoProductionError("Catalog reuse invariants are absent or changed")

    audio_root = _audio_root(audio_run_id)
    failures: list[str] = []
    for expected, (job, segment) in enumerate(zip(jobs, segments), start=1):
        if int(job.get("chunk_index", 0)) != expected:
            failures.append(f"job-index:{expected}")
            continue
        if int(segment.get("chunk_index", 0)) != expected:
            failures.append(f"segment-index:{expected}")
            continue
        audio = audio_root / str(job["master_mp3_path"])
        transcript = audio_root / str(job["input_path"])
        if not audio.is_file() or file_sha256(audio) != job.get("master_mp3_sha256"):
            failures.append(f"audio:{expected}")
        if not transcript.is_file():
            failures.append(f"transcript:{expected}")
        elif text_sha256(transcript.read_text(encoding="utf-8")) != job.get("text_sha256"):
            failures.append(f"transcript-hash:{expected}")
        if segment.get("source_audio_sha256") != job.get("master_mp3_sha256"):
            failures.append(f"plan-audio:{expected}")
        if segment.get("source_text_sha256") != job.get("text_sha256"):
            failures.append(f"plan-text:{expected}")
    if failures:
        raise VideoProductionError(
            f"Production input verification failed ({len(failures)}): {failures[:12]}"
        )


def prepare_video_production(
    *,
    run_id: str = DEFAULT_VIDEO_RUN_ID,
    audio_run_id: str = DEFAULT_AUDIO_RUN_ID,
    narration_manifest: Path = DEFAULT_NARRATION_MANIFEST,
    catalog_plan: Path = DEFAULT_CATALOG_PLAN,
    listening_edition: Path = DEFAULT_LISTENING_EDITION,
) -> dict[str, Any]:
    root = _run_root(run_id)
    state_path = root / "RUN.json"
    immutable = _immutable_inputs(
        narration_manifest=narration_manifest,
        catalog_plan=catalog_plan,
        listening_edition=listening_edition,
        audio_run_id=audio_run_id,
    )
    manifest = _read_object(narration_manifest)
    plan = _read_object(catalog_plan)
    _validate_inputs(manifest, plan, audio_run_id=audio_run_id)

    if state_path.exists():
        state = _read_object(state_path)
        if state.get("immutable") != immutable:
            raise VideoProductionError(
                "Production manifest or implementation changed; refusing a mixed-version resume"
            )
        return state

    root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "chunk_index": int(job["chunk_index"]),
            "job_id": str(job["job_id"]),
            "alignment": "pending",
            "render": "pending",
            "attempts": 0,
        }
        for job in manifest["jobs"]
    ]
    state = {
        "version": "quran-video-production-run-v1",
        "run_id": run_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "prepared",
        "immutable": immutable,
        "jobs": jobs,
        "catalogs": {"juz": "pending", "surahs": "pending"},
    }
    atomic_json(state_path, state)
    return state


def migrate_video_duration_cap(run_id: str = DEFAULT_VIDEO_RUN_ID) -> dict[str, Any]:
    """Record the v1 duration-cap fix without disturbing completed alignments."""

    root = _run_root(run_id)
    state_path = root / "RUN.json"
    state = _read_object(state_path)
    immutable = state.get("immutable")
    if not isinstance(immutable, dict):
        raise VideoProductionError("Production RUN.json has no immutable input block")
    current = _immutable_inputs(
        narration_manifest=Path(str(immutable["narration_manifest"])),
        catalog_plan=Path(str(immutable["catalog_plan"])),
        listening_edition=Path(str(immutable["listening_edition"])),
        audio_run_id=str(immutable["audio_run_id"]),
    )
    prior_non_source = {key: value for key, value in immutable.items() if key != "source_hashes"}
    current_non_source = {key: value for key, value in current.items() if key != "source_hashes"}
    if prior_non_source != current_non_source:
        raise VideoProductionError("Non-code production inputs changed; migration refused")
    if any(job.get("alignment") != "complete" for job in state.get("jobs", [])):
        raise VideoProductionError("Duration-cap migration requires all alignments to be complete")
    if any(value == "complete" for value in state.get("catalogs", {}).values()):
        raise VideoProductionError("Duration-cap migration cannot follow catalog assembly")
    if (root / "PRODUCTION_COMPLETE.json").exists():
        raise VideoProductionError("Completed production cannot be migrated in place")

    marker_path = root / "MIGRATION_DURATION_CAP_v1.json"
    affected = [
        int(job["chunk_index"])
        for job in state["jobs"]
        if job.get("render") in {"complete", "failed", "rendering"}
    ]
    prior_hashes = immutable.get("source_hashes")
    state["immutable"] = current
    state["status"] = "aligned"
    for job in state["jobs"]:
        job["render"] = "pending"
        job.pop("render_sha256", None)
        job.pop("render_error", None)
    state["catalogs"] = {"juz": "pending", "surahs": "pending"}
    _save_state(root, state)
    marker = {
        "version": "quran-video-production-migration-v1",
        "run_id": run_id,
        "migrated_at": utc_now(),
        "reason": "Cap the muxed output duration to the canonical display/audio window",
        "preserved_alignments": 313,
        "invalidated_render_indices": affected,
        "prior_source_hashes": prior_hashes,
        "new_source_hashes": current["source_hashes"],
    }
    atomic_json(marker_path, marker)
    return marker


def migrate_video_duration_contract(run_id: str = DEFAULT_VIDEO_RUN_ID) -> dict[str, Any]:
    """Adopt the MP3-tail tolerance while proving aligned speech is preserved."""

    root = _run_root(run_id)
    state = _read_object(root / "RUN.json")
    immutable = state.get("immutable")
    if not isinstance(immutable, dict):
        raise VideoProductionError("Production RUN.json has no immutable input block")
    current = _immutable_inputs(
        narration_manifest=Path(str(immutable["narration_manifest"])),
        catalog_plan=Path(str(immutable["catalog_plan"])),
        listening_edition=Path(str(immutable["listening_edition"])),
        audio_run_id=str(immutable["audio_run_id"]),
    )
    prior_non_source = {key: value for key, value in immutable.items() if key != "source_hashes"}
    current_non_source = {key: value for key, value in current.items() if key != "source_hashes"}
    if prior_non_source != current_non_source:
        raise VideoProductionError("Non-code production inputs changed; migration refused")
    if any(job.get("alignment") != "complete" for job in state.get("jobs", [])):
        raise VideoProductionError("Duration-contract migration requires complete alignments")
    if any(value == "complete" for value in state.get("catalogs", {}).values()):
        raise VideoProductionError("Duration-contract migration cannot follow catalog assembly")
    if (root / "PRODUCTION_COMPLETE.json").exists():
        raise VideoProductionError("Completed production cannot be migrated in place")

    manifest = _read_object(Path(str(immutable["narration_manifest"])))
    plan = _read_object(Path(str(immutable["catalog_plan"])))
    prior_hashes = immutable.get("source_hashes")
    preserved: list[tuple[Path, dict[str, Any]]] = []
    for job, segment, job_state in zip(manifest["jobs"], plan["segments"], state["jobs"]):
        if job_state.get("render") != "complete":
            continue
        index = int(job["chunk_index"])
        qa_path = root / "segment-work" / f"{index:04d}" / "QA.json"
        video_path = root / str(segment["output"])
        alignment_path = root / "alignments" / f"{index:04d}" / "normalized.json"
        qa = _read_object(qa_path)
        if (
            qa.get("implementation_hashes") != prior_hashes
            or qa.get("render", {}).get("sha256") != file_sha256(video_path)
            or qa.get("alignment", {}).get("sha256") != file_sha256(alignment_path)
            or qa.get("checks", {}).get("decode_passed") is not True
            or qa.get("encoded_timeline", {}).get("passed") is not True
        ):
            raise VideoProductionError(
                f"Completed render provenance failed during migration at chunk {index:04d}"
            )
        display = _read_object(root / "segment-work" / f"{index:04d}" / "DISPLAY_EVENTS.json")
        expected = float(display["selection"]["duration"])
        clip_start = float(display["selection"]["clip_start"])
        content_end = max(float(event["source_end"]) for event in display["events"]) - clip_start
        actual = float(_probe_video(video_path)["duration_seconds"])
        delta = abs(actual - expected)
        margin = actual - content_end
        if delta > MAX_DURATION_DELTA_SECONDS or margin < MIN_CONTENT_END_MARGIN_SECONDS:
            raise VideoProductionError(
                f"Completed render fails the revised duration contract at chunk {index:04d}"
            )
        qa["checks"]["duration_seconds"] = round(actual, 3)
        qa["checks"]["duration_delta_seconds"] = round(delta, 3)
        qa["checks"]["duration_tolerance_seconds"] = MAX_DURATION_DELTA_SECONDS
        qa["checks"]["content_end_margin_seconds"] = round(margin, 3)
        qa["implementation_hashes"] = current["source_hashes"]
        preserved.append((qa_path, qa))

    reset = []
    for job_state in state["jobs"]:
        if job_state.get("render") in {"failed", "rendering"}:
            reset.append(int(job_state["chunk_index"]))
            job_state["render"] = "pending"
            job_state.pop("render_sha256", None)
            job_state.pop("render_error", None)
    state["immutable"] = current
    state["status"] = "rendering"
    _save_state(root, state)
    for qa_path, qa in preserved:
        atomic_json(qa_path, qa)

    marker = {
        "version": "quran-video-duration-contract-migration-v1",
        "run_id": run_id,
        "migrated_at": utc_now(),
        "reason": "Permit the intentional 250 ms visual tail while forbidding speech cutoff",
        "duration_tolerance_seconds": MAX_DURATION_DELTA_SECONDS,
        "minimum_content_end_margin_seconds": MIN_CONTENT_END_MARGIN_SECONDS,
        "preserved_and_revalidated_renders": len(preserved),
        "reset_render_indices": reset,
        "prior_source_hashes": prior_hashes,
        "new_source_hashes": current["source_hashes"],
    }
    atomic_json(root / "MIGRATION_DURATION_CONTRACT_v1.json", marker)
    return marker


def migrate_video_loudness_contract(run_id: str = DEFAULT_VIDEO_RUN_ID) -> dict[str, Any]:
    """Adopt a practical short-clip LUFS tolerance while preserving peak safety."""

    root = _run_root(run_id)
    state = _read_object(root / "RUN.json")
    immutable = state.get("immutable")
    if not isinstance(immutable, dict):
        raise VideoProductionError("Production RUN.json has no immutable input block")
    current = _immutable_inputs(
        narration_manifest=Path(str(immutable["narration_manifest"])),
        catalog_plan=Path(str(immutable["catalog_plan"])),
        listening_edition=Path(str(immutable["listening_edition"])),
        audio_run_id=str(immutable["audio_run_id"]),
    )
    prior_non_source = {key: value for key, value in immutable.items() if key != "source_hashes"}
    current_non_source = {key: value for key, value in current.items() if key != "source_hashes"}
    if prior_non_source != current_non_source:
        raise VideoProductionError("Non-code production inputs changed; migration refused")
    if any(job.get("alignment") != "complete" for job in state.get("jobs", [])):
        raise VideoProductionError("Loudness-contract migration requires complete alignments")
    if any(value == "complete" for value in state.get("catalogs", {}).values()):
        raise VideoProductionError("Loudness-contract migration cannot follow catalog assembly")
    if (root / "PRODUCTION_COMPLETE.json").exists():
        raise VideoProductionError("Completed production cannot be migrated in place")

    manifest = _read_object(Path(str(immutable["narration_manifest"])))
    plan = _read_object(Path(str(immutable["catalog_plan"])))
    prior_hashes = immutable.get("source_hashes")
    preserved: list[tuple[Path, dict[str, Any]]] = []
    for job, segment, job_state in zip(manifest["jobs"], plan["segments"], state["jobs"]):
        if job_state.get("render") != "complete":
            continue
        index = int(job["chunk_index"])
        qa_path = root / "segment-work" / f"{index:04d}" / "QA.json"
        video_path = root / str(segment["output"])
        alignment_path = root / "alignments" / f"{index:04d}" / "normalized.json"
        qa = _read_object(qa_path)
        checks = qa.get("checks", {})
        loudness = checks.get("output_loudness", {})
        integrated = float(loudness.get("input_i", 999))
        true_peak = float(loudness.get("input_tp", 999))
        if (
            qa.get("implementation_hashes") != prior_hashes
            or qa.get("render", {}).get("sha256") != file_sha256(video_path)
            or qa.get("alignment", {}).get("sha256") != file_sha256(alignment_path)
            or checks.get("decode_passed") is not True
            or qa.get("encoded_timeline", {}).get("passed") is not True
            or abs(integrated - TARGET_LOUDNESS_LUFS) > MAX_LOUDNESS_DELTA_LUFS
            or true_peak > -1.0
        ):
            raise VideoProductionError(
                f"Completed render fails loudness migration at chunk {index:04d}"
            )
        checks["loudness_target_lufs"] = TARGET_LOUDNESS_LUFS
        checks["loudness_tolerance_lufs"] = MAX_LOUDNESS_DELTA_LUFS
        qa["implementation_hashes"] = current["source_hashes"]
        preserved.append((qa_path, qa))

    reset = []
    for job_state in state["jobs"]:
        if job_state.get("render") in {"failed", "rendering"}:
            reset.append(int(job_state["chunk_index"]))
            job_state["render"] = "pending"
            job_state.pop("render_sha256", None)
            job_state.pop("render_error", None)
    state["immutable"] = current
    state["status"] = "rendering"
    _save_state(root, state)
    for qa_path, qa in preserved:
        atomic_json(qa_path, qa)

    marker = {
        "version": "quran-video-loudness-contract-migration-v1",
        "run_id": run_id,
        "migrated_at": utc_now(),
        "reason": "Use a practical integrated-loudness tolerance for short clips",
        "target_loudness_lufs": TARGET_LOUDNESS_LUFS,
        "loudness_tolerance_lufs": MAX_LOUDNESS_DELTA_LUFS,
        "true_peak_ceiling_dbtp": -1.0,
        "preserved_and_revalidated_renders": len(preserved),
        "reset_render_indices": reset,
        "prior_source_hashes": prior_hashes,
        "new_source_hashes": current["source_hashes"],
    }
    atomic_json(root / "MIGRATION_LOUDNESS_CONTRACT_v1.json", marker)
    return marker


def _load_run(run_id: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _run_root(run_id)
    state = _read_object(root / "RUN.json")
    immutable = state.get("immutable")
    if not isinstance(immutable, dict):
        raise VideoProductionError("Production RUN.json has no immutable input block")
    current = _immutable_inputs(
        narration_manifest=Path(str(immutable["narration_manifest"])),
        catalog_plan=Path(str(immutable["catalog_plan"])),
        listening_edition=Path(str(immutable["listening_edition"])),
        audio_run_id=str(immutable["audio_run_id"]),
    )
    if current != immutable:
        raise VideoProductionError("Production inputs changed; refusing resume")
    manifest = _read_object(Path(str(immutable["narration_manifest"])))
    plan = _read_object(Path(str(immutable["catalog_plan"])))
    _validate_inputs(manifest, plan, audio_run_id=str(immutable["audio_run_id"]))
    return root, state, manifest, plan


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json(root / "RUN.json", state)


def _validate_alignment(job: dict[str, Any], payload: dict[str, Any], audio: Path) -> None:
    if payload.get("engine") != "elevenlabs-forced-alignment":
        raise VideoProductionError(f"Wrong alignment engine for chunk {job['chunk_index']}")
    if payload.get("audio_sha256") != job.get("master_mp3_sha256"):
        raise VideoProductionError(f"Alignment audio drift at chunk {job['chunk_index']}")
    if payload.get("transcript_sha256") != job.get("text_sha256"):
        raise VideoProductionError(f"Alignment text drift at chunk {job['chunk_index']}")
    spans = payload.get("spans")
    expected = job.get("spans")
    if not isinstance(spans, list) or not isinstance(expected, list) or len(spans) != len(expected):
        raise VideoProductionError(f"Alignment span coverage failed at chunk {job['chunk_index']}")
    expected_refs = [span.get("ref") for span in expected if span.get("kind") == "ayah"]
    actual_refs = [span.get("ref") for span in spans if span.get("kind") == "ayah"]
    if actual_refs != expected_refs:
        raise VideoProductionError(f"Alignment ayah coverage failed at chunk {job['chunk_index']}")
    prior_start = -1.0
    for span in spans:
        start = float(span.get("start", -1))
        end = float(span.get("end", -1))
        if start < 0 or end <= start or start < prior_start:
            raise VideoProductionError(f"Non-monotonic alignment at chunk {job['chunk_index']}")
        prior_start = start
    if file_sha256(audio) != payload.get("audio_sha256"):
        raise VideoProductionError(f"Live audio hash drift at chunk {job['chunk_index']}")


def _provider_blocker(message: str) -> bool:
    lowered = message.casefold()
    return any(
        marker in lowered
        for marker in ("(401)", "(402)", "(403)", "authentication", "billing", "quota")
    )


def align_video_production(
    *,
    run_id: str,
    api_key: str,
    max_attempts: int = 3,
    request_timeout: int = 600,
    limit: int | None = None,
) -> dict[str, Any]:
    if not api_key.strip():
        raise VideoProductionError("ELEVENLABS_API_KEY is required for missing alignments")
    root, state, manifest, _ = _load_run(run_id)
    audio_root = _audio_root(str(state["immutable"]["audio_run_id"]))
    completed_this_call = 0
    state["status"] = "aligning"
    _save_state(root, state)

    for job, job_state in zip(manifest["jobs"], state["jobs"]):
        index = int(job["chunk_index"])
        alignment_dir = root / "alignments" / f"{index:04d}"
        raw_path = alignment_dir / "raw.json"
        normalized_path = alignment_dir / "normalized.json"
        audio = audio_root / str(job["master_mp3_path"])
        transcript_path = audio_root / str(job["input_path"])
        transcript = transcript_path.read_text(encoding="utf-8")

        if normalized_path.exists():
            payload = _read_object(normalized_path)
            _validate_alignment(job, payload, audio)
            job_state["alignment"] = "complete"
            job_state["alignment_sha256"] = file_sha256(normalized_path)
            continue
        if limit is not None and completed_this_call >= limit:
            break

        alignment_dir.mkdir(parents=True, exist_ok=True)
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            job_state["attempts"] = int(job_state.get("attempts", 0)) + 1
            job_state["alignment"] = "requesting"
            _save_state(root, state)
            try:
                if raw_path.exists():
                    raw = _read_object(raw_path)
                else:
                    raw = request_forced_alignment(
                        audio_path=audio,
                        transcript=transcript,
                        api_key=api_key,
                        output_path=raw_path,
                        timeout_seconds=request_timeout,
                    )
                payload = normalize_forced_alignment(
                    raw_payload=raw,
                    transcript=transcript,
                    spans=job["spans"],
                    audio_path=audio,
                    raw_path=raw_path,
                    output_path=normalized_path,
                )
                _validate_alignment(job, payload, audio)
                job_state["alignment"] = "complete"
                job_state["alignment_sha256"] = file_sha256(normalized_path)
                job_state.pop("alignment_error", None)
                completed_this_call += 1
                _save_state(root, state)
                print(f"alignment {index:04d}/0313 complete", flush=True)
                break
            except (AlignmentError, VideoProductionError, OSError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                job_state["alignment"] = "failed"
                job_state["alignment_error"] = last_error
                atomic_json(
                    alignment_dir / f"failure-attempt-{job_state['attempts']:02d}.json",
                    {"at": utc_now(), "attempt": attempt, "error": last_error},
                )
                _save_state(root, state)
                if _provider_blocker(last_error):
                    raise VideoProductionError(last_error) from exc
                if attempt < max_attempts:
                    time.sleep(min(30, 5 * attempt))
        else:
            raise VideoProductionError(
                f"Alignment exhausted for chunk {index:04d}: {last_error}"
            )

    counts = video_production_status(run_id)
    if counts["alignment"]["complete"] == 313:
        state["status"] = "aligned"
    _save_state(root, state)
    return counts


def _listening_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_object(path)
    rows = payload.get("ayahs")
    if not isinstance(rows, list) or len(rows) != 6236:
        raise VideoProductionError("Listening edition must contain 6,236 ayahs")
    return {str(row["ref"]): row for row in rows if isinstance(row, dict)}


def _valid_completed_render(
    *,
    qa_path: Path,
    video_path: Path,
    alignment_path: Path,
    audio_path: Path,
) -> bool:
    if not qa_path.is_file() or not video_path.is_file():
        return False
    qa = _read_object(qa_path)
    return (
        qa.get("render", {}).get("sha256") == file_sha256(video_path)
        and qa.get("render", {}).get("source_audio_sha256") == file_sha256(audio_path)
        and qa.get("alignment", {}).get("sha256") == file_sha256(alignment_path)
        and qa.get("checks", {}).get("decode_passed") is True
        and qa.get("srt_qa", {}).get("timeline_identity") is True
        and qa.get("encoded_timeline", {}).get("passed") is True
        and qa.get("implementation_hashes") == _source_hashes()
    )


def render_video_production(
    *,
    run_id: str,
    limit: int | None = None,
) -> dict[str, Any]:
    root, state, manifest, plan = _load_run(run_id)
    if any(job_state.get("alignment") != "complete" for job_state in state["jobs"]):
        raise VideoProductionError("All 313 alignments must complete before rendering")
    listening = _listening_index(Path(str(state["immutable"]["listening_edition"])))
    audio_root = _audio_root(str(state["immutable"]["audio_run_id"]))
    rendered_this_call = 0
    state["status"] = "rendering"
    _save_state(root, state)

    for job, segment, job_state in zip(manifest["jobs"], plan["segments"], state["jobs"]):
        index = int(job["chunk_index"])
        alignment_path = root / "alignments" / f"{index:04d}" / "normalized.json"
        audio_path = audio_root / str(job["master_mp3_path"])
        output_path = root / str(segment["output"])
        work = root / "segment-work" / f"{index:04d}"
        qa_path = work / "QA.json"
        if _valid_completed_render(
            qa_path=qa_path,
            video_path=output_path,
            alignment_path=alignment_path,
            audio_path=audio_path,
        ):
            job_state["render"] = "complete"
            job_state["render_sha256"] = file_sha256(output_path)
            continue
        if limit is not None and rendered_this_call >= limit:
            break

        alignment = _read_object(alignment_path)
        _validate_alignment(job, alignment, audio_path)
        display = build_display_events(alignment)
        first_row = listening[str(job["start_ref"])]
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / "DISPLAY_EVENTS.json", display)
        job_state["render"] = "rendering"
        _save_state(root, state)
        try:
            frames = render_frames(
                display=display,
                output_dir=work / "frames-2x",
                surah_name=str(first_row["surah_name_en"]),
                surah_meaning=str(first_row["surah_meaning_en"]),
                juz_number=int(job["juz_number"]),
            )
            mobile = write_mobile_review(
                frames=frames,
                output_dir=work / "mobile-360",
                max_samples=3,
            )
            srt_path = write_srt(display, work / "captions.en.srt")
            srt_qa = validate_srt_identity(display, srt_path)
            metadata = write_metadata_kit(
                output_path=work / "SEGMENT_METADATA.json",
                surah_number=int(first_row["surah"]),
                surah_name=str(first_row["surah_name_en"]),
                surah_meaning=str(first_row["surah_meaning_en"]),
                start_ref=str(job["start_ref"]),
                end_ref=str(job["end_ref"]),
            )
            output_path.unlink(missing_ok=True)
            render_result = render_video(
                display=display,
                frames=frames,
                audio_path=audio_path,
                output_path=output_path,
            )
            minimum_content_duration = (
                max(float(event["source_end"]) for event in display["events"])
                - float(display["selection"]["clip_start"])
            )
            checks = validate_video(
                output_path,
                float(display["selection"]["duration"]),
                minimum_content_duration=minimum_content_duration,
            )
            encoded = validate_encoded_timeline(
                video_path=output_path,
                display=display,
                frames=frames,
                max_samples=6,
            )
            qa = {
                "version": "quran-video-segment-qa-v1",
                "chunk_index": index,
                "job_id": job["job_id"],
                "alignment": {
                    "path": str(alignment_path),
                    "sha256": file_sha256(alignment_path),
                    "engine": alignment.get("engine"),
                },
                "selection": display["selection"],
                "display_events": len(display["events"]),
                "frames": len(frames),
                "mobile_review": mobile,
                "srt": str(srt_path),
                "srt_qa": srt_qa,
                "metadata": metadata,
                "render": render_result,
                "checks": checks,
                "encoded_timeline": encoded,
                "implementation_hashes": _source_hashes(),
            }
            atomic_json(qa_path, qa)
            job_state["render"] = "complete"
            job_state["render_sha256"] = file_sha256(output_path)
            job_state.pop("render_error", None)
            rendered_this_call += 1
            _save_state(root, state)
            print(f"render {index:04d}/0313 complete", flush=True)
        except (AlignmentError, OSError, subprocess.CalledProcessError) as exc:
            job_state["render"] = "failed"
            job_state["render_error"] = str(exc)
            atomic_json(work / "FAILURE.json", {"at": utc_now(), "error": str(exc)})
            _save_state(root, state)
            raise VideoProductionError(f"Render failed at chunk {index:04d}: {exc}") from exc

    counts = video_production_status(run_id)
    if counts["render"]["complete"] == 313:
        state["status"] = "rendered"
    _save_state(root, state)
    return counts


def _ffconcat_line(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


def _probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    video = next((row for row in payload["streams"] if row["codec_type"] == "video"), None)
    audio = next((row for row in payload["streams"] if row["codec_type"] == "audio"), None)
    if video is None or audio is None:
        raise VideoProductionError(f"Catalog video is missing a stream: {path}")
    return {
        "duration_seconds": float(payload["format"]["duration"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "pixel_format": video.get("pix_fmt"),
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": int(audio.get("sample_rate", 0)),
        "audio_channels": int(audio.get("channels", 0)),
    }


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_catalog_srt(
    *,
    displays: list[dict[str, Any]],
    segment_durations: list[float],
    output_path: Path,
) -> Path:
    if len(displays) != len(segment_durations) or not displays:
        raise VideoProductionError("Catalog captions require matching displays and durations")
    blocks: list[str] = []
    offset = 0.0
    cue = 1
    for display, duration in zip(displays, segment_durations):
        for event in display.get("events", []):
            start = offset + float(event["start"])
            end = offset + min(float(event["end"]), duration)
            if end <= start:
                continue
            blocks.append(
                f"{cue}\n{_srt_time(start)} --> {_srt_time(end)}\n{event['caption']}"
            )
            cue += 1
        offset += duration
    atomic_text(output_path, "\n\n".join(blocks) + "\n")
    return output_path


def _chapter_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _catalog_chapters(
    *,
    kind: str,
    jobs: list[dict[str, Any]],
    displays: list[dict[str, Any]],
    segment_durations: list[float],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    offset = 0.0
    prior_surah: int | None = None
    ayah_points: list[tuple[float, str]] = []
    for job, display, duration in zip(jobs, displays, segment_durations):
        surah = int(job["surah_number"])
        if prior_surah != surah or not points:
            points.append(
                {
                    "seconds": round(offset, 3),
                    "timestamp": _chapter_time(offset),
                    "label": f"Surah {surah}: {job['surah_name']}",
                    "ref": job["start_ref"],
                }
            )
        prior_surah = surah
        seen: set[str] = set()
        for event in display.get("events", []):
            ref = event.get("active_ref")
            if isinstance(ref, str) and ref not in seen:
                ayah_points.append((offset + float(event["start"]), ref))
                seen.add(ref)
        offset += duration
    if kind == "surah" and offset >= 600:
        target = 600.0
        used_refs = {str(row["ref"]) for row in points}
        while target < offset:
            candidates = [row for row in ayah_points if row[0] >= target and row[1] not in used_refs]
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
            used_refs.add(ref)
            target += 600.0
    points.sort(key=lambda row: float(row["seconds"]))
    if points:
        points[0]["seconds"] = 0.0
        points[0]["timestamp"] = "0:00"
    if len(points) < 3:
        return []
    return points


def _metadata(
    *,
    kind: str,
    entry: dict[str, Any],
    jobs: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    listening: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    chapter_text = "\n".join(f"{row['timestamp']} {row['label']}" for row in chapters)
    disclosure = (
        "English listening translation with synchronized captions aligned to the exact "
        "published text."
    )
    if kind == "juz":
        number = int(entry["number"])
        title = f"Quran Para {number} of 30 (Juz {number}) | English Listening Edition"
        label = f"PARA {number:02d} OF 30"
        playlist = "The Quran by Para - English Listening Edition"
        description = (
            f"Para {number} of 30 (Juz {number}), "
            f"{jobs[0]['start_ref']}–{jobs[-1]['end_ref']}.\n\n{disclosure}"
        )
    else:
        number = int(entry["number"])
        first = listening[str(jobs[0]["start_ref"])]
        name = str(entry["name"])
        meaning = str(first["surah_meaning_en"])
        title = f"{name} ({number}) - {meaning} | English Quran Audiobook"
        label = f"{number:03d}  {name}"
        playlist = "The Quran by Surah - English Listening Edition"
        description = f"{name}, {jobs[0]['start_ref']}–{jobs[-1]['end_ref']}.\n\n{disclosure}"
    if chapter_text:
        description += "\n\n" + chapter_text
    return {
        "version": "quran-youtube-metadata-v1",
        "title": title,
        "description": description + "\n",
        "thumbnail_label": label,
        "playlist": playlist,
        "language": "English",
        "caption_language": "English",
        "synthetic_content_disclosure": True,
        "chapters": chapters,
    }


def _assemble_one(
    *,
    root: Path,
    kind: str,
    entry: dict[str, Any],
    manifest_jobs: list[dict[str, Any]],
    plan_segments: list[dict[str, Any]],
    listening: dict[str, dict[str, Any]],
    decode_check: bool,
) -> dict[str, Any]:
    indices = [int(value) for value in entry["segment_indices"]]
    jobs = [manifest_jobs[index - 1] for index in indices]
    segment_rows = [plan_segments[index - 1] for index in indices]
    inputs = [root / str(row["output"]) for row in segment_rows]
    for path in inputs:
        if not path.is_file():
            raise VideoProductionError(f"Missing rendered segment: {path}")
    output = root / str(entry["output"])
    sidecar = output.with_suffix("")
    qa_path = sidecar.parent / f"{sidecar.name}.qa.json"
    contract = catalog_contract(
        inputs, [root / "segment-work" / f"{index:04d}" / "DISPLAY_EVENTS.json" for index in indices],
        {"kind": kind, "entry": entry, "jobs": jobs, "listening": listening,
         "implementation_hashes": _source_hashes()},
    )
    def restore_sidecars(captions_path: Path, metadata_path: Path) -> None:
        durations = [float(_probe_video(path)["duration_seconds"]) for path in inputs]
        displays = [_read_object(root / "segment-work" / f"{index:04d}" / "DISPLAY_EVENTS.json")
                    for index in indices]
        write_catalog_srt(displays=displays, segment_durations=durations, output_path=captions_path)
        chapters = _catalog_chapters(kind=kind, jobs=jobs, displays=displays, segment_durations=durations)
        atomic_json(metadata_path, _metadata(
            kind=kind, entry=entry, jobs=jobs, chapters=chapters, listening=listening
        ))

    prior = validate_cached_catalog(output, qa_path, contract, "en", restore_sidecars)
    if prior is not None:
        return prior

    output.parent.mkdir(parents=True, exist_ok=True)
    concat = output.with_suffix(".concat.txt")
    temp = output.with_name(f".{output.name}.tmp.mp4")
    atomic_text(concat, "\n".join(_ffconcat_line(path) for path in inputs) + "\n")
    try:
        subprocess.run(
            [
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
                str(concat),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(temp),
            ],
            check=True,
        )
        os.replace(temp, output)
    finally:
        concat.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)

    probes = [_probe_video(path) for path in inputs]
    segment_durations = [float(row["duration_seconds"]) for row in probes]
    displays = [
        _read_object(root / "segment-work" / f"{index:04d}" / "DISPLAY_EVENTS.json")
        for index in indices
    ]
    captions = output.with_suffix(".en.srt")
    write_catalog_srt(
        displays=displays,
        segment_durations=segment_durations,
        output_path=captions,
    )
    chapters = _catalog_chapters(
        kind=kind,
        jobs=jobs,
        displays=displays,
        segment_durations=segment_durations,
    )
    metadata_path = output.with_suffix(".metadata.json")
    metadata = _metadata(
        kind=kind,
        entry=entry,
        jobs=jobs,
        chapters=chapters,
        listening=listening,
    )
    atomic_json(metadata_path, metadata)

    probe = _probe_video(output)
    expected = sum(segment_durations)
    tolerance = max(1.0, expected * 0.001)
    if abs(float(probe["duration_seconds"]) - expected) > tolerance:
        raise VideoProductionError(
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
        raise VideoProductionError(f"Catalog media contract failed: {output}")
    if decode_check:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
            check=True,
        )
    qa = {
        "version": "quran-video-catalog-qa-v1",
        "input_contract": contract,
        "kind": kind,
        "number": int(entry["number"]),
        "path": str(output),
        "sha256": file_sha256(output),
        "bytes": output.stat().st_size,
        "source_segments": indices,
        "source_segment_sha256s": [file_sha256(path) for path in inputs],
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


def _write_release_checksums(root: Path) -> Path:
    paths = sorted(
        path
        for folder in (root / "juz", root / "surahs")
        for path in folder.glob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    checksum = root / "SHA256SUMS.txt"
    atomic_text(
        checksum,
        "\n".join(f"{file_sha256(path)}  {path.relative_to(root)}" for path in paths) + "\n",
    )
    return checksum


def assemble_video_production(
    *,
    run_id: str,
    catalog: str = "all",
    decode_check: bool = True,
) -> dict[str, Any]:
    if catalog not in {"juz", "surahs", "all"}:
        raise VideoProductionError("Catalog must be juz, surahs, or all")
    root, state, manifest, plan = _load_run(run_id)
    if any(job_state.get("render") != "complete" for job_state in state["jobs"]):
        raise VideoProductionError("All 313 segments must pass QA before catalog assembly")
    listening = _listening_index(Path(str(state["immutable"]["listening_edition"])))
    state["status"] = "assembling"
    _save_state(root, state)
    selected = ["juz", "surahs"] if catalog == "all" else [catalog]
    results: dict[str, list[dict[str, Any]]] = {}
    for kind in selected:
        results[kind] = []
        for entry in plan[kind]:
            qa = _assemble_one(
                root=root,
                kind=kind,
                entry=entry,
                manifest_jobs=manifest["jobs"],
                plan_segments=plan["segments"],
                listening=listening,
                decode_check=decode_check,
            )
            results[kind].append(qa)
            print(f"assemble {kind} {int(entry['number']):03d} complete", flush=True)
        state["catalogs"][kind] = "complete"
        _save_state(root, state)

    checksum = _write_release_checksums(root)
    if state["catalogs"].get("juz") == "complete" and state["catalogs"].get("surahs") == "complete":
        state["status"] = "complete"
        marker = {
            "version": "quran-video-production-complete-v1",
            "run_id": run_id,
            "completed_at": utc_now(),
            "segments": 313,
            "juz_videos": 30,
            "surah_videos": 114,
            "checksums": str(checksum),
            "checksums_sha256": file_sha256(checksum),
            "decode_checked": bool(decode_check),
        }
        atomic_json(root / "PRODUCTION_COMPLETE.json", marker)
    _save_state(root, state)
    return {"run_id": run_id, "results": results, "status": state["status"]}


def video_production_status(run_id: str = DEFAULT_VIDEO_RUN_ID) -> dict[str, Any]:
    root = _run_root(run_id)
    state = _read_object(root / "RUN.json")
    jobs = state.get("jobs", [])

    def counts(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for job in jobs:
            value = str(job.get(field, "pending"))
            result[value] = result.get(value, 0) + 1
        return result

    return {
        "run_id": run_id,
        "status": state.get("status"),
        "alignment": counts("alignment"),
        "render": counts("render"),
        "catalogs": state.get("catalogs"),
        "production_complete": (root / "PRODUCTION_COMPLETE.json").is_file(),
        "root": str(root),
    }
