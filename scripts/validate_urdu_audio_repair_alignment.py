#!/usr/bin/env python3
"""Validate a repaired Urdu master clip against its frozen transcript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx_whisper

from quran_translate.config import file_sha256, text_sha256
from quran_translate.db import utc_now
from quran_translate.production_packets import atomic_json
from quran_translate.urdu_video_production import (
    _jsonable,
    _normalize_urdu_alignment,
    _spans,
    _validate_alignment,
)


MODEL = "mlx-community/whisper-large-v3-turbo"


def _read_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    return [dict(item) for item in value]


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return dict(value)


def _apply_component_boundaries(
    *,
    payload: dict[str, Any],
    unit: dict[str, Any],
    generation_path: Path,
    audio: Path,
    normalized_path: Path,
) -> None:
    generation = _read_object(generation_path)
    records = generation.get("ayah_records")
    refs = [str(ref) for ref in unit["refs"]]
    if (
        generation.get("status") != "ready_for_alignment_validation"
        or generation.get("unit_id") != unit["unit_id"]
        or generation.get("candidate_sha256") != file_sha256(audio)
        or generation.get("refs") != refs
        or not isinstance(records, list)
        or [str(row.get("ref")) for row in records] != refs
    ):
        raise RuntimeError("Component generation evidence does not match the candidate")

    spans = payload.get("spans")
    if not isinstance(spans, list) or len(spans) != len(records):
        raise RuntimeError("Component boundaries do not match normalized ayah spans")

    cursor = 0.0
    evidence: list[dict[str, Any]] = []
    for span, record in zip(spans, records):
        component = Path(str(record["normalized_path"]))
        duration = float(record.get("probe", {}).get("duration_seconds", 0))
        if (
            not component.is_file()
            or file_sha256(component) != record.get("normalized_sha256")
            or duration <= 0
        ):
            raise RuntimeError(f"Invalid component evidence for {record.get('ref')}")
        start = cursor
        cursor += duration
        span.update(
            {
                "start": round(start, 3),
                "end": round(cursor, 3),
                "timing_source": "audited_component_boundary",
            }
        )
        evidence.append(
            {
                "ref": record["ref"],
                "path": str(component),
                "sha256": record["normalized_sha256"],
                "duration_seconds": duration,
            }
        )

    candidate_duration = float(generation.get("probe", {}).get("duration_seconds", 0))
    if candidate_duration <= 0 or abs(cursor - candidate_duration) > 0.5:
        raise RuntimeError("Component durations do not cohere with the candidate")
    payload["engine"] = f"{payload['engine']}+audited-component-boundaries"
    payload["component_boundary_evidence"] = {
        "generation_path": str(generation_path),
        "generation_sha256": file_sha256(generation_path),
        "summed_duration_seconds": round(cursor, 3),
        "candidate_duration_seconds": candidate_duration,
        "components": evidence,
    }
    atomic_json(normalized_path, payload)


def validate(
    *,
    audio: Path,
    units_path: Path,
    unit_id: str,
    output_dir: Path,
    generation_path: Path | None = None,
) -> dict[str, Any]:
    units = {item["unit_id"]: item for item in _read_list(units_path)}
    unit = units.get(unit_id)
    if unit is None:
        raise RuntimeError(f"Unknown unit: {unit_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.json"
    normalized_path = output_dir / "normalized.json"
    validation_path = output_dir / "VALIDATION.json"

    if raw_path.is_file():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        raw = _jsonable(
            mlx_whisper.transcribe(
                str(audio),
                path_or_hf_repo=MODEL,
                language="ur",
                task="transcribe",
                word_timestamps=True,
                verbose=False,
                condition_on_previous_text=False,
                initial_prompt=str(unit["speech_text"]),
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                hallucination_silence_threshold=2.0,
            )
        )
        atomic_json(raw_path, raw)

    payload = _normalize_urdu_alignment(
        raw=raw,
        transcript=str(unit["speech_text"]),
        spans=_spans(unit),
        audio_path=audio,
        raw_path=raw_path,
        normalized_path=normalized_path,
        model=MODEL,
    )
    if generation_path is not None:
        _apply_component_boundaries(
            payload=payload,
            unit=unit,
            generation_path=generation_path,
            audio=audio,
            normalized_path=normalized_path,
        )
    try:
        _validate_alignment(unit, payload, audio)
    except Exception as exc:
        failure = {
            "version": "quran-urdu-content-repair-alignment-validation-v1",
            "passed": False,
            "at": utc_now(),
            "unit_id": unit_id,
            "audio_sha256": file_sha256(audio),
            "speech_text_sha256": text_sha256(str(unit["speech_text"])),
            "metrics": payload.get("metrics"),
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(output_dir / "FAILURE.json", failure)
        raise

    validation = {
        "version": "quran-urdu-content-repair-alignment-validation-v1",
        "passed": True,
        "at": utc_now(),
        "unit_id": unit_id,
        "audio_path": str(audio),
        "audio_sha256": file_sha256(audio),
        "speech_text_sha256": text_sha256(str(unit["speech_text"])),
        "metrics": payload["metrics"],
        "expected_refs": list(unit["refs"]),
        "actual_refs": [span["ref"] for span in payload["spans"]],
        "normalized_alignment": str(normalized_path),
        "normalized_alignment_sha256": file_sha256(normalized_path),
    }
    if generation_path is not None:
        validation["component_boundary_evidence"] = payload["component_boundary_evidence"]
    atomic_json(validation_path, validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--units-path", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generation-path", type=Path)
    args = parser.parse_args()
    result = validate(
        audio=args.audio,
        units_path=args.units_path,
        unit_id=args.unit_id,
        output_dir=args.output_dir,
        generation_path=args.generation_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
