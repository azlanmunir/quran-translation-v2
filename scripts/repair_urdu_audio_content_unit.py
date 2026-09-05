#!/usr/bin/env python3
"""Regenerate one Urdu audiobook master unit ayah-by-ayah after content QA."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from quran_translate.config import file_sha256, load_dotenv, text_sha256
from quran_translate.db import utc_now
from quran_translate.production_packets import atomic_json
from quran_translate.state_safety import exclusive_lock
from quran_translate.urdu_audio_pilot import (
    AUDIO_TOKENS_PER_SECOND,
    DIRECTION,
    MODEL_ID,
    STANDARD_AUDIO_USD_PER_MILLION,
    STANDARD_INPUT_USD_PER_MILLION,
    VOICE_ID,
    _gemini_audio,
)
from quran_translate.urdu_audio_production import (
    HARD_ESTIMATED_BATCH_COST_USD,
    PRODUCTION_ROOT,
    RELEASE_AUDIO_ROOT,
    _concat_wav,
    _normalize,
    _probe,
    _save_state,
    _write_wav,
    assemble,
    status,
)


class ContentRepairError(RuntimeError):
    """Raised when a content repair cannot preserve the production contract."""


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContentRepairError(f"Expected JSON object: {path}")
    return value


def _read_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ContentRepairError(f"Expected JSON list: {path}")
    return [dict(item) for item in value]


def _repair_root(unit_id: str) -> Path:
    return PRODUCTION_ROOT / "content-repairs" / unit_id / "attempt-0001"


def _standard_cost(prompt_tokens: int, output_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * STANDARD_INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * STANDARD_AUDIO_USD_PER_MILLION
    )


def _atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.content-repair.tmp")
    shutil.copy2(source, temp)
    os.replace(temp, destination)


def _preserve_link(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise ContentRepairError(f"Preserved artifact drift: {destination}")
        return
    os.link(source, destination)



MAX_REPAIR_OUTPUT_TOKENS = 8192


def _repair_spend() -> float:
    state = _read_object(PRODUCTION_ROOT / "RUN.json")
    recorded = sum(float(job.get("actual_batch_cost_usd", 0)) for job in state["jobs"])
    adopted = sum(
        float(repair.get("repair_standard_cost_usd", 0))
        for job in state["jobs"] for repair in job.get("content_integrity_repairs", [])
    )
    if not all(math.isfinite(value) and value >= 0 for value in (recorded, adopted)):
        raise ContentRepairError("Invalid recorded audio spend")
    generated = 0.0
    for directory in (PRODUCTION_ROOT / "content-repairs").glob("*/attempt-*/ayahs/*"):
        call = directory / "CALL.json"
        complete = directory / "COMPLETE.json"
        if call.exists():
            receipt = _read_object(call)
            if receipt.get("state") != "accounted":
                raise ContentRepairError(f"Uncertain paid call requires reconciliation: {call}")
            cost = float(receipt["actual_standard_cost_usd"])
        elif complete.exists():
            cost = float(_read_object(complete)["actual_standard_cost_usd"])
        elif any(directory.iterdir()):
            raise ContentRepairError(f"Historical repair spend is unresolved: {directory}")
        else:
            continue
        if not math.isfinite(cost) or cost < 0:
            raise ContentRepairError("Invalid repair cost evidence")
        generated += cost
    return max(recorded, recorded - adopted + generated)


def _reserve_repair_call(directory: Path, line: str) -> Path:
    call = directory / "CALL.json"
    if call.exists():
        raise ContentRepairError(f"Existing paid attempt must not be replayed: {call}")
    reservation = _standard_cost(
        len((DIRECTION + line).encode("utf-8")) + 128, MAX_REPAIR_OUTPUT_TOKENS
    ) * 1.25
    if _repair_spend() + reservation > HARD_ESTIMATED_BATCH_COST_USD:
        raise ContentRepairError("Next ayah would exceed the cumulative audio-repair ceiling")
    atomic_json(call, {
        "state": "submitting", "at": utc_now(), "reserved_usd": reservation,
        "speech_text_sha256": text_sha256(line),
    })
    return call


def _account_repair_call(call: Path, response: Any) -> tuple[int, int, float]:
    usage = getattr(response, "usage_metadata", None)
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    output = int(getattr(usage, "candidates_token_count", 0) or 0)
    if prompt < 0 or output <= 0:
        raise ContentRepairError(f"Provider usage missing; reservation retained: {call}")
    cost = _standard_cost(prompt, output)
    receipt = _read_object(call)
    receipt.update({
        "state": "accounted", "actual_standard_cost_usd": cost,
        "provider_usage": {"prompt_tokens": prompt, "output_tokens": output},
    })
    atomic_json(call, receipt)
    if _repair_spend() > HARD_ESTIMATED_BATCH_COST_USD:
        raise ContentRepairError("Actual audio spend reached the ceiling; no further calls permitted")
    return prompt, output, cost


def generate(unit_id: str) -> dict[str, Any]:
    with exclusive_lock(PRODUCTION_ROOT / "content-repairs" / ".budget.lock"):
        return _generate(unit_id)

def _generate(unit_id: str) -> dict[str, Any]:
    state = _read_object(PRODUCTION_ROOT / "RUN.json")
    units = {item["unit_id"]: item for item in _read_list(PRODUCTION_ROOT / "UNITS.json")}
    jobs = {item["unit_id"]: item for item in state["jobs"]}
    unit = units.get(unit_id)
    job = jobs.get(unit_id)
    if not unit or not job:
        raise ContentRepairError(f"Unknown unit: {unit_id}")
    if job.get("status") != "complete":
        raise ContentRepairError(f"Content repair requires a complete unit: {unit_id}")

    refs = [str(ref) for ref in unit["refs"]]
    lines = str(unit["speech_text"]).splitlines()
    if len(refs) != len(lines) or not refs:
        raise ContentRepairError(f"Ayah-line alignment failed: {unit_id}")

    root = _repair_root(unit_id)
    complete_path = root / "GENERATION_COMPLETE.json"
    if complete_path.is_file():
        complete = _read_object(complete_path)
        candidate = Path(str(complete["candidate_mp3"]))
        if candidate.is_file() and file_sha256(candidate) == complete["candidate_sha256"]:
            return complete
        raise ContentRepairError("Completed content-repair candidate changed")

    current_spend = _repair_spend()
    old_duration = float(job["probe"]["duration_seconds"])
    estimated_repair = _standard_cost(
        math.ceil(sum(len(DIRECTION) + len(line) for line in lines) / 4),
        math.ceil(old_duration * 1.35 * AUDIO_TOKENS_PER_SECOND),
    )
    if current_spend + estimated_repair * 1.25 > HARD_ESTIMATED_BATCH_COST_USD:
        raise ContentRepairError(
            f"Repair estimate would exceed the ${HARD_ESTIMATED_BATCH_COST_USD:.2f} guard"
        )

    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ContentRepairError("Missing GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    records: list[dict[str, Any]] = []

    for index, (ref, line) in enumerate(zip(refs, lines), start=1):
        ayah_root = root / "ayahs" / f"{index:02d}-{ref.replace(':', '-') }"
        raw = ayah_root / "raw.wav"
        normalized = ayah_root / "normalized.mp3"
        sidecar = ayah_root / "COMPLETE.json"
        if sidecar.is_file():
            record = _read_object(sidecar)
            if (
                raw.is_file()
                and normalized.is_file()
                and file_sha256(raw) == record["raw_sha256"]
                and file_sha256(normalized) == record["normalized_sha256"]
                and record["speech_text_sha256"] == text_sha256(line)
            ):
                records.append(record)
                continue
            raise ContentRepairError(f"Completed ayah repair artifact changed: {ref}")

        ayah_root.mkdir(parents=True, exist_ok=True)
        call = _reserve_repair_call(ayah_root, line)
        started = time.monotonic()
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=f"{DIRECTION}\n\n<text>{line}</text>",
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_REPAIR_OUTPUT_TOKENS,
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        language_code="ur",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=VOICE_ID
                            )
                        ),
                    ),
                ),
            )
            prompt_tokens, output_tokens, actual_cost = _account_repair_call(call, response)
            _write_wav(raw, _gemini_audio(response))
            _normalize(raw, normalized)
            probe = _probe(normalized)
            ratio = float(probe["duration_seconds"]) / max(1, len(line))
            if (
                probe["codec"] != "mp3"
                or probe["sample_rate"] != 44_100
                or probe["channels"] != 1
                or not 0.02 <= ratio <= 0.35
            ):
                raise ContentRepairError(f"Ayah audio contract failed at {ref}: {probe}")
            record = {
                "version": "quran-urdu-content-repair-ayah-v1",
                "at": utc_now(),
                "unit_id": unit_id,
                "ref": ref,
                "speech_text_sha256": text_sha256(line),
                "model_id": MODEL_ID,
                "voice_id": VOICE_ID,
                "raw_path": str(raw),
                "raw_sha256": file_sha256(raw),
                "normalized_path": str(normalized),
                "normalized_sha256": file_sha256(normalized),
                "probe": probe,
                "provider_usage": {
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                },
                "actual_standard_cost_usd": round(
                    actual_cost, 6
                ),
                "latency_seconds": round(time.monotonic() - started, 3),
            }
            atomic_json(sidecar, record)
            records.append(record)
        except Exception as exc:
            atomic_json(
                ayah_root / "FAILURE.json",
                {"at": utc_now(), "ref": ref, "error": f"{type(exc).__name__}: {exc}"},
            )
            raise

    candidate_raw = root / "candidate.wav"
    candidate_mp3 = root / "candidate.mp3"
    _concat_wav([Path(record["raw_path"]) for record in records], candidate_raw)
    _normalize(candidate_raw, candidate_mp3)
    probe = _probe(candidate_mp3)
    ratio = float(probe["duration_seconds"]) / max(1, int(unit["speech_characters"]))
    if (
        probe["codec"] != "mp3"
        or probe["sample_rate"] != 44_100
        or probe["channels"] != 1
        or not 0.04 <= ratio <= 0.22
    ):
        raise ContentRepairError(f"Candidate audio contract failed: {probe}")

    total_cost = round(sum(float(item["actual_standard_cost_usd"]) for item in records), 6)
    if _repair_spend() > HARD_ESTIMATED_BATCH_COST_USD:
        raise ContentRepairError("Actual repair spend exceeds the production guard")
    complete = {
        "version": "quran-urdu-content-repair-generation-v1",
        "status": "ready_for_alignment_validation",
        "at": utc_now(),
        "unit_id": unit_id,
        "unit_index": unit["unit_index"],
        "refs": refs,
        "speech_text_sha256": unit["speech_text_sha256"],
        "candidate_raw": str(candidate_raw),
        "candidate_raw_sha256": file_sha256(candidate_raw),
        "candidate_mp3": str(candidate_mp3),
        "candidate_sha256": file_sha256(candidate_mp3),
        "probe": probe,
        "ayah_records": records,
        "actual_standard_cost_usd": total_cost,
        "recorded_run_spend_before_repair_usd": round(current_spend, 6),
        "recorded_run_spend_after_repair_usd": round(_repair_spend(), 6),
    }
    atomic_json(complete_path, complete)
    return complete


def commit(unit_id: str, validation_path: Path, omitted_refs: list[str]) -> dict[str, Any]:
    with exclusive_lock(PRODUCTION_ROOT / "content-repairs" / ".budget.lock"):
        return _commit(unit_id, validation_path, omitted_refs)


def _commit(unit_id: str, validation_path: Path, omitted_refs: list[str]) -> dict[str, Any]:
    root = _repair_root(unit_id)
    generation = _read_object(root / "GENERATION_COMPLETE.json")
    validation = _read_object(validation_path)
    candidate_raw = Path(str(generation["candidate_raw"]))
    candidate_mp3 = Path(str(generation["candidate_mp3"]))
    if generation["status"] != "ready_for_alignment_validation":
        raise ContentRepairError("Candidate is not ready for commit")
    if (
        validation.get("passed") is not True
        or validation.get("unit_id") != unit_id
        or validation.get("audio_sha256") != generation["candidate_sha256"]
        or validation.get("speech_text_sha256") != generation["speech_text_sha256"]
    ):
        raise ContentRepairError("Alignment validation does not authorize this candidate")
    expected_refs = set(validation.get("expected_refs", []))
    if not omitted_refs or any(ref not in expected_refs for ref in omitted_refs):
        raise ContentRepairError("Omitted refs must be present in the validated unit")

    state = _read_object(PRODUCTION_ROOT / "RUN.json")
    units = {item["unit_id"]: item for item in _read_list(PRODUCTION_ROOT / "UNITS.json")}
    jobs = {item["unit_id"]: item for item in state["jobs"]}
    unit = units[unit_id]
    job = jobs[unit_id]
    canonical_raw = Path(str(job["raw_path"]))
    canonical_mp3 = Path(str(job["normalized_path"]))
    if (
        file_sha256(canonical_raw) != job["raw_sha256"]
        or file_sha256(canonical_mp3) != job["normalized_sha256"]
    ):
        raise ContentRepairError("Canonical master changed before content repair commit")

    preserved = root / "preserved"
    _preserve_link(canonical_raw, preserved / "master" / canonical_raw.name)
    _preserve_link(canonical_mp3, preserved / "master" / canonical_mp3.name)
    for marker_name in ("SYNTHESIS_COMPLETE.json", "PRODUCTION_COMPLETE.json"):
        marker = PRODUCTION_ROOT / marker_name
        _preserve_link(marker, preserved / "production" / marker_name)

    assembly_path = PRODUCTION_ROOT / "ASSEMBLY_STATE.json"
    assembly_state = _read_object(assembly_path)
    affected = [
        record
        for record in assembly_state["outputs"]
        if unit_id in record.get("source_units", [])
    ]
    if len(affected) != 3:
        raise ContentRepairError(f"Expected 3 affected release outputs, found {len(affected)}")
    for record in affected:
        path = Path(str(record["path"]))
        relative = path.relative_to(RELEASE_AUDIO_ROOT)
        _preserve_link(path, preserved / "release" / relative)
    for sidecar_name in ("RELEASE_MANIFEST.json", "QA_REPORT.json", "SHA256SUMS.txt"):
        sidecar = RELEASE_AUDIO_ROOT / sidecar_name
        _preserve_link(sidecar, preserved / "release" / sidecar_name)

    _atomic_replace(candidate_raw, canonical_raw)
    _atomic_replace(candidate_mp3, canonical_mp3)
    previous_cost = float(job.get("actual_batch_cost_usd", 0))
    repair_cost = float(generation["actual_standard_cost_usd"])
    repair_record = {
        "version": "quran-urdu-content-integrity-repair-v1",
        "committed_at": utc_now(),
        "reason": "forced-alignment detected omitted narration",
        "omitted_refs_detected": omitted_refs,
        "candidate_generation": str(root / "GENERATION_COMPLETE.json"),
        "alignment_validation": str(validation_path),
        "preserved_artifacts": str(preserved),
        "previous_raw_sha256": job["raw_sha256"],
        "previous_normalized_sha256": job["normalized_sha256"],
        "repair_standard_cost_usd": repair_cost,
    }
    job.setdefault("content_integrity_repairs", []).append(repair_record)
    job.update(
        {
            "status": "complete",
            "source": "ayah_by_ayah_content_integrity_repair",
            "raw_sha256": file_sha256(canonical_raw),
            "normalized_sha256": file_sha256(canonical_mp3),
            "probe": _probe(canonical_mp3),
            "historical_batch_cost_usd": previous_cost,
            "actual_repair_standard_cost_usd": repair_cost,
            "actual_batch_cost_usd": round(previous_cost + repair_cost, 6),
            "last_error": None,
        }
    )
    state["status"] = "synthesis_complete"
    state["content_integrity_repaired_at"] = utc_now()
    state.pop("completed_at", None)
    _save_state(PRODUCTION_ROOT, state)
    atomic_json(PRODUCTION_ROOT / "SYNTHESIS_COMPLETE.json", status(PRODUCTION_ROOT))
    (PRODUCTION_ROOT / "PRODUCTION_COMPLETE.json").unlink(missing_ok=True)

    affected_paths = {record["path"] for record in affected}
    assembly_state["outputs"] = [
        record for record in assembly_state["outputs"] if record["path"] not in affected_paths
    ]
    assembly_state["status"] = "assembling"
    assembly_state["content_integrity_repair"] = repair_record
    assembly_state.pop("completed_at", None)
    atomic_json(assembly_path, assembly_state)
    for record in affected:
        Path(str(record["path"])).unlink(missing_ok=True)
    for sidecar_name in ("RELEASE_MANIFEST.json", "QA_REPORT.json", "SHA256SUMS.txt"):
        (RELEASE_AUDIO_ROOT / sidecar_name).unlink(missing_ok=True)

    manifest = assemble(PRODUCTION_ROOT, RELEASE_AUDIO_ROOT, decode_check=True)
    complete = {
        "version": "quran-urdu-content-repair-complete-v1",
        "status": "complete",
        "at": utc_now(),
        "unit_id": unit_id,
        "validation": validation,
        "new_master_sha256": file_sha256(canonical_mp3),
        "repair_standard_cost_usd": repair_cost,
        "recorded_run_spend_usd": manifest["totals"]["recorded_batch_spend_usd"],
        "rebuilt_outputs": [record["path"] for record in affected],
        "release_manifest": str(RELEASE_AUDIO_ROOT / "RELEASE_MANIFEST.json"),
    }
    atomic_json(root / "CONTENT_REPAIR_COMPLETE.json", complete)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "commit"))
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--validation-path", type=Path)
    parser.add_argument("--omitted-refs")
    args = parser.parse_args()
    if args.action == "generate":
        result = generate(args.unit_id)
    else:
        if args.validation_path is None:
            parser.error("commit requires --validation-path")
        if not args.omitted_refs:
            parser.error("commit requires --omitted-refs")
        omitted_refs = [ref.strip() for ref in args.omitted_refs.split(",") if ref.strip()]
        result = commit(args.unit_id, args.validation_path, omitted_refs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
