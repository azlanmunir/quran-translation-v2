"""Non-destructive recovery for exhausted spoken-English review contracts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .db import connect, utc_now
from .production_clients import GeminiSynchronousClient, ProviderError
from .production_packets import ProductionUnit, atomic_json
from .production_runner import (
    DEFAULT_DB_PATH,
    GEMINI_MAX_TOKENS,
    GEMINI_MODEL,
    ProductionConfig,
    ProductionError,
    _gemini_text,
    _reader_with_overrides,
    _stage_input_hash,
    _unit_packet,
    artifact_path,
    extract_json,
    file_hash,
    load_artifact,
    load_environment,
    prepare_production,
    prompt_material,
    reader_text,
    run_dir,
    shared_inputs,
    stable_hash,
    unit_dir,
)
from .spoken_english_v1 import (
    SPOKEN_ENGLISH_SCHEMA,
    SPOKEN_ENGLISH_SYSTEM,
    validate_spoken_english,
)


RECOVERY_VERSION = "spoken-contract-recovery-v1"
RECOVERY_ATTEMPTS = (3, 4)
RECOVERY_MANIFEST = "SPOKEN_RECOVERY_MANIFEST.json"
RECOVERY_COMPLETE = "SPOKEN_RECOVERY_COMPLETE.json"


def _request_hash(system: str, user: str) -> str:
    return stable_hash(
        {
            "model": GEMINI_MODEL,
            "system": system,
            "user": user,
            "response_schema": SPOKEN_ENGLISH_SCHEMA,
            "max_output_tokens": GEMINI_MAX_TOKENS,
            "temperature": 0,
        }
    )


def _compact_assignment(
    unit: ProductionUnit,
    verses: dict[tuple[int, int], str],
    reader: list[dict[str, Any]],
    attempt: int,
) -> str:
    paired = [
        {
            "ayah": ayah,
            "arabic": verses[(unit.surah, ayah)],
            "english": reader_text(reader)[ayah],
        }
        for ayah in unit.expected_ayahs
    ]
    return (
        f"=== SPOKEN REVIEW CONTRACT RECOVERY ATTEMPT {attempt} ===\n"
        "The earlier response exhausted its output budget before completing the JSON "
        "contract. Perform only the narrow spoken-English checks in the system "
        "instruction. Do not redo translation research or extended philological "
        "analysis. Return one compact top-level JSON array with exactly one row per "
        "ayah below, in order. A pass row is "
        "{\"ayah\":N,\"findings\":[],\"verdict\":\"pass\"}. Keep any findings "
        "concise, copy `where` exactly from the matching English, and verify complete "
        "coverage before returning. Return JSON only.\n\n"
        "=== SOURCE-GROUNDED ENGLISH TO CHECK ===\n"
        + json.dumps(paired, ensure_ascii=False, separators=(",", ":"))
    )


def recover_spoken_units(
    *,
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    readers_by_unit: dict[str, list[dict[str, Any]]],
    shared_by_unit: dict[str, str],
    spoken_system: str,
    client: GeminiSynchronousClient,
    delay_seconds: float = 0.5,
) -> dict[str, Any]:
    """Make fresh, checkpointed requests only for missing spoken-review units."""
    attempted: list[dict[str, Any]] = []
    for attempt in RECOVERY_ATTEMPTS:
        for unit in units:
            reader = readers_by_unit[unit.unit_id]
            hash_payload = {
                "system": spoken_system,
                "reader": reader,
                "shared": shared_by_unit[unit.unit_id],
                "mode": "advisory_spoken_english_review",
            }
            input_hash = _stage_input_hash("spoken", unit, hash_payload)
            validator = lambda data, unit=unit, reader=reader: validate_spoken_english(
                data,
                unit.expected_ayahs,
                reader_text(reader),
            )
            if load_artifact(artifact_path(base, unit, "spoken"), input_hash, validator):
                continue

            recovery_user = _compact_assignment(unit, verses, reader, attempt)
            request_hash = _request_hash(spoken_system, recovery_user)
            job_path = (
                base / "jobs" / f"spoken-sync-recovery-a{attempt}-{unit.unit_id}.json"
            )
            row: dict[str, Any]
            submitted = False
            if job_path.exists():
                job = json.loads(job_path.read_text(encoding="utf-8"))
                if job.get("request_hash") != request_hash:
                    raise ProductionError(f"Spoken recovery job mismatch: {job_path}")
                row = job.get("row")
                if not isinstance(row, dict):
                    raise ProductionError(f"Recovery job lacks response: {job_path}")
            else:
                try:
                    row = client.generate(
                        model=GEMINI_MODEL,
                        system=spoken_system,
                        user=recovery_user,
                        response_schema=SPOKEN_ENGLISH_SCHEMA,
                        max_output_tokens=GEMINI_MAX_TOKENS,
                    )
                except ProviderError as exc:
                    atomic_json(
                        unit_dir(base, unit)
                        / f"spoken-recovery-attempt{attempt}-FAILED.json",
                        {
                            "input_hash": input_hash,
                            "attempt": attempt,
                            "error": str(exc),
                        },
                    )
                    attempted.append(
                        {"unit_id": unit.unit_id, "attempt": attempt, "valid": False}
                    )
                    continue
                submitted = True
                atomic_json(
                    job_path,
                    {
                        "provider": "google",
                        "transport": "sync-recovery",
                        "stage": "spoken",
                        "attempt": attempt,
                        "unit_id": unit.unit_id,
                        "request_hash": request_hash,
                        "state": "succeeded",
                        "completed_at": utc_now(),
                        "row": row,
                    },
                )

            try:
                raw, usage = _gemini_text(row)
                result = validator(extract_json(raw))
            except (ProviderError, json.JSONDecodeError, TypeError, ValueError):
                result = None
                raw = json.dumps(row, ensure_ascii=False)
                usage = {}
            if result is None:
                atomic_json(
                    unit_dir(base, unit)
                    / f"spoken-recovery-attempt{attempt}-FAILED.json",
                    {
                        "input_hash": input_hash,
                        "attempt": attempt,
                        "raw": raw,
                        "usage": usage,
                    },
                )
                attempted.append(
                    {"unit_id": unit.unit_id, "attempt": attempt, "valid": False}
                )
                continue

            atomic_json(
                artifact_path(base, unit, "spoken"),
                {
                    "input_hash": input_hash,
                    "model": GEMINI_MODEL,
                    "transport": "sync-recovery",
                    "attempt": attempt,
                    "usage": usage,
                    "result": result,
                    "raw": raw,
                    "recovery": {
                        "version": RECOVERY_VERSION,
                        "job": job_path.name,
                        "compact_assignment": True,
                    },
                },
            )
            attempted.append(
                {"unit_id": unit.unit_id, "attempt": attempt, "valid": True}
            )
            if submitted and delay_seconds:
                time.sleep(delay_seconds)

    pending = [
        unit.unit_id
        for unit in units
        if not artifact_path(base, unit, "spoken").exists()
    ]
    result = {
        "version": RECOVERY_VERSION,
        "target_count": len(units),
        "recovered_count": len(units) - len(pending),
        "pending": pending,
        "attempts": attempted,
    }
    if pending:
        raise ProductionError(
            "Spoken recovery exhausted fresh attempts for: " + ", ".join(pending)
        )
    return result


def _load_run_config(base: Path) -> ProductionConfig:
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ProductionError("Production manifest lacks config")
    return ProductionConfig(**config)


def _target_units(base: Path, units: list[ProductionUnit]) -> list[ProductionUnit]:
    manifest_path = base / RECOVERY_MANIFEST
    by_id = {unit.unit_id: unit for unit in units}
    if manifest_path.exists():
        recovery = json.loads(manifest_path.read_text(encoding="utf-8"))
        target_ids = recovery.get("target_units")
        if not isinstance(target_ids, list) or any(
            unit_id not in by_id for unit_id in target_ids
        ):
            raise ProductionError("Spoken recovery manifest has bad targets")
        return [by_id[str(unit_id)] for unit_id in target_ids]
    targets = [
        unit
        for unit in units
        if not artifact_path(base, unit, "spoken").exists()
        and (unit_dir(base, unit) / "spoken-attempt2-FAILED.json").exists()
    ]
    if not targets:
        raise ProductionError("No exhausted spoken-review units need recovery")
    return targets


def _freeze_recovery(
    base: Path,
    units: list[ProductionUnit],
    readers_by_unit: dict[str, list[dict[str, Any]]],
) -> None:
    path = base / RECOVERY_MANIFEST
    record = {
        "version": RECOVERY_VERSION,
        "production_manifest_sha256": file_hash(base / "manifest.json"),
        "refrains_sha256": file_hash(base / "REFRAINS.json"),
        "recovery_code_sha256": file_hash(Path(__file__)),
        "spoken_contract_sha256": file_hash(
            Path(__file__).with_name("spoken_english_v1.py")
        ),
        "model": GEMINI_MODEL,
        "max_output_tokens": GEMINI_MAX_TOKENS,
        "attempts": list(RECOVERY_ATTEMPTS),
        "response_schema_sha256": stable_hash(SPOKEN_ENGLISH_SCHEMA),
        "target_units": [unit.unit_id for unit in units],
        "target_inputs": {
            unit.unit_id: {
                "final_reader_sha256": stable_hash(readers_by_unit[unit.unit_id]),
                "evidence_sha256": file_hash(base / "evidence" / f"{unit.unit_id}.md"),
                "attempt2_failure_sha256": file_hash(
                    unit_dir(base, unit) / "spoken-attempt2-FAILED.json"
                ),
            }
            for unit in units
        },
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ProductionError("Spoken recovery manifest changed")
    else:
        atomic_json(path, record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = run_dir(args.run_id)
    config = _load_run_config(base)
    with connect(args.db) as conn:
        prepared_base, units, verses = prepare_production(conn, config)
        targets = _target_units(prepared_base, units)
        opening_bismillah = {
            int(row["surah_number"]): str(row["bismillah"])
            for row in conn.execute(
                """
                SELECT surah_number, bismillah
                FROM source_ayahs
                WHERE ayah_number = 1 AND bismillah IS NOT NULL
                """
            )
        }

    refrains = json.loads((prepared_base / "REFRAINS.json").read_text(encoding="utf-8"))
    overrides = refrains.get("overrides")
    if not isinstance(overrides, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in overrides.items()
    ):
        raise ProductionError("REFRAINS.json has invalid overrides")
    readers_by_unit = {
        unit.unit_id: _reader_with_overrides(prepared_base, unit, overrides)
        for unit in targets
    }
    _freeze_recovery(prepared_base, targets, readers_by_unit)

    load_environment()
    prompt, _ledger_md, _ledger_json = prompt_material()
    spoken_system = f"{SPOKEN_ENGLISH_SYSTEM}\n\n=== PROJECT REGISTER POLICY ===\n{prompt}"
    shared_by_unit = {
        unit.unit_id: shared_inputs(
            unit,
            verses=verses,
            packet=_unit_packet(prepared_base, unit),
            opening_bismillah=opening_bismillah.get(unit.surah),
        )
        for unit in targets
    }
    client = GeminiSynchronousClient(os.environ.get("GOOGLE_API_KEY", ""))
    result = recover_spoken_units(
        base=prepared_base,
        units=targets,
        verses=verses,
        readers_by_unit=readers_by_unit,
        shared_by_unit=shared_by_unit,
        spoken_system=spoken_system,
        client=client,
    )
    complete = {
        **result,
        "recovery_manifest_sha256": file_hash(prepared_base / RECOVERY_MANIFEST),
        "artifacts": {
            unit.unit_id: file_hash(artifact_path(prepared_base, unit, "spoken"))
            for unit in targets
        },
    }
    complete_path = prepared_base / RECOVERY_COMPLETE
    if complete_path.exists():
        current = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            current.get("version") != RECOVERY_VERSION
            or current.get("recovery_manifest_sha256")
            != complete["recovery_manifest_sha256"]
            or current.get("artifacts") != complete["artifacts"]
        ):
            raise ProductionError("Spoken completion record changed")
        complete = current
    else:
        atomic_json(complete_path, complete)
    print(json.dumps(complete, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
