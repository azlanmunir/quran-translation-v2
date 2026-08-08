"""Non-destructive recovery for exhausted post-revision verification contracts.

The production runner preserves provider checkpoints, including responses that are
valid provider calls but invalid under the strict critic contract. Resuming the
runner therefore cannot repair an exhausted verification unit. This command makes
separately named fresh attempts for only those units, preserves every prior job and
failure record, and writes ``verification.json`` only after the production validator
accepts the response.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .critic_v2 import CRITIC_JSON_SCHEMA, CRITIC_SYSTEM
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
    _reader_after_first_revision,
    _stage_input_hash,
    _unit_packet,
    artifact_path,
    critic_assignment,
    extract_json,
    file_hash,
    load_artifact,
    load_environment,
    prepare_production,
    prompt_material,
    run_dir,
    shared_inputs,
    stable_hash,
    unit_arabic,
    unit_dir,
    validate_critic_response,
)


RECOVERY_VERSION = "verification-contract-recovery-v1"
RECOVERY_ATTEMPTS = (3, 4)
RECOVERY_MANIFEST = "VERIFICATION_RECOVERY_MANIFEST.json"
RECOVERY_COMPLETE = "VERIFICATION_RECOVERY_COMPLETE.json"


def _recovery_instruction(
    unit: ProductionUnit,
    verses: dict[tuple[int, int], str],
    attempt: int,
) -> str:
    exact_arabic = [
        {"ayah": ayah, "arabic": verses[(unit.surah, ayah)]}
        for ayah in unit.expected_ayahs
    ]
    return (
        f"\n\n=== VERIFICATION CONTRACT RECOVERY ATTEMPT {attempt} ===\n"
        "The earlier response failed the interface contract, not necessarily the "
        "fidelity audit. Return one top-level JSON array with exactly one row per "
        "requested ayah, in the supplied order. Keep explanations concise so every "
        "row is completed. For every finding, arabic_ground must be copied as an "
        "exact contiguous substring, with identical Unicode codepoints, from the "
        "matching arabic value below. Never retype it from memory, normalize its "
        "spelling, combine separate spans, or use an ellipsis. The where field must "
        "likewise be an exact contiguous quote from that ayah's supplied English, "
        "except that a true omission may use <missing>. Recheck verdict consistency "
        "before returning JSON.\n\n"
        "EXACT_ARABIC_COPY_SOURCE:\n"
        + json.dumps(exact_arabic, ensure_ascii=False, indent=2)
    )


def _request_hash(system: str, user: str) -> str:
    return stable_hash(
        {
            "model": GEMINI_MODEL,
            "system": system,
            "user": user,
            "response_schema": CRITIC_JSON_SCHEMA,
            "max_output_tokens": GEMINI_MAX_TOKENS,
            "temperature": 0,
        }
    )


def recover_verification_units(
    *,
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    shared_by_unit: dict[str, str],
    critic_system: str,
    client: GeminiSynchronousClient,
    delay_seconds: float = 0.5,
) -> dict[str, Any]:
    """Issue fresh, checkpointed requests only for missing verification artifacts."""
    attempted: list[dict[str, Any]] = []
    for attempt in RECOVERY_ATTEMPTS:
        for unit in units:
            reader = _reader_after_first_revision(base, unit)
            original_user = (
                critic_assignment(shared_by_unit[unit.unit_id], reader)
                + "\n\nThis is a post-revision verification pass. Look especially for "
                "defects introduced during revision. Do not assume the earlier critic "
                "or reviser was correct."
            )
            hash_payload = {
                "system": critic_system,
                "reader": reader,
                "shared": shared_by_unit[unit.unit_id],
                "mode": "post_revision_verification",
            }
            input_hash = _stage_input_hash("verification", unit, hash_payload)
            validator = lambda data, unit=unit, reader=reader: validate_critic_response(
                data,
                unit.expected_ayahs,
                reader,
                unit_arabic(verses, unit),
            )
            if load_artifact(
                artifact_path(base, unit, "verification"), input_hash, validator
            ):
                continue

            recovery_user = original_user + _recovery_instruction(unit, verses, attempt)
            request_hash = _request_hash(critic_system, recovery_user)
            job_path = (
                base
                / "jobs"
                / f"verification-sync-recovery-a{attempt}-{unit.unit_id}.json"
            )
            row: dict[str, Any]
            submitted = False
            if job_path.exists():
                job = json.loads(job_path.read_text(encoding="utf-8"))
                if job.get("request_hash") != request_hash:
                    raise ProductionError(
                        f"Verification recovery provider job mismatch: {job_path}"
                    )
                row = job.get("row")
                if not isinstance(row, dict):
                    raise ProductionError(f"Recovery job lacks response: {job_path}")
            else:
                try:
                    row = client.generate(
                        model=GEMINI_MODEL,
                        system=critic_system,
                        user=recovery_user,
                        response_schema=CRITIC_JSON_SCHEMA,
                        max_output_tokens=GEMINI_MAX_TOKENS,
                    )
                except ProviderError as exc:
                    atomic_json(
                        unit_dir(base, unit)
                        / f"verification-recovery-attempt{attempt}-FAILED.json",
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
                        "stage": "verification",
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
                    / f"verification-recovery-attempt{attempt}-FAILED.json",
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
                artifact_path(base, unit, "verification"),
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
        if not artifact_path(base, unit, "verification").exists()
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
            "Verification recovery exhausted fresh attempts for: " + ", ".join(pending)
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
            raise ProductionError("Verification recovery manifest has invalid targets")
        return [by_id[str(unit_id)] for unit_id in target_ids]
    targets = [
        unit
        for unit in units
        if artifact_path(base, unit, "revision").exists()
        and not artifact_path(base, unit, "verification").exists()
        and (
            unit_dir(base, unit) / "verification-attempt2-FAILED.json"
        ).exists()
    ]
    if not targets:
        raise ProductionError("No exhausted verification units require recovery")
    return targets


def _freeze_recovery(base: Path, units: list[ProductionUnit]) -> None:
    path = base / RECOVERY_MANIFEST
    record = {
        "version": RECOVERY_VERSION,
        "production_manifest_sha256": file_hash(base / "manifest.json"),
        "recovery_code_sha256": file_hash(Path(__file__)),
        "model": GEMINI_MODEL,
        "max_output_tokens": GEMINI_MAX_TOKENS,
        "attempts": list(RECOVERY_ATTEMPTS),
        "response_schema_sha256": stable_hash(CRITIC_JSON_SCHEMA),
        "target_units": [unit.unit_id for unit in units],
        "target_inputs": {
            unit.unit_id: {
                "draft_sha256": file_hash(artifact_path(base, unit, "draft")),
                "critic_sha256": file_hash(artifact_path(base, unit, "critic")),
                "revision_sha256": file_hash(artifact_path(base, unit, "revision")),
                "evidence_sha256": file_hash(base / "evidence" / f"{unit.unit_id}.md"),
                "attempt2_failure_sha256": file_hash(
                    unit_dir(base, unit) / "verification-attempt2-FAILED.json"
                ),
            }
            for unit in units
        },
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ProductionError("Verification recovery manifest changed")
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
        _freeze_recovery(prepared_base, targets)
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

    load_environment()
    prompt, ledger_md, ledger_json = prompt_material()
    critic_system = (
        f"{CRITIC_SYSTEM}\n\n=== PROJECT TRANSLATION POLICY ===\n{prompt}\n\n"
        f"=== MODEL-FACING SENSE LEDGER ===\n{ledger_md}\n\n"
        f"=== STRUCTURED SENSE RECORDS ===\n{ledger_json}"
    )
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
    result = recover_verification_units(
        base=prepared_base,
        units=targets,
        verses=verses,
        shared_by_unit=shared_by_unit,
        critic_system=critic_system,
        client=client,
    )
    complete = {
        **result,
        "recovery_manifest_sha256": file_hash(prepared_base / RECOVERY_MANIFEST),
        "artifacts": {
            unit.unit_id: file_hash(
                artifact_path(prepared_base, unit, "verification")
            )
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
            raise ProductionError("Verification recovery completion record changed")
        complete = current
    else:
        atomic_json(complete_path, complete)
    print(json.dumps(complete, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
