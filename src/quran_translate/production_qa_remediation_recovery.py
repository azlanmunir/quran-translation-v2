"""Recover QA-remediation verification from Gemini schema incompatibility."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_DB_PATH
from .critic_v2 import CRITIC_SYSTEM
from .db import connect, utc_now
from .production_clients import GeminiSynchronousClient, ProviderError
from .production_packets import atomic_json
from .production_qa_remediation import (
    GEMINI_ATTEMPTS,
    MANIFEST_FILE,
    REMEDIATION_DIR,
    TARGETS_FILE,
    VERIFICATION_BUNDLE_SIZE,
    VERIFICATION_SCHEMA,
    _candidate_targets,
    _freeze_inputs,
    _load_run_config,
    _validate_verification,
    _verification_user,
)
from .production_runner import (
    GEMINI_MAX_TOKENS,
    GEMINI_MODEL,
    ProductionError,
    _gemini_text,
    extract_json,
    file_hash,
    load_environment,
    prepare_production,
    prompt_material,
    run_dir,
    stable_hash,
)


RECOVERY_VERSION = "qa-remediation-gemini-schema-recovery-v1"
RECOVERY_ATTEMPTS = (3, 4)
RECOVERY_MANIFEST = "GEMINI_SCHEMA_RECOVERY_MANIFEST.json"
RECOVERY_COMPLETE = "GEMINI_SCHEMA_RECOVERY_COMPLETE.json"


def google_compatible_schema(value: Any) -> Any:
    """Remove JSON Schema keywords unsupported by Gemini's response-schema API."""
    if isinstance(value, dict):
        return {
            key: google_compatible_schema(item)
            for key, item in value.items()
            if key not in {"additionalProperties", "$schema"}
        }
    if isinstance(value, list):
        return [google_compatible_schema(item) for item in value]
    return value


def _bundle_hash(system: str, targets: list[dict[str, Any]]) -> str:
    """Match the frozen v1 runner's expected artifact input hash exactly."""
    return stable_hash(
        {
            "model": GEMINI_MODEL,
            "system": system,
            "targets": targets,
            "schema": VERIFICATION_SCHEMA,
        }
    )


def _request_hash(
    system: str,
    user: str,
    response_schema: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "version": RECOVERY_VERSION,
            "model": GEMINI_MODEL,
            "system": system,
            "user": user,
            "response_schema": response_schema,
            "max_output_tokens": GEMINI_MAX_TOKENS,
            "temperature": 0,
        }
    )


def recover_verification_bundles(
    *,
    remediation: Path,
    targets: list[dict[str, Any]],
    target_bundle_numbers: list[int],
    system: str,
    client: GeminiSynchronousClient,
    delay_seconds: float = 0.5,
) -> dict[str, Any]:
    """Write only missing verifier bundles using a provider-compatible schema."""
    bundles = [
        targets[index : index + VERIFICATION_BUNDLE_SIZE]
        for index in range(0, len(targets), VERIFICATION_BUNDLE_SIZE)
    ]
    response_schema = google_compatible_schema(VERIFICATION_SCHEMA)
    jobs = remediation / "jobs"
    verification_dir = remediation / "verification"
    jobs.mkdir(parents=True, exist_ok=True)
    verification_dir.mkdir(parents=True, exist_ok=True)
    attempted: list[dict[str, Any]] = []

    for bundle_number in target_bundle_numbers:
        if bundle_number < 1 or bundle_number > len(bundles):
            raise ProductionError(f"Invalid recovery bundle number: {bundle_number}")
        bundle = bundles[bundle_number - 1]
        artifact = verification_dir / f"bundle-{bundle_number:03d}.json"
        expected_hash = _bundle_hash(system, bundle)
        if artifact.exists():
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            if payload.get("input_hash") != expected_hash:
                raise ProductionError(f"Recovered bundle input changed: {artifact}")
            if _validate_verification(payload.get("result"), bundle) is None:
                raise ProductionError(f"Recovered bundle fails contract: {artifact}")
            continue

        result: list[dict[str, Any]] | None = None
        for attempt in RECOVERY_ATTEMPTS:
            user = _verification_user(bundle, attempt)
            request_hash = _request_hash(system, user, response_schema)
            job_path = jobs / (
                f"gemini-schema-recovery-b{bundle_number:03d}-a{attempt}.json"
            )
            submitted = False
            if job_path.exists():
                job = json.loads(job_path.read_text(encoding="utf-8"))
                if job.get("request_hash") != request_hash:
                    raise ProductionError(f"Gemini recovery job changed: {job_path}")
                row = job.get("row")
                if not isinstance(row, dict):
                    raise ProductionError(f"Gemini recovery job lacks response: {job_path}")
            else:
                try:
                    row = client.generate(
                        model=GEMINI_MODEL,
                        system=system,
                        user=user,
                        response_schema=response_schema,
                        max_output_tokens=GEMINI_MAX_TOKENS,
                    )
                except ProviderError as exc:
                    atomic_json(
                        verification_dir
                        / f"bundle-{bundle_number:03d}-recovery-attempt{attempt}-FAILED.json",
                        {"attempt": attempt, "error": str(exc)},
                    )
                    attempted.append(
                        {
                            "bundle": bundle_number,
                            "attempt": attempt,
                            "valid": False,
                            "provider_error": True,
                        }
                    )
                    continue
                submitted = True
                atomic_json(
                    job_path,
                    {
                        "provider": "google",
                        "transport": "sync-schema-recovery",
                        "attempt": attempt,
                        "bundle": bundle_number,
                        "request_hash": request_hash,
                        "completed_at": utc_now(),
                        "row": row,
                    },
                )

            try:
                raw, usage = _gemini_text(row)
                result = _validate_verification(extract_json(raw), bundle)
            except (ProviderError, json.JSONDecodeError, TypeError, ValueError):
                result = None
                raw = json.dumps(row, ensure_ascii=False)
                usage = {}
            if result is None:
                atomic_json(
                    verification_dir
                    / f"bundle-{bundle_number:03d}-recovery-attempt{attempt}-FAILED.json",
                    {"attempt": attempt, "raw": raw, "usage": usage},
                )
                attempted.append(
                    {
                        "bundle": bundle_number,
                        "attempt": attempt,
                        "valid": False,
                        "provider_error": False,
                    }
                )
                continue

            atomic_json(
                artifact,
                {
                    "input_hash": expected_hash,
                    "model": GEMINI_MODEL,
                    "transport": "sync-schema-recovery",
                    "attempt": attempt,
                    "usage": usage,
                    "result": result,
                    "raw": raw,
                    "recovery": {
                        "version": RECOVERY_VERSION,
                        "job": job_path.name,
                        "provider_schema_sha256": stable_hash(response_schema),
                    },
                },
            )
            attempted.append(
                {
                    "bundle": bundle_number,
                    "attempt": attempt,
                    "valid": True,
                    "provider_error": False,
                }
            )
            if submitted and delay_seconds:
                time.sleep(delay_seconds)
            break

        if result is None:
            raise ProductionError(
                f"Gemini schema recovery failed twice for bundle {bundle_number}"
            )

    pending = [
        number
        for number in target_bundle_numbers
        if not (verification_dir / f"bundle-{number:03d}.json").exists()
    ]
    if pending:
        raise ProductionError(
            "Gemini schema recovery left pending bundles: "
            + ", ".join(map(str, pending))
        )
    return {
        "version": RECOVERY_VERSION,
        "target_count": len(target_bundle_numbers),
        "recovered_count": len(target_bundle_numbers),
        "pending": pending,
        "attempts": attempted,
    }


def _freeze_recovery(
    remediation: Path,
    target_bundle_numbers: list[int],
) -> None:
    path = remediation / RECOVERY_MANIFEST
    response_schema = google_compatible_schema(VERIFICATION_SCHEMA)
    record = {
        "version": RECOVERY_VERSION,
        "qa_remediation_manifest_sha256": file_hash(remediation / MANIFEST_FILE),
        "targets_sha256": file_hash(remediation / TARGETS_FILE),
        "adjudication_sha256": file_hash(remediation / "OPUS_ADJUDICATION.json"),
        "recovery_code_sha256": file_hash(Path(__file__)),
        "model": GEMINI_MODEL,
        "max_output_tokens": GEMINI_MAX_TOKENS,
        "attempts": list(RECOVERY_ATTEMPTS),
        "original_attempts": list(GEMINI_ATTEMPTS),
        "original_schema_sha256": stable_hash(VERIFICATION_SCHEMA),
        "provider_schema_sha256": stable_hash(response_schema),
        "target_bundles": target_bundle_numbers,
        "original_failure_sha256": {
            path.name: file_hash(path)
            for attempt in GEMINI_ATTEMPTS
            for path in sorted(
                remediation.glob(
                    f"verification/bundle-*-attempt{attempt}-FAILED.json"
                )
            )
        },
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ProductionError("Gemini schema recovery manifest changed")
    else:
        atomic_json(path, record)


def _critic_system() -> str:
    prompt, ledger_md, ledger_json = prompt_material()
    return (
        f"{CRITIC_SYSTEM}\n\n"
        "For this remediation audit, use the supplied string `ref` rather than an "
        "integer ayah field. Return the ref-based registered schema exactly.\n\n"
        f"=== PROJECT TRANSLATION POLICY ===\n{prompt}\n\n"
        f"=== MODEL-FACING SENSE LEDGER ===\n{ledger_md}\n\n"
        f"=== STRUCTURED SENSE RECORDS ===\n{ledger_json}"
    )


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
        prepared_base, _, _ = prepare_production(conn, config)
    remediation = prepared_base / REMEDIATION_DIR
    frozen_manifest = json.loads((remediation / MANIFEST_FILE).read_text(encoding="utf-8"))
    if frozen_manifest.get("remediation_code_sha256") != file_hash(
        Path(__file__).with_name("production_qa_remediation.py")
    ):
        raise ProductionError("Frozen QA remediation source no longer matches its manifest")
    targets = json.loads((remediation / TARGETS_FILE).read_text(encoding="utf-8"))["targets"]
    _freeze_inputs(prepared_base, targets)
    adjudication = json.loads(
        (remediation / "OPUS_ADJUDICATION.json").read_text(encoding="utf-8")
    )["result"]
    candidates = _candidate_targets(targets, adjudication)
    bundle_count = (len(candidates) + VERIFICATION_BUNDLE_SIZE - 1) // VERIFICATION_BUNDLE_SIZE
    target_bundle_numbers = list(range(1, bundle_count + 1))
    _freeze_recovery(remediation, target_bundle_numbers)

    complete_path = remediation / RECOVERY_COMPLETE
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        expected_manifest_hash = file_hash(remediation / RECOVERY_MANIFEST)
        if (
            complete.get("version") != RECOVERY_VERSION
            or complete.get("recovery_manifest_sha256") != expected_manifest_hash
            or any(
                file_hash(
                    remediation / "verification" / f"bundle-{number:03d}.json"
                )
                != complete.get("artifacts", {}).get(f"{number:03d}")
                for number in target_bundle_numbers
            )
        ):
            raise ProductionError("Gemini schema recovery completion record changed")
        print(json.dumps(complete, ensure_ascii=False, indent=2))
        return

    load_environment()
    client = GeminiSynchronousClient(os.environ.get("GOOGLE_API_KEY", ""))
    result = recover_verification_bundles(
        remediation=remediation,
        targets=candidates,
        target_bundle_numbers=target_bundle_numbers,
        system=_critic_system(),
        client=client,
    )
    complete = {
        **result,
        "recovery_manifest_sha256": file_hash(remediation / RECOVERY_MANIFEST),
        "artifacts": {
            f"{number:03d}": file_hash(
                remediation / "verification" / f"bundle-{number:03d}.json"
            )
            for number in target_bundle_numbers
        },
    }
    atomic_json(complete_path, complete)
    print(json.dumps(complete, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
