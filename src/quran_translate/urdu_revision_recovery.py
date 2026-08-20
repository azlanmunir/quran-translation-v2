"""Non-destructive recovery for Anthropic-billing-blocked Urdu revisions."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import DEFAULT_DB_PATH
from .db import connect
from .production_clients import (
    AnthropicBatchClient,
    ProviderError,
    anthropic_result_text,
)
from .production_packets import ProductionUnit, atomic_json
from .urdu_costs import estimate_request_ceiling, usage_cost
from .urdu_production import (
    REVISION_MAX_OUTPUT_TOKENS,
    REVISION_MODEL,
    BudgetLedger,
    UrduProductionConfig,
    UrduProductionError,
    _load_json,
    _revision_assignment,
    _revision_request,
    _units_requiring_revision,
    artifact_path,
    prepare_production,
    unit_dir,
)
from .urdu_quality import REVISION_SCHEMA, validate_revision
from .urdu_translation_bakeoff import (
    CONTRACT_ATTEMPTS,
    extract_json,
    file_hash,
    load_environment,
    stable_hash,
)


TARGETS_NAME = "REVISION_RECOVERY_TARGETS.json"
BATCH_NAME = "REVISION_RECOVERY_BATCH.json"
COMPLETE_NAME = "REVISION_RECOVERY_COMPLETE.json"
BLOCKED_NAME = "REVISION_RECOVERY_BLOCKED.json"
BILLING_MARKER = "credit balance is too low"
RECOVERY_ATTEMPT = 2


def _revision_material(
    base: Path,
    unit: ProductionUnit,
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
) -> tuple[str, str, list[str], str]:
    system, user, finding_ids = _revision_assignment(
        base, unit, verses, bismillah
    )
    input_hash = stable_hash(
        {
            "stage": "revision",
            "unit": unit.to_dict(),
            "model": asdict(REVISION_MODEL),
            "system": system,
            "user": user,
            "schema": REVISION_SCHEMA,
            "contract_attempts": CONTRACT_ATTEMPTS,
        }
    )
    return system, user, finding_ids, input_hash


def _is_billing_failure(document: dict[str, Any]) -> bool:
    errors = " ".join(str(item) for item in document.get("errors", [])).lower()
    return (
        document.get("stage") == "revision"
        and document.get("attempt") == 1
        and not document.get("usage")
        and BILLING_MARKER in errors
    )


def _validate_targets(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
    marker: dict[str, Any],
) -> dict[str, Any]:
    targets = marker.get("targets")
    if (
        marker.get("version") != "urdu-revision-billing-recovery-targets-v1"
        or marker.get("run_id") != base.name
        or not isinstance(targets, list)
        or marker.get("target_count") != len(targets)
        or marker.get("target_hash") != stable_hash(targets)
    ):
        raise UrduProductionError("Invalid frozen Urdu revision recovery marker")
    unit_by_id = {unit.unit_id: unit for unit in units}
    for target in targets:
        unit_id = str(target.get("unit_id"))
        unit = unit_by_id.get(unit_id)
        if unit is None:
            raise UrduProductionError(
                f"Unknown frozen Urdu revision recovery unit: {unit_id}"
            )
        failure = unit_dir(base, unit) / "revision-attempt1-FAILED.json"
        if not failure.is_file() or file_hash(failure) != target.get(
            "billing_failure_sha256"
        ):
            raise UrduProductionError(
                f"Frozen Urdu revision billing evidence changed: {failure}"
            )
        if not _is_billing_failure(_load_json(failure)):
            raise UrduProductionError(
                f"Frozen Urdu revision target is not a billing failure: {failure}"
            )
        _system, _user, _finding_ids, input_hash = _revision_material(
            base, unit, verses, bismillah
        )
        if input_hash != target.get("input_hash"):
            raise UrduProductionError(
                f"Frozen Urdu revision recovery input changed: {unit_id}"
            )
    return marker


def freeze_targets(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
) -> dict[str, Any]:
    marker_path = base / TARGETS_NAME
    if marker_path.exists():
        return _validate_targets(
            base, units, verses, bismillah, _load_json(marker_path)
        )

    targets: list[dict[str, Any]] = []
    for unit in _units_requiring_revision(base, units):
        if artifact_path(base, unit, "revision").is_file():
            continue
        failure = unit_dir(base, unit) / "revision-attempt1-FAILED.json"
        if not failure.is_file():
            raise UrduProductionError(
                f"Revision recovery found missing billing evidence: {failure}"
            )
        document = _load_json(failure)
        if not _is_billing_failure(document):
            raise UrduProductionError(
                f"Revision recovery refuses a non-billing failure: {failure}"
            )
        _system, _user, _finding_ids, input_hash = _revision_material(
            base, unit, verses, bismillah
        )
        if document.get("input_hash") != input_hash:
            raise UrduProductionError(
                f"Revision billing failure input changed: {failure}"
            )
        targets.append(
            {
                "unit_id": unit.unit_id,
                "input_hash": input_hash,
                "billing_failure": failure.name,
                "billing_failure_sha256": file_hash(failure),
            }
        )
    if not targets:
        raise UrduProductionError("No billing-blocked Urdu revisions need recovery")
    marker = {
        "version": "urdu-revision-billing-recovery-targets-v1",
        "run_id": base.name,
        "reason": "anthropic_credit_balance_too_low",
        "targets": targets,
        "target_count": len(targets),
    }
    marker["target_hash"] = stable_hash(targets)
    atomic_json(marker_path, marker)
    return _validate_targets(base, units, verses, bismillah, marker)


def recovery_status(base: Path, marker: dict[str, Any]) -> dict[str, Any]:
    completed = sum(
        (base / "units" / str(target["unit_id"]) / "revision.json").is_file()
        for target in marker["targets"]
    )
    failed = sum(
        (
            base
            / "units"
            / str(target["unit_id"])
            / f"revision-recovery-attempt{RECOVERY_ATTEMPT}-FAILED.json"
        ).is_file()
        for target in marker["targets"]
    )
    return {
        "version": "urdu-revision-billing-recovery-status-v1",
        "run_id": base.name,
        "total": int(marker["target_count"]),
        "complete": completed,
        "pending": int(marker["target_count"]) - completed - failed,
        "failed": failed,
        "provider_blocked": (base / BLOCKED_NAME).is_file()
        and not (base / COMPLETE_NAME).is_file(),
        "recovery_complete": (base / COMPLETE_NAME).is_file(),
    }


def _write_complete(base: Path, marker: dict[str, Any]) -> dict[str, Any]:
    status = recovery_status(base, marker)
    if status["complete"] != status["total"] or status["failed"]:
        raise UrduProductionError("Urdu revision recovery is not complete")
    artifacts = {
        str(target["unit_id"]): file_hash(
            base / "units" / str(target["unit_id"]) / "revision.json"
        )
        for target in marker["targets"]
    }
    complete = {
        "version": "urdu-revision-billing-recovery-v1",
        "run_id": base.name,
        "target_hash": marker["target_hash"],
        "target_count": marker["target_count"],
        "artifacts": artifacts,
        "artifact_hash": stable_hash(artifacts),
    }
    path = base / COMPLETE_NAME
    if path.exists() and _load_json(path) != complete:
        raise UrduProductionError("Frozen Urdu revision recovery result changed")
    atomic_json(path, complete)
    return recovery_status(base, marker)


def run_recovery(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
    marker: dict[str, Any],
    *,
    budget: BudgetLedger,
    client: AnthropicBatchClient,
    poll_seconds: int = 30,
) -> dict[str, Any]:
    marker = _validate_targets(base, units, verses, bismillah, marker)
    unit_by_id = {unit.unit_id: unit for unit in units}
    pending: list[tuple[ProductionUnit, str, str, list[str], str]] = []
    for target in marker["targets"]:
        unit = unit_by_id[str(target["unit_id"])]
        system, user, finding_ids, input_hash = _revision_material(
            base, unit, verses, bismillah
        )
        path = artifact_path(base, unit, "revision")
        if path.is_file():
            document = _load_json(path)
            result = document.get("result")
            if (
                document.get("input_hash") != input_hash
                or validate_revision(
                    result,
                    expected=unit.expected_ayahs,
                    finding_ids=finding_ids,
                )
                is None
            ):
                raise UrduProductionError(
                    f"Recovered Urdu revision artifact changed: {path}"
                )
            continue
        pending.append((unit, system, user, finding_ids, input_hash))
    if not pending:
        return _write_complete(base, marker)

    requests = [
        {"custom_id": unit.unit_id, "params": _revision_request(system, user)}
        for unit, system, user, _finding_ids, _input_hash in pending
    ]
    request_hash = stable_hash(requests)
    batch_path = base / BATCH_NAME
    reservation = 0.0
    try:
        if batch_path.exists():
            job = _load_json(batch_path)
            if (
                job.get("version") != "urdu-revision-billing-recovery-batch-v1"
                or job.get("request_hash") != request_hash
                or job.get("unit_ids")
                != [unit.unit_id for unit, *_rest in pending]
            ):
                raise UrduProductionError("Frozen revision recovery batch changed")
            state = client.retrieve(str(job["batch_id"]))
        else:
            reservation = sum(
                estimate_request_ceiling(
                    REVISION_MODEL.model_id,
                    system,
                    user,
                    REVISION_MAX_OUTPUT_TOKENS,
                )
                for _unit, system, user, _finding_ids, _input_hash in pending
            )
            budget.reserve(reservation)
            state = client.submit(requests)
            atomic_json(
                batch_path,
                {
                    "version": "urdu-revision-billing-recovery-batch-v1",
                    "batch_id": state.batch_id,
                    "state": state.state,
                    "request_hash": request_hash,
                    "target_hash": marker["target_hash"],
                    "unit_ids": [unit.unit_id for unit, *_rest in pending],
                },
            )
        while not state.ended:
            job = _load_json(batch_path)
            job["state"] = state.state
            atomic_json(batch_path, job)
            print(
                f"revision recovery batch {state.batch_id}: {state.state}",
                flush=True,
            )
            time.sleep(poll_seconds)
            state = client.retrieve(state.batch_id)
        job = _load_json(batch_path)
        job["state"] = state.state
        atomic_json(batch_path, job)
        if not state.succeeded:
            raise ProviderError(f"Revision recovery batch ended in {state.state}")

        rows = {
            str(row.get("custom_id")): row for row in client.results(state.batch_id)
        }
        errors: dict[str, list[str]] = {}
        for unit, _system, _user, finding_ids, input_hash in pending:
            row = rows.get(unit.unit_id)
            usage: dict[str, Any] = {}
            try:
                if row is None:
                    raise ProviderError("Anthropic recovery batch omitted the unit")
                text, metadata = anthropic_result_text(row)
                usage = metadata.get("usage", {})
                result = validate_revision(
                    extract_json(text),
                    expected=unit.expected_ayahs,
                    finding_ids=finding_ids,
                )
                if result is None:
                    raise UrduProductionError("revision failed strict contract")
                atomic_json(
                    artifact_path(base, unit, "revision"),
                    {
                        "version": "urdu-production-stage-v1",
                        "stage": "revision",
                        "unit_id": unit.unit_id,
                        "input_hash": input_hash,
                        "status": "complete",
                        "model": asdict(REVISION_MODEL),
                        "attempts": RECOVERY_ATTEMPT,
                        "recovery": {
                            "version": "urdu-revision-billing-recovery-v1",
                            "target_hash": marker["target_hash"],
                            "preserved_failure": "revision-attempt1-FAILED.json",
                        },
                        "usage": usage,
                        "cost_usd": usage_cost(REVISION_MODEL.model_id, usage),
                        "result": result,
                        "raw": row,
                    },
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:3000]
                errors[unit.unit_id] = [error]
                payload: dict[str, Any] = {
                    "version": "urdu-revision-billing-recovery-failure-v1",
                    "stage": "revision_recovery",
                    "unit_id": unit.unit_id,
                    "input_hash": input_hash,
                    "attempt": RECOVERY_ATTEMPT,
                    "errors": [error],
                    "usage": usage,
                    "raw": row,
                }
                if usage:
                    payload["cost_usd"] = usage_cost(
                        REVISION_MODEL.model_id, usage
                    )
                atomic_json(
                    unit_dir(base, unit)
                    / f"revision-recovery-attempt{RECOVERY_ATTEMPT}-FAILED.json",
                    payload,
                )
        if errors:
            atomic_json(
                base / BLOCKED_NAME,
                {
                    "version": "urdu-revision-billing-recovery-blocked-v1",
                    "run_id": base.name,
                    "target_hash": marker["target_hash"],
                    "errors": errors,
                },
            )
            raise UrduProductionError(
                "Urdu revision recovery has failed units: "
                + ", ".join(sorted(errors))
            )
        return _write_complete(base, marker)
    except Exception as exc:
        if not (base / BLOCKED_NAME).exists():
            atomic_json(
                base / BLOCKED_NAME,
                {
                    "version": "urdu-revision-billing-recovery-blocked-v1",
                    "run_id": base.name,
                    "target_hash": marker["target_hash"],
                    "errors": [f"{type(exc).__name__}: {exc}"[:3000]],
                },
            )
        raise
    finally:
        if reservation:
            budget.release(reservation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover Anthropic-billing-blocked Urdu revisions"
    )
    parser.add_argument("command", choices=["prepare", "run", "status"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--hard-cost-ceiling-usd", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = UrduProductionConfig(
        run_id=args.run_id,
        hard_cost_ceiling_usd=args.hard_cost_ceiling_usd,
        workers=args.workers,
    )
    with connect(args.db) as conn:
        base, units, verses, bismillah = prepare_production(conn, config)
    marker = freeze_targets(base, units, verses, bismillah)
    if args.command == "run":
        load_environment()
        client = AnthropicBatchClient(os.environ.get("ANTHROPIC_API_KEY", ""))
        result = run_recovery(
            base,
            units,
            verses,
            bismillah,
            marker,
            budget=BudgetLedger(base, args.hard_cost_ceiling_usd),
            client=client,
            poll_seconds=args.poll_seconds,
        )
    else:
        result = recovery_status(base, marker)
    print(result)


if __name__ == "__main__":
    main()
