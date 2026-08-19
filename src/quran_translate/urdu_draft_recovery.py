"""Non-destructive recovery for externally blocked Urdu draft requests."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import DEFAULT_DB_PATH
from .db import connect
from .production_packets import ProductionUnit, atomic_json, atomic_text
from .urdu_costs import estimate_request_ceiling, usage_cost
from .urdu_production import (
    DRAFT_MODEL,
    DRAFT_PROMPT_PATH,
    MAX_OUTPUT_TOKENS,
    BudgetLedger,
    UrduProductionConfig,
    UrduProductionError,
    _draft_input,
    _load_json,
    artifact_path,
    prepare_production,
    run_dir,
    unit_dir,
)
from .urdu_provider import is_terminal_provider_failure
from .urdu_translation_bakeoff import (
    CONTRACT_ATTEMPTS,
    PROVIDER_CALLS,
    TRANSLATION_SCHEMA,
    extract_json,
    file_hash,
    load_environment,
    stable_hash,
    validate_translation,
)


TARGETS_NAME = "DRAFT_RECOVERY_TARGETS.json"
COMPLETE_NAME = "DRAFT_RECOVERY_COMPLETE.json"
BLOCKED_NAME = "DRAFT_RECOVERY_BLOCKED.json"
KEY_LIMIT_MARKER = "key limit exceeded"


class RecoveryProviderBlocked(UrduProductionError):
    """The provider must be unblocked before recovery can continue."""


def _errors(document: dict[str, Any]) -> list[str]:
    return [str(item) for item in document.get("errors", [])]


def _is_key_limit_failure(document: dict[str, Any]) -> bool:
    return any(KEY_LIMIT_MARKER in error.lower() for error in _errors(document))


def _draft_input_hash(
    base: Path,
    unit: ProductionUnit,
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
) -> tuple[str, str, str]:
    system, user = _draft_input(base, unit, verses, bismillah)
    input_hash = stable_hash(
        {
            "stage": "draft",
            "unit": unit.to_dict(),
            "model": asdict(DRAFT_MODEL),
            "system": system,
            "user": user,
            "schema": TRANSLATION_SCHEMA,
            "contract_attempts": CONTRACT_ATTEMPTS,
        }
    )
    return system, user, input_hash


def _validate_frozen_targets(base: Path, marker: dict[str, Any]) -> dict[str, Any]:
    targets = marker.get("targets")
    if (
        marker.get("version") != "urdu-draft-recovery-targets-v1"
        or marker.get("run_id") != base.name
        or not isinstance(targets, list)
        or marker.get("target_count") != len(targets)
        or marker.get("target_hash") != stable_hash(targets)
    ):
        raise UrduProductionError("Invalid frozen Urdu draft recovery marker")
    for target in targets:
        unit_root = base / "units" / str(target["unit_id"])
        current = unit_root / "draft.json"
        archive = unit_root / "draft-pre-recovery-FAILED.json"
        source = archive if archive.exists() else current
        if not source.exists() or file_hash(source) != target["failed_summary_sha256"]:
            raise UrduProductionError(
                f"Frozen draft recovery input changed: {source}"
            )
        for name, expected_hash in target.get("failure_sidecars", {}).items():
            sidecar = unit_root / str(name)
            if not sidecar.exists() or file_hash(sidecar) != expected_hash:
                raise UrduProductionError(
                    f"Frozen draft recovery sidecar changed: {sidecar}"
                )
    return marker


def freeze_targets(base: Path, units: list[ProductionUnit]) -> dict[str, Any]:
    marker_path = base / TARGETS_NAME
    if marker_path.exists():
        return _validate_frozen_targets(base, _load_json(marker_path))

    targets: list[dict[str, Any]] = []
    for unit in units:
        draft_path = artifact_path(base, unit, "draft")
        if not draft_path.is_file():
            raise UrduProductionError(
                f"Draft recovery found a missing artifact: {draft_path}"
            )
        document = _load_json(draft_path)
        if document.get("status") == "complete":
            continue
        if document.get("status") != "failed" or not _is_key_limit_failure(document):
            raise UrduProductionError(
                f"Draft recovery refuses a non-key-limit failure: {draft_path}"
            )
        if document.get("usage") or document.get("raw_text"):
            raise UrduProductionError(
                f"Draft recovery refuses a failure containing provider output: {draft_path}"
            )
        sidecars = sorted(unit_dir(base, unit).glob("draft-attempt*-FAILED.json"))
        targets.append(
            {
                "unit_id": unit.unit_id,
                "failed_summary_sha256": file_hash(draft_path),
                "failure_sidecars": {
                    item.name: file_hash(item) for item in sidecars
                },
            }
        )
    marker = {
        "version": "urdu-draft-recovery-targets-v1",
        "run_id": base.name,
        "reason": "openrouter_key_limit_exceeded",
        "targets": targets,
        "target_count": len(targets),
    }
    marker["target_hash"] = stable_hash(targets)
    atomic_json(marker_path, marker)
    return _validate_frozen_targets(base, marker)


def _archive_failed_summary(path: Path, expected_sha256: str) -> Path:
    if file_hash(path) != expected_sha256:
        raise UrduProductionError(f"Failed draft summary changed before recovery: {path}")
    archive = path.with_name("draft-pre-recovery-FAILED.json")
    if archive.exists():
        if file_hash(archive) != expected_sha256:
            raise UrduProductionError(f"Draft recovery archive changed: {archive}")
        return archive
    atomic_text(archive, path.read_text(encoding="utf-8"))
    if file_hash(archive) != expected_sha256:
        raise UrduProductionError(f"Draft recovery archive was not byte exact: {archive}")
    return archive


def _recover_one(
    *,
    base: Path,
    unit: ProductionUnit,
    target: dict[str, Any],
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
    budget: BudgetLedger,
) -> dict[str, str]:
    path = artifact_path(base, unit, "draft")
    system, user, input_hash = _draft_input_hash(base, unit, verses, bismillah)
    current = _load_json(path)
    if current.get("status") == "complete":
        if current.get("input_hash") != input_hash or validate_translation(
            current.get("result"), unit.expected_ayahs
        ) is None:
            raise UrduProductionError(f"Recovered draft fails validation: {path}")
        return {"unit_id": unit.unit_id, "status": "reused"}
    if current.get("input_hash") != input_hash:
        raise UrduProductionError(f"Draft input changed before recovery: {path}")
    archive = _archive_failed_summary(path, str(target["failed_summary_sha256"]))

    consumed: set[int] = set()
    prior_errors: list[str] = []
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        failed_path = unit_dir(base, unit) / f"draft-recovery-attempt{attempt}-FAILED.json"
        if not failed_path.exists():
            continue
        failed = _load_json(failed_path)
        if failed.get("input_hash") != input_hash:
            raise UrduProductionError(f"Recovery attempt input changed: {failed_path}")
        consumed.add(attempt)
        prior_errors.extend(_errors(failed))

    started = time.monotonic()
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        if attempt in consumed:
            continue
        assignment = user + (
            "\n\nPrevious response failed the strict contract: " + prior_errors[-1]
            if prior_errors
            else ""
        )
        reservation = estimate_request_ceiling(
            DRAFT_MODEL.model_id, system, assignment, MAX_OUTPUT_TOKENS
        )
        budget.reserve(reservation)
        raw_text = ""
        raw_response: dict[str, Any] | None = None
        usage: dict[str, Any] = {}
        try:
            raw_text, usage, raw_response = PROVIDER_CALLS[DRAFT_MODEL.provider](
                DRAFT_MODEL, system, assignment, TRANSLATION_SCHEMA
            )
            result = validate_translation(extract_json(raw_text), unit.expected_ayahs)
            if result is None:
                raise UrduProductionError("response failed strict draft contract")
            atomic_json(
                path,
                {
                    "version": "urdu-production-stage-v1",
                    "stage": "draft",
                    "unit_id": unit.unit_id,
                    "input_hash": input_hash,
                    "status": "complete",
                    "model": asdict(DRAFT_MODEL),
                    "attempts": attempt,
                    "recovery": {
                        "version": "urdu-draft-recovery-v1",
                        "failed_summary_archive": archive.name,
                        "failed_summary_sha256": target["failed_summary_sha256"],
                    },
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "usage": usage,
                    "cost_usd": usage_cost(DRAFT_MODEL.model_id, usage),
                    "result": result,
                    "errors_before_success": prior_errors,
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                },
            )
            return {"unit_id": unit.unit_id, "status": "complete"}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:3000]
            terminal = is_terminal_provider_failure(exc)
            payload: dict[str, Any] = {
                "version": "urdu-draft-recovery-attempt-v1",
                "stage": "draft_recovery",
                "unit_id": unit.unit_id,
                "input_hash": input_hash,
                "attempt": attempt,
                "errors": [error],
                "usage": usage,
                "raw_text": raw_text,
                "raw_response": raw_response,
                "terminal_provider_failure": terminal,
            }
            if usage:
                payload["cost_usd"] = usage_cost(DRAFT_MODEL.model_id, usage)
            if terminal:
                blocked_path = unit_dir(base, unit) / (
                    f"draft-recovery-provider-block-{time.time_ns()}-FAILED.json"
                )
                atomic_json(blocked_path, payload)
                raise RecoveryProviderBlocked(error) from exc
            atomic_json(
                unit_dir(base, unit) / f"draft-recovery-attempt{attempt}-FAILED.json",
                payload,
            )
            prior_errors.append(error)
        finally:
            budget.release(reservation)

    atomic_json(
        unit_dir(base, unit) / "draft-recovery-FAILED.json",
        {
            "version": "urdu-draft-recovery-failure-v1",
            "unit_id": unit.unit_id,
            "input_hash": input_hash,
            "status": "failed",
            "attempts": CONTRACT_ATTEMPTS,
            "errors": prior_errors,
        },
    )
    return {"unit_id": unit.unit_id, "status": "failed"}


def _run_bounded(
    jobs: list[tuple[ProductionUnit, Any]], workers: int
) -> None:
    blocked: RecoveryProviderBlocked | None = None
    next_job = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active: dict[Any, ProductionUnit] = {}
        while next_job < len(jobs) and len(active) < workers:
            unit, call = jobs[next_job]
            active[executor.submit(call)] = unit
            next_job += 1
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                unit = active.pop(future)
                try:
                    outcome = future.result()
                    print(f"{outcome['status']}: {unit.unit_id}", flush=True)
                except RecoveryProviderBlocked as exc:
                    blocked = exc
                    print(f"provider-blocked: {unit.unit_id}", flush=True)
                if blocked is None and next_job < len(jobs):
                    next_unit, call = jobs[next_job]
                    active[executor.submit(call)] = next_unit
                    next_job += 1
    if blocked is not None:
        raise blocked


def recovery_status(base: Path, marker: dict[str, Any]) -> dict[str, Any]:
    complete = failed = pending = 0
    for target in marker["targets"]:
        path = base / "units" / target["unit_id"] / "draft.json"
        document = _load_json(path)
        if document.get("status") == "complete":
            complete += 1
        elif (path.parent / "draft-recovery-FAILED.json").exists():
            failed += 1
        else:
            pending += 1
    return {
        "version": "urdu-draft-recovery-status-v1",
        "run_id": base.name,
        "targets": marker["target_count"],
        "recovered": complete,
        "failed": failed,
        "pending": pending,
        "provider_blocked": (base / BLOCKED_NAME).exists()
        and not (base / COMPLETE_NAME).exists(),
        "complete": (base / COMPLETE_NAME).exists(),
    }


def run_recovery(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
    marker: dict[str, Any],
    *,
    budget: BudgetLedger,
    workers: int,
) -> dict[str, Any]:
    unit_by_id = {unit.unit_id: unit for unit in units}
    jobs: list[tuple[ProductionUnit, Any]] = []
    for target in marker["targets"]:
        unit = unit_by_id[str(target["unit_id"])]
        jobs.append(
            (
                unit,
                lambda unit=unit, target=target: _recover_one(
                    base=base,
                    unit=unit,
                    target=target,
                    verses=verses,
                    bismillah=bismillah,
                    budget=budget,
                ),
            )
        )
    try:
        _run_bounded(jobs, workers)
    except RecoveryProviderBlocked as exc:
        atomic_json(
            base / BLOCKED_NAME,
            {
                "version": "urdu-draft-recovery-blocked-v1",
                "run_id": base.name,
                "reason": str(exc),
                "status": recovery_status(base, marker),
            },
        )
        raise
    status = recovery_status(base, marker)
    if status["pending"] or status["failed"]:
        raise UrduProductionError(f"Draft recovery is incomplete: {status}")
    complete = {
        "version": "urdu-draft-recovery-complete-v1",
        "run_id": base.name,
        "target_hash": marker["target_hash"],
        "recovered": marker["target_count"],
    }
    path = base / COMPLETE_NAME
    if path.exists() and _load_json(path) != complete:
        raise UrduProductionError("Urdu draft recovery completion marker changed")
    atomic_json(path, complete)
    return recovery_status(base, marker)


def run_canary(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
    marker: dict[str, Any],
    *,
    budget: BudgetLedger,
) -> dict[str, Any]:
    unit_by_id = {unit.unit_id: unit for unit in units}
    for target in marker["targets"]:
        unit = unit_by_id[str(target["unit_id"])]
        document = _load_json(artifact_path(base, unit, "draft"))
        if document.get("status") == "complete":
            continue
        _recover_one(
            base=base,
            unit=unit,
            target=target,
            verses=verses,
            bismillah=bismillah,
            budget=budget,
        )
        break
    return recovery_status(base, marker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover blocked Urdu draft units")
    parser.add_argument("command", choices=["prepare", "canary", "run", "status"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--hard-cost-ceiling-usd", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=3)
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
    marker = freeze_targets(base, units)
    if args.command in {"canary", "run"}:
        load_environment()
        budget = BudgetLedger(base, args.hard_cost_ceiling_usd)
        if args.command == "canary":
            result = run_canary(
                base,
                units,
                verses,
                bismillah,
                marker,
                budget=budget,
            )
        else:
            result = run_recovery(
                base,
                units,
                verses,
                bismillah,
                marker,
                budget=budget,
                workers=args.workers,
            )
    else:
        result = recovery_status(base, marker)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
