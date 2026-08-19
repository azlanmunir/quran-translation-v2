"""Revalidate Urdu critic responses rejected by Arabic orthography mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import urdu_production
from .config import DEFAULT_DB_PATH, PROJECT_ROOT
from .db import connect
from .production_packets import ProductionUnit, atomic_json, atomic_text
from .urdu_production import (
    UrduProductionConfig,
    UrduProductionError,
    _load_json,
    artifact_path,
    prepare_production,
    run_dir,
    unit_dir,
)
from .urdu_quality import validate_critic
from .urdu_translation_bakeoff import extract_json, file_hash, stable_hash


MANIFEST_ARCHIVE_NAME = "MANIFEST_PRE_CRITIC_ORTHOGRAPHY.json"
MANIFEST_AMENDMENT_NAME = "MANIFEST_AMENDMENT_CRITIC_ORTHOGRAPHY.json"
TARGETS_NAME = "CRITIC_REVALIDATION_TARGETS.json"
COMPLETE_NAME = "CRITIC_REVALIDATION_COMPLETE.json"


def _quality_path() -> Path:
    return Path(urdu_production.__file__).with_name("urdu_quality.py")


def _provider_path() -> Path:
    return Path(urdu_production.__file__).with_name("urdu_translation_bakeoff.py")


def amend_quality_manifest(base: Path) -> dict[str, Any]:
    manifest_path = base / "MANIFEST.json"
    archive_path = base / MANIFEST_ARCHIVE_NAME
    amendment_path = base / MANIFEST_AMENDMENT_NAME
    if not manifest_path.exists():
        raise UrduProductionError(f"Missing Urdu production manifest: {manifest_path}")

    current = _load_json(manifest_path)
    new_quality_sha256 = file_hash(_quality_path())
    if amendment_path.exists():
        amendment = _load_json(amendment_path)
        if (
            current.get("quality_sha256") != new_quality_sha256
            or amendment.get("new_quality_sha256") != new_quality_sha256
            or not archive_path.exists()
            or file_hash(archive_path) != amendment.get("old_manifest_sha256")
        ):
            raise UrduProductionError("Critic orthography manifest amendment changed")
        return amendment

    if current.get("quality_sha256") == new_quality_sha256:
        if not archive_path.exists():
            raise UrduProductionError(
                "Quality hash already changed without an amendment archive"
            )
        old_manifest_sha256 = file_hash(archive_path)
        old = _load_json(archive_path)
    else:
        old_manifest_sha256 = file_hash(manifest_path)
        old = current
        if old.get("runner_sha256") != file_hash(Path(urdu_production.__file__)):
            raise UrduProductionError("Runner changed during critic manifest amendment")
        if old.get("provider_sha256") != file_hash(_provider_path()):
            raise UrduProductionError("Provider code changed during critic manifest amendment")
        inputs = old.get("inputs")
        if not isinstance(inputs, dict):
            raise UrduProductionError("Urdu production manifest inputs are malformed")
        for relative, expected_sha256 in inputs.items():
            path = PROJECT_ROOT / str(relative)
            if not path.exists() or file_hash(path) != expected_sha256:
                raise UrduProductionError(
                    f"Frozen production input changed during amendment: {path}"
                )
        atomic_text(archive_path, manifest_path.read_text(encoding="utf-8"))
        if file_hash(archive_path) != old_manifest_sha256:
            raise UrduProductionError("Manifest amendment archive was not byte exact")
        amended = dict(old)
        amended["quality_sha256"] = new_quality_sha256
        atomic_json(manifest_path, amended)
        current = amended

    amendment = {
        "version": "urdu-critic-orthography-manifest-amendment-v1",
        "run_id": base.name,
        "old_manifest_sha256": old_manifest_sha256,
        "old_quality_sha256": old.get("quality_sha256"),
        "new_quality_sha256": new_quality_sha256,
        "changed_field": "quality_sha256",
        "reason": (
            "Accept orthographically equivalent Arabic evidence grounds across "
            "fully marked and minimized Uthmani source forms."
        ),
        "provider_calls": 0,
    }
    if current.get("quality_sha256") != new_quality_sha256:
        raise UrduProductionError("Manifest amendment did not install the quality hash")
    atomic_json(amendment_path, amendment)
    return amendment


def _validate_targets(base: Path, marker: dict[str, Any]) -> dict[str, Any]:
    targets = marker.get("targets")
    if (
        marker.get("version") != "urdu-critic-revalidation-targets-v1"
        or marker.get("run_id") != base.name
        or not isinstance(targets, list)
        or marker.get("target_count") != len(targets)
        or marker.get("target_hash") != stable_hash(targets)
    ):
        raise UrduProductionError("Invalid frozen critic revalidation targets")
    for target in targets:
        root = base / "units" / str(target["unit_id"])
        current = root / "critic.json"
        archive = root / "critic-pre-revalidation-FAILED.json"
        source = archive if archive.exists() else current
        if not source.exists() or file_hash(source) != target["failed_summary_sha256"]:
            raise UrduProductionError(f"Frozen critic summary changed: {source}")
        for name, expected_sha256 in target["attempts"].items():
            path = root / name
            if not path.exists() or file_hash(path) != expected_sha256:
                raise UrduProductionError(f"Frozen critic attempt changed: {path}")
    return marker


def freeze_targets(base: Path, units: list[ProductionUnit]) -> dict[str, Any]:
    marker_path = base / TARGETS_NAME
    if marker_path.exists():
        return _validate_targets(base, _load_json(marker_path))
    targets: list[dict[str, Any]] = []
    for unit in units:
        summary_path = artifact_path(base, unit, "critic")
        if not summary_path.exists():
            raise UrduProductionError(f"Missing critic artifact: {summary_path}")
        summary = _load_json(summary_path)
        if summary.get("status") == "complete":
            continue
        if summary.get("status") != "failed" or summary.get(
            "terminal_provider_failure"
        ):
            raise UrduProductionError(
                f"Critic revalidation refuses provider or unknown failure: {summary_path}"
            )
        attempts = sorted(unit_dir(base, unit).glob("critic-attempt*-FAILED.json"))
        if len(attempts) != 2:
            raise UrduProductionError(
                f"Critic revalidation requires two preserved attempts: {summary_path}"
            )
        targets.append(
            {
                "unit_id": unit.unit_id,
                "failed_summary_sha256": file_hash(summary_path),
                "attempts": {path.name: file_hash(path) for path in attempts},
            }
        )
    marker = {
        "version": "urdu-critic-revalidation-targets-v1",
        "run_id": base.name,
        "reason": "arabic_ground_orthography_normalization",
        "target_count": len(targets),
        "targets": targets,
    }
    marker["target_hash"] = stable_hash(targets)
    atomic_json(marker_path, marker)
    return _validate_targets(base, marker)


def _archive_summary(path: Path, expected_sha256: str) -> Path:
    if file_hash(path) != expected_sha256:
        raise UrduProductionError(f"Critic summary changed before revalidation: {path}")
    archive = path.with_name("critic-pre-revalidation-FAILED.json")
    if archive.exists():
        if file_hash(archive) != expected_sha256:
            raise UrduProductionError(f"Critic revalidation archive changed: {archive}")
        return archive
    atomic_text(archive, path.read_text(encoding="utf-8"))
    if file_hash(archive) != expected_sha256:
        raise UrduProductionError(f"Critic archive was not byte exact: {archive}")
    return archive


def revalidation_status(base: Path, marker: dict[str, Any]) -> dict[str, Any]:
    recovered = failed = 0
    for target in marker["targets"]:
        document = _load_json(base / "units" / target["unit_id"] / "critic.json")
        if document.get("status") == "complete":
            recovered += 1
        else:
            failed += 1
    return {
        "version": "urdu-critic-revalidation-status-v1",
        "run_id": base.name,
        "targets": marker["target_count"],
        "recovered": recovered,
        "failed": failed,
        "complete": (base / COMPLETE_NAME).exists(),
    }


def run_revalidation(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    marker: dict[str, Any],
) -> dict[str, Any]:
    units_by_id = {unit.unit_id: unit for unit in units}
    for target in marker["targets"]:
        unit = units_by_id[str(target["unit_id"])]
        summary_path = artifact_path(base, unit, "critic")
        summary = _load_json(summary_path)
        if summary.get("status") == "complete":
            continue
        archive = _archive_summary(
            summary_path, str(target["failed_summary_sha256"])
        )
        draft = _load_json(artifact_path(base, unit, "draft"))["result"]
        urdu = {int(row["ayah"]): str(row["urdu"]) for row in draft["ayahs"]}
        arabic = {ayah: verses[(unit.surah, ayah)] for ayah in unit.expected_ayahs}
        selected: tuple[int, dict[str, Any], dict[str, Any]] | None = None
        errors: list[str] = []
        for attempt in (1, 2):
            path = unit_dir(base, unit) / f"critic-attempt{attempt}-FAILED.json"
            failed = _load_json(path)
            try:
                parsed = extract_json(str(failed.get("raw_text", "")))
                result = validate_critic(
                    parsed,
                    expected=unit.expected_ayahs,
                    arabic_by_ayah=arabic,
                    urdu_by_ayah=urdu,
                )
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                continue
            if result is None:
                errors.append(f"attempt {attempt}: failed amended critic contract")
                continue
            selected = (attempt, failed, result)
            break
        if selected is None:
            raise UrduProductionError(
                f"Stored critic responses remain invalid for {unit.unit_id}: {errors}"
            )
        attempt, failed, result = selected
        atomic_json(
            summary_path,
            {
                "version": "urdu-production-stage-v1",
                "stage": "critic",
                "unit_id": unit.unit_id,
                "input_hash": summary["input_hash"],
                "status": "complete",
                "model": summary["model"],
                "attempts": attempt,
                "latency_seconds": 0.0,
                "usage": failed.get("usage", {}),
                "result": result,
                "errors_before_success": summary.get("errors", []),
                "raw_text": failed.get("raw_text", ""),
                "raw_response": failed.get("raw_response"),
                "recovery": {
                    "version": "urdu-critic-orthography-revalidation-v1",
                    "failed_summary_archive": archive.name,
                    "failed_summary_sha256": target["failed_summary_sha256"],
                    "selected_attempt": attempt,
                    "additional_provider_cost_usd": 0.0,
                },
            },
        )
        print(f"revalidated: {unit.unit_id} attempt={attempt}", flush=True)

    status = revalidation_status(base, marker)
    if status["failed"]:
        raise UrduProductionError(f"Critic revalidation incomplete: {status}")
    complete = {
        "version": "urdu-critic-revalidation-complete-v1",
        "run_id": base.name,
        "target_hash": marker["target_hash"],
        "recovered": marker["target_count"],
        "provider_calls": 0,
        "additional_cost_usd": 0.0,
    }
    complete_path = base / COMPLETE_NAME
    if complete_path.exists() and _load_json(complete_path) != complete:
        raise UrduProductionError("Critic revalidation completion marker changed")
    atomic_json(complete_path, complete)
    return revalidation_status(base, marker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover Urdu critic orthography rejects")
    parser.add_argument(
        "command", choices=["amend-manifest", "prepare", "run", "status"]
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = run_dir(args.run_id)
    if args.command == "amend-manifest":
        result = amend_quality_manifest(base)
    else:
        with connect(args.db) as conn:
            base, units, verses, _ = prepare_production(
                conn, UrduProductionConfig(run_id=args.run_id)
            )
        marker = freeze_targets(base, units)
        if args.command == "run":
            result = run_revalidation(base, units, verses, marker)
        else:
            result = revalidation_status(base, marker)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
