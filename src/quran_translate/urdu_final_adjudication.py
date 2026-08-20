"""Guarded human adjudication for the terminal Urdu production QA block."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .config import DATA_DIR, DEFAULT_DB_PATH, PROJECT_ROOT
from .db import connect
from .production_packets import ProductionUnit, atomic_json, atomic_text, source_verses
from .refrains import repeated_ayah_groups
from .urdu_production import (
    UrduProductionError,
    _preliminary_rows,
    production_status,
    run_dir,
)
from .urdu_quality import deterministic_quality_gate
from .urdu_translation_bakeoff import file_hash, stable_hash


DECISIONS_PATH = (
    DATA_DIR / "evidence" / "urdu-production-final-adjudications-v1.json"
)
BACKUP_NAMES = (
    "QA_BLOCKED.json",
    "QA_REPORT.json",
    "REVIEW_QUEUE.json",
    "RUN.json",
    "output/quran-urdu.json",
    "output/quran-urdu.txt",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_decisions(payload: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    if payload.get("version") != "quran-urdu-final-adjudications-v1":
        raise UrduProductionError("Unknown Urdu final-adjudication version")
    if payload.get("run_id") != run_id:
        raise UrduProductionError("Urdu final adjudications target a different run")
    expected_hash = payload.get("expected_unresolved_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise UrduProductionError("Final adjudications lack the unresolved-set hash")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise UrduProductionError("Final adjudications require decisions")
    seen: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise UrduProductionError("Every final adjudication must be an object")
        ref = decision.get("ref")
        if not isinstance(ref, str) or ref in seen:
            raise UrduProductionError(f"Duplicate or invalid adjudication ref: {ref}")
        seen.add(ref)
        try:
            surah_text, ayah_text = ref.split(":", 1)
            int(surah_text)
            ayah = int(ayah_text)
        except (AttributeError, TypeError, ValueError) as exc:
            raise UrduProductionError(f"Invalid adjudication ref: {ref}") from exc
        required = (
            "unit_id",
            "before",
            "after",
            "category",
            "rationale",
            "evidence",
            "expected_unresolved_records",
        )
        if any(not decision.get(field) for field in required):
            raise UrduProductionError(f"Incomplete final adjudication at {ref}")
        if decision.get("ayah") != ayah:
            raise UrduProductionError(f"Ayah guard does not match ref {ref}")
        if decision["before"] == decision["after"]:
            raise UrduProductionError(f"No-op final adjudication at {ref}")
        if not isinstance(decision["evidence"], list) or not decision["evidence"]:
            raise UrduProductionError(f"Evidence list is invalid at {ref}")
        if not isinstance(decision["expected_unresolved_records"], int):
            raise UrduProductionError(f"Unresolved count is invalid at {ref}")
    return decisions


def _apply_decisions(
    rows: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = validate_decisions(payload, str(payload["run_id"]))
    actual_hash = stable_hash(unresolved)
    if actual_hash != payload["expected_unresolved_sha256"]:
        raise UrduProductionError(
            "Frozen unresolved Urdu QA set changed before final adjudication"
        )
    rows_by_ref = {
        f"{int(row['surah'])}:{int(row['ayah'])}": row for row in rows
    }
    resolved_keys: set[tuple[str, int]] = set()
    for decision in decisions:
        ref = str(decision["ref"])
        row = rows_by_ref.get(ref)
        if row is None:
            raise UrduProductionError(f"Missing final translation row for {ref}")
        if row["urdu"] != decision["before"]:
            raise UrduProductionError(f"Exact before-text guard failed at {ref}")
        key = (str(decision["unit_id"]), int(decision["ayah"]))
        matches = [
            item
            for item in unresolved
            if (str(item.get("unit_id")), int(item.get("ayah", -1))) == key
        ]
        if len(matches) != int(decision["expected_unresolved_records"]):
            raise UrduProductionError(
                f"Unresolved-record guard failed at {ref}: found {len(matches)}"
            )
        row["urdu"] = str(decision["after"])
        resolved_keys.add(key)
    remaining = [
        item
        for item in unresolved
        if (str(item.get("unit_id")), int(item.get("ayah", -1)))
        not in resolved_keys
    ]
    return rows, remaining


def _resolved_rows(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, unresolved = _preliminary_rows(base, units)
    refrain_path = base / "REFRAINS_COMPLETE.json"
    if not refrain_path.is_file():
        raise UrduProductionError("Refrain resolution has not completed")
    resolved = _load_json(refrain_path).get("resolved")
    if not isinstance(resolved, dict):
        raise UrduProductionError("Refrain resolution marker is malformed")
    group_by_ref = {
        tuple(ref): group_id
        for group_id, group in repeated_ayah_groups(verses).items()
        for ref in group["refs"]
    }
    for row in rows:
        group_id = group_by_ref.get((int(row["surah"]), int(row["ayah"])))
        choice = resolved.get(group_id) if group_id else None
        if isinstance(choice, dict):
            row["urdu"] = str(choice["urdu"])
    return rows, unresolved


def _backup_pre_adjudication(base: Path) -> dict[str, Any]:
    backup = base / "pre-final-adjudication"
    manifest_path = backup / "BACKUP_MANIFEST.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        for name, expected in manifest["files"].items():
            path = backup / name
            if not path.is_file() or file_hash(path) != expected:
                raise UrduProductionError("Pre-adjudication backup changed")
        return manifest
    files: dict[str, str] = {}
    for name in BACKUP_NAMES:
        source = base / name
        if not source.is_file():
            continue
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, target)
        files[name] = file_hash(target)
    if "QA_BLOCKED.json" not in files:
        raise UrduProductionError("Expected pre-adjudication QA block is absent")
    manifest = {
        "version": "quran-urdu-pre-final-adjudication-backup-v1",
        "files": files,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def apply_final_adjudications(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    decisions_path: Path = DECISIONS_PATH,
) -> dict[str, Any]:
    payload = _load_json(decisions_path)
    decisions = validate_decisions(payload, base.name)
    decision_hash = file_hash(decisions_path)
    marker_path = base / "FINAL_ADJUDICATION_COMPLETE.json"
    complete_path = base / "PRODUCTION_COMPLETE.json"
    if marker_path.exists() and complete_path.exists():
        marker = _load_json(marker_path)
        if marker.get("decisions_sha256") != decision_hash:
            raise UrduProductionError("Completed final adjudication record changed")
        for name, expected in marker["output_sha256"].items():
            if file_hash(base / name) != expected:
                raise UrduProductionError("Final Urdu adjudication output changed")
        return marker

    rows, unresolved = _resolved_rows(base, units, verses)
    rows, unresolved = _apply_decisions(rows, unresolved, payload)
    if unresolved:
        raise UrduProductionError(
            f"Final adjudication left {len(unresolved)} unresolved record(s)"
        )
    translations = {
        (int(row["surah"]), int(row["ayah"])): str(row["urdu"]) for row in rows
    }
    quality = deterministic_quality_gate(translations, verses)
    if not quality["passed"]:
        raise UrduProductionError("Final adjudicated Urdu failed deterministic QA")

    backup = _backup_pre_adjudication(base)
    original_queue = _load_json(base / "REVIEW_QUEUE.json")
    decision_by_key = {
        (str(item["unit_id"]), int(item["ayah"])): item for item in decisions
    }
    updated_critic: list[dict[str, Any]] = []
    adjudicated_critic = 0
    for finding in original_queue["critic_findings"]:
        item = dict(finding)
        decision = decision_by_key.get(
            (str(item.get("unit_id")), int(item.get("ayah", -1)))
        )
        if decision is not None and item.get("severity") in {
            "blocking",
            "significant",
        }:
            item["resolution"] = "human_final_adjudication"
            item["final_adjudication"] = {
                "category": decision["category"],
                "rationale": decision["rationale"],
                "decisions_sha256": decision_hash,
            }
            adjudicated_critic += 1
        updated_critic.append(item)
    if adjudicated_critic != len(decisions):
        raise UrduProductionError(
            "Final adjudications did not match the expected critic findings"
        )

    translator_flags = [
        {
            "ref": f"{row['surah']}:{row['ayah']}",
            "flags": row["review_flags"],
        }
        for row in rows
        if row["review_flags"]
    ]
    review_queue = {
        "version": "quran-urdu-review-queue-v1-final-adjudicated",
        "translator_flags": translator_flags,
        "critic_findings": updated_critic,
        "unresolved": [],
        "final_adjudication": {
            "decisions": len(decisions),
            "decisions_sha256": decision_hash,
            "refs": [item["ref"] for item in decisions],
        },
        "counts": {
            "translator_flagged_ayahs": len(translator_flags),
            "critic_findings": len(updated_critic),
            "blocking_or_significant_findings": sum(
                item["severity"] in {"blocking", "significant"}
                for item in updated_critic
            ),
            "unresolved": 0,
        },
    }
    output = base / "output"
    atomic_json(output / "quran-urdu.json", rows)
    atomic_text(
        output / "quran-urdu.txt",
        "\n".join(
            f"{row['surah']}:{row['ayah']}\t{row['urdu']}" for row in rows
        )
        + "\n",
    )
    atomic_json(
        base / "QA_REPORT.json",
        {
            "quality": quality,
            "unresolved": [],
            "final_adjudication": review_queue["final_adjudication"],
        },
    )
    atomic_json(base / "REVIEW_QUEUE.json", review_queue)

    output_hashes = {
        name: file_hash(base / name)
        for name in (
            "output/quran-urdu.json",
            "output/quran-urdu.txt",
            "QA_REPORT.json",
            "REVIEW_QUEUE.json",
        )
    }
    marker = {
        "version": "quran-urdu-final-adjudication-complete-v1",
        "run_id": base.name,
        "decisions": len(decisions),
        "adjudicated_refs": [item["ref"] for item in decisions],
        "resolved_records": sum(
            int(item["expected_unresolved_records"]) for item in decisions
        ),
        "decisions_path": str(decisions_path),
        "decisions_sha256": decision_hash,
        "backup_manifest_sha256": stable_hash(backup),
        "output_sha256": output_hashes,
        "deterministic_qa": quality,
    }
    atomic_json(marker_path, marker)
    complete = {
        "version": "quran-urdu-production-complete-v1-final-adjudicated",
        "run_id": base.name,
        "ayahs": len(rows),
        "output_json": str(output / "quran-urdu.json"),
        "output_text": str(output / "quran-urdu.txt"),
        "qa_report": str(base / "QA_REPORT.json"),
        "review_queue": str(base / "REVIEW_QUEUE.json"),
        "final_adjudication": str(marker_path),
    }
    atomic_json(complete_path, complete)
    (base / "QA_BLOCKED.json").unlink(missing_ok=True)
    atomic_json(base / "RUN.json", production_status(base, units))
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply guarded final adjudications to a blocked Urdu run"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--decisions", type=Path, default=DECISIONS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = run_dir(args.run_id)
    if not base.is_dir():
        raise UrduProductionError(f"Unknown Urdu production run: {args.run_id}")
    units = [ProductionUnit(**item) for item in _load_json(base / "UNITS.json")]
    manifest = _load_json(base / "MANIFEST.json")
    if stable_hash([unit.to_dict() for unit in units]) != manifest["unit_hash"]:
        raise UrduProductionError("Frozen Urdu unit manifest changed")
    runner = Path(__file__).with_name("urdu_production.py")
    if file_hash(runner) != manifest["runner_sha256"]:
        raise UrduProductionError("Frozen Urdu production runner changed")
    with connect(args.db) as conn:
        verses = source_verses(conn)
    if len(verses) != 6236:
        raise UrduProductionError("Canonical source does not contain 6,236 ayahs")
    marker = apply_final_adjudications(base, units, verses, args.decisions)
    print(json.dumps(marker, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
