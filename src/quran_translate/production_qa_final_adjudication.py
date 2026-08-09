"""Terminal adjudication and persistence for bounded production QA remediation."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .config import DEFAULT_DB_PATH
from .db import connect, utc_now
from .production_clients import AnthropicBatchClient
from .production_packets import ProductionUnit, atomic_json, atomic_text
from .production_qa_remediation import (
    ADJUDICATION_FILE,
    BLOCKED_FILE,
    COMPLETE_FILE,
    CORRECTIONS_FILE,
    MANIFEST_FILE,
    REMEDIATION_DIR,
    REPORT_JSON,
    REPORT_MD,
    TARGETS_FILE,
    VERIFICATION_BUNDLE_SIZE,
    _candidate_targets,
    _deterministic_candidate_issues,
    _finding_id,
    _load_run_config,
    _markdown_report,
    _validate_verification,
    run_opus_adjudication,
)
from .production_runner import (
    ANTHROPIC_MODEL,
    ProductionConfig,
    ProductionError,
    cached_system_blocks,
    file_hash,
    load_environment,
    prepare_production,
    production_status,
    prompt_material,
    run_dir,
    stable_hash,
)
from .validation import validate_run, validate_source


FINAL_VERSION = "production-qa-remediation-v2-final-adjudication"
FINAL_DIR = "final-adjudication"
FINAL_TARGETS = "TARGETS.json"
FINAL_MANIFEST = "MANIFEST.json"
FINAL_COMPLETE = "FINAL_ADJUDICATION_COMPLETE.json"


FINAL_SYSTEM = """You are the terminal adjudicator for a Quran translation QA
remediation. A separate Gemini audit has returned the supplied findings after an
earlier Opus adjudication. Each finding is a falsifiable hypothesis, not a command.

For every finding, decide applied, rejected, or escalated from the supplied Arabic,
morphology, immediate context, project policy, and documented sense ledger. Apply
the smallest correction for a real defect. Reject a finding when it overstates one
interpretation or would erase materially live source ambiguity. Escalate only when
the supplied evidence cannot support a responsible decision.

Priority is fidelity, then preserved ambiguity, natural spoken English, and literary
force. Root imagery is not a definition. Do not add imagery, agency, causality,
motive, specificity, temporal sequence, or moral judgment. Do not treat a familiar
published rendering as evidence. Reasons may cite only supplied material.

Identical-Arabic groups remain mechanically invariant. If one shared line occurs in
different temporal contexts, choose one English wording that is faithful and
grammatically usable in every supplied context; do not change non-target ayahs.
This adjudication is terminal: resolve every finding explicitly and return only the
registered JSON contract in target order.
"""


def _load_v1_candidates(remediation: Path) -> list[dict[str, Any]]:
    targets = json.loads((remediation / TARGETS_FILE).read_text(encoding="utf-8"))[
        "targets"
    ]
    adjudication = json.loads(
        (remediation / ADJUDICATION_FILE).read_text(encoding="utf-8")
    )["result"]
    return _candidate_targets(targets, adjudication)


def _load_verification(
    remediation: Path,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bundles = [
        candidates[index : index + VERIFICATION_BUNDLE_SIZE]
        for index in range(0, len(candidates), VERIFICATION_BUNDLE_SIZE)
    ]
    rows: list[dict[str, Any]] = []
    for bundle_number, bundle in enumerate(bundles, start=1):
        path = remediation / "verification" / f"bundle-{bundle_number:03d}.json"
        if not path.exists():
            raise ProductionError(f"Missing remediation verifier bundle: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = _validate_verification(payload.get("result"), bundle)
        if result is None:
            raise ProductionError(f"Invalid remediation verifier bundle: {path}")
        rows.extend(result)
    validated = _validate_verification(rows, candidates)
    if validated is None:
        raise ProductionError("Combined remediation verification fails contract")
    return validated


def build_final_targets(
    candidates: list[dict[str, Any]],
    verification: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_ref = {target["ref"]: target for target in candidates}
    findings_by_ref: dict[str, list[dict[str, Any]]] = {}
    for row in verification:
        for finding in row["findings"]:
            record = {
                "ref": row["ref"],
                "stage": "qa_remediation_verification",
                **finding,
            }
            record["finding_id"] = _finding_id(record)
            findings_by_ref.setdefault(row["ref"], []).append(record)
    if not findings_by_ref:
        return []

    target_refs = set(findings_by_ref)
    for ref in list(target_refs):
        group = by_ref[ref].get("identical_group")
        if group:
            target_refs.update(group["refs"])

    targets: list[dict[str, Any]] = []
    for ref in sorted(target_refs, key=lambda value: tuple(map(int, value.split(":")))):
        prior = by_ref[ref]
        targets.append(
            {
                "ref": ref,
                "unit_id": prior["unit_id"],
                "arabic": prior["arabic"],
                "current_english": prior["candidate_english"],
                "findings": findings_by_ref.get(ref, []),
                "closure_only": ref not in findings_by_ref,
                "identical_group": prior.get("identical_group"),
                "local_context": prior["local_context"],
                "parallel_occurrences": prior["parallel_occurrences"],
                "morphology": prior["morphology"],
                "prior_adjudication": prior["decisions"],
            }
        )
    return targets


def _freeze_final(
    remediation: Path,
    targets: list[dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]]]:
    final_dir = remediation / FINAL_DIR
    final_dir.mkdir(parents=True, exist_ok=True)
    targets_path = final_dir / FINAL_TARGETS
    record = {"version": FINAL_VERSION, "targets": targets}
    if targets_path.exists():
        if json.loads(targets_path.read_text(encoding="utf-8")) != record:
            raise ProductionError("Frozen final-adjudication targets changed")
    else:
        atomic_json(targets_path, record)

    frozen_targets = json.loads(targets_path.read_text(encoding="utf-8"))["targets"]
    verification_paths = sorted((remediation / "verification").glob("bundle-???.json"))
    manifest = {
        "version": FINAL_VERSION,
        "qa_remediation_manifest_sha256": file_hash(remediation / MANIFEST_FILE),
        "targets_sha256": file_hash(remediation / TARGETS_FILE),
        "opus_adjudication_sha256": file_hash(remediation / ADJUDICATION_FILE),
        "schema_recovery_manifest_sha256": file_hash(
            remediation / "GEMINI_SCHEMA_RECOVERY_MANIFEST.json"
        ),
        "schema_recovery_complete_sha256": file_hash(
            remediation / "GEMINI_SCHEMA_RECOVERY_COMPLETE.json"
        ),
        "verification_sha256": {
            path.name: file_hash(path) for path in verification_paths
        },
        "final_code_sha256": file_hash(Path(__file__)),
        "final_targets_sha256": file_hash(targets_path),
        "model": ANTHROPIC_MODEL,
        "system_sha256": stable_hash(FINAL_SYSTEM),
    }
    manifest_path = final_dir / FINAL_MANIFEST
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ProductionError("Final-adjudication manifest changed")
    else:
        atomic_json(manifest_path, manifest)
    return final_dir, frozen_targets


def _merge_final_candidates(
    candidates: list[dict[str, Any]],
    final_targets: list[dict[str, Any]],
    final_adjudication: dict[str, Any],
) -> list[dict[str, Any]]:
    final_item_by_ref = {
        item["ref"]: item for item in final_adjudication.get("items", [])
    }
    final_target_refs = {target["ref"] for target in final_targets}
    if set(final_item_by_ref) != final_target_refs:
        raise ProductionError("Final adjudication coverage changed")
    merged: list[dict[str, Any]] = []
    for candidate in candidates:
        item = final_item_by_ref.get(candidate["ref"])
        merged.append(
            {
                **candidate,
                "candidate_english": (
                    item["english"] if item else candidate["candidate_english"]
                ),
                "final_decisions": item["decisions"] if item else [],
            }
        )
    return merged


def _persist(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    candidates: list[dict[str, Any]],
    corrections_hash: str,
    final_artifact_hash: str,
) -> None:
    now = utc_now()
    for target in candidates:
        row = conn.execute(
            """
            SELECT translation, raw_translation_json
            FROM translations
            WHERE run_id = ? AND verse_key = ?
            """,
            (run_id, target["ref"]),
        ).fetchone()
        if row is None:
            raise ProductionError(f"Cannot persist remediation; missing {target['ref']}")
        raw = json.loads(row["raw_translation_json"])
        raw["qa_remediation"] = {
            "version": FINAL_VERSION,
            "source_translation": target["current_english"],
            "translation": target["candidate_english"],
            "initial_adjudication": target["decisions"],
            "verification_adjudication": target["final_decisions"],
            "corrections_sha256": corrections_hash,
            "final_adjudication_sha256": final_artifact_hash,
        }
        conn.execute(
            """
            UPDATE translations
            SET translation = ?, raw_translation_json = ?, updated_at = ?
            WHERE run_id = ? AND verse_key = ?
            """,
            (
                target["candidate_english"],
                json.dumps(raw, ensure_ascii=False, sort_keys=True),
                now,
                run_id,
                target["ref"],
            ),
        )


def _write_report(remediation: Path, report: dict[str, Any]) -> None:
    atomic_json(remediation / REPORT_JSON, report)
    markdown = _markdown_report(report)
    if report.get("governance"):
        markdown += f"\nFinal adjudication: {report['governance']}\n"
    atomic_text(remediation / REPORT_MD, markdown)


def finalize(
    conn: sqlite3.Connection,
    *,
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    candidates: list[dict[str, Any]],
    verification: list[dict[str, Any]],
    final_artifact: Path,
) -> dict[str, Any]:
    remediation = base / REMEDIATION_DIR
    final_decisions = [
        decision for target in candidates for decision in target["final_decisions"]
    ]
    issues = [
        {
            "code": "final_adjudication_escalated",
            "ref": decision["finding_id"],
            "message": decision["reason"],
        }
        for decision in final_decisions
        if decision["decision"] == "escalated"
    ]
    english = {
        str(row["verse_key"]): str(row["translation"])
        for row in conn.execute(
            "SELECT verse_key, translation FROM translations WHERE run_id = ?",
            (base.name,),
        )
    }
    for target in candidates:
        english[target["ref"]] = target["candidate_english"]
    issues.extend(_deterministic_candidate_issues(verses=verses, english=english))

    original_qa = json.loads((base / "QA_REPORT.json").read_text(encoding="utf-8"))
    initial_decisions = [decision for target in candidates for decision in target["decisions"]]
    final_counts = {
        status: sum(d["decision"] == status for d in final_decisions)
        for status in ("applied", "rejected", "escalated")
    }
    initial_counts = {
        status: sum(d["decision"] == status for d in initial_decisions)
        for status in ("applied", "rejected", "escalated")
    }
    changed = [
        target
        for target in candidates
        if target["candidate_english"] != target["current_english"]
    ]
    major_findings = sum(
        finding["severity"] in {"blocking", "significant"}
        for row in verification
        for finding in row["findings"]
    )
    report = {
        "version": FINAL_VERSION,
        "run_id": base.name,
        "passed": not issues,
        "original_qa_sha256": file_hash(base / "QA_REPORT.json"),
        "original_gate_errors": original_qa["issue_counts"]["error"],
        "target_refs": len(candidates),
        "changed_ayahs": len(changed),
        "decisions": final_counts,
        "initial_decisions": initial_counts,
        "verification_findings": sum(
            len(row["findings"]) for row in verification
        ),
        "major_verification_findings": major_findings,
        "unresolved_major_verification_findings": sum(
            decision["decision"] == "escalated"
            for decision in final_decisions
        ),
        "issues": issues,
        "governance": (
            "Gemini findings were hypotheses adjudicated by Opus; rejected findings "
            "are resolved, not silently discarded. The terminal stage does not loop."
        ),
    }
    _write_report(remediation, report)
    if issues:
        atomic_json(remediation / BLOCKED_FILE, report)
        raise ProductionError(
            f"Final QA adjudication blocked by {len(issues)} issue(s)"
        )

    corrections = {
        "version": FINAL_VERSION,
        "run_id": base.name,
        "targets": [
            {
                "ref": target["ref"],
                "before": target["current_english"],
                "after": target["candidate_english"],
                "changed": target["candidate_english"]
                != target["current_english"],
                "initial_decisions": target["decisions"],
                "verification_adjudication": target["final_decisions"],
            }
            for target in candidates
        ],
    }
    atomic_json(remediation / CORRECTIONS_FILE, corrections)
    corrections_hash = file_hash(remediation / CORRECTIONS_FILE)
    final_artifact_hash = file_hash(final_artifact)

    conn.execute("BEGIN")
    try:
        _persist(
            conn,
            run_id=base.name,
            candidates=candidates,
            corrections_hash=corrections_hash,
            final_artifact_hash=final_artifact_hash,
        )
        run_issues = validate_source(conn) + validate_run(conn, base.name)
        hard = [
            issue
            for issue in run_issues
            if issue.severity == "error"
            or issue.message.startswith("Banned/jargon term")
        ]
        if hard:
            raise ProductionError(
                "Post-remediation database validation failed: "
                + "; ".join(issue.message for issue in hard[:10])
            )
        conn.execute(
            """
            UPDATE translation_runs
            SET status = 'complete', updated_at = ?
            WHERE run_id = ?
            """,
            (utc_now(), base.name),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        blocked_report = {
            **report,
            "passed": False,
            "issues": [
                *report["issues"],
                {"code": "post_persistence_validation", "message": str(exc)},
            ],
        }
        _write_report(remediation, blocked_report)
        atomic_json(remediation / BLOCKED_FILE, blocked_report)
        raise

    status = production_status(base, units)
    status.update(
        {
            "completion_mode": FINAL_VERSION,
            "quality": {
                "passed": True,
                "original_gate_errors": report["original_gate_errors"],
                "remediation_report": f"{REMEDIATION_DIR}/{REPORT_JSON}",
                "changed_ayahs": report["changed_ayahs"],
                "verification_findings_adjudicated": report[
                    "verification_findings"
                ],
            },
            "remediation_sha256": corrections_hash,
            "final_adjudication_sha256": final_artifact_hash,
        }
    )
    final_dir = remediation / FINAL_DIR
    atomic_json(final_dir / FINAL_COMPLETE, status)
    atomic_json(remediation / COMPLETE_FILE, status)
    (remediation / BLOCKED_FILE).unlink(missing_ok=True)
    (base / "QA_BLOCKED.json").unlink(missing_ok=True)
    atomic_json(base / "PRODUCTION_COMPLETE.json", status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = run_dir(args.run_id)
    config: ProductionConfig = _load_run_config(base)
    with connect(args.db) as conn:
        prepared_base, units, verses = prepare_production(conn, config)
        remediation = prepared_base / REMEDIATION_DIR
        candidates = _load_v1_candidates(remediation)
        verification = _load_verification(remediation, candidates)
        targets = build_final_targets(candidates, verification)
        if not targets:
            raise ProductionError("No verifier findings require final adjudication")
        final_dir, frozen_targets = _freeze_final(remediation, targets)

        complete_path = final_dir / FINAL_COMPLETE
        if complete_path.exists():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                complete.get("completion_mode") != FINAL_VERSION
                or complete.get("remediation_sha256")
                != file_hash(remediation / CORRECTIONS_FILE)
                or complete.get("final_adjudication_sha256")
                != file_hash(final_dir / ADJUDICATION_FILE)
            ):
                raise ProductionError("Final-adjudication completion record changed")
            if not (prepared_base / "PRODUCTION_COMPLETE.json").exists():
                atomic_json(prepared_base / "PRODUCTION_COMPLETE.json", complete)
            print(json.dumps(complete, ensure_ascii=False, indent=2))
            return

        load_environment()
        prompt, ledger_md, ledger_json = prompt_material()
        system = cached_system_blocks(FINAL_SYSTEM, prompt, ledger_md, ledger_json)
        client = AnthropicBatchClient(os.environ.get("ANTHROPIC_API_KEY", ""))
        final_adjudication = run_opus_adjudication(
            remediation=final_dir,
            targets=frozen_targets,
            system=system,
            client=client,
            poll_seconds=config.poll_seconds,
        )
        merged = _merge_final_candidates(
            candidates,
            frozen_targets,
            final_adjudication,
        )
        status = finalize(
            conn,
            base=prepared_base,
            units=units,
            verses=verses,
            candidates=merged,
            verification=verification,
            final_artifact=final_dir / ADJUDICATION_FILE,
        )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
