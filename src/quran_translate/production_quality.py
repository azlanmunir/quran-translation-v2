"""Whole-book quality gate for the v2.4 production run."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .production_packets import ProductionUnit, atomic_json, atomic_text
from .refrains import repeated_ayah_groups
from .spoken_english_v1 import signature_phrase_findings
from .validation import validate_run, validate_source


MAJOR_SEVERITIES = {"blocking", "significant"}
FINAL_FIDELITY_STAGES = ("final_verification", "refrain_verification")


def _stage_path(base: Path, unit: ProductionUnit, stage: str) -> Path:
    return base / "units" / unit.unit_id / f"{stage}.json"


def _stage_findings(
    base: Path,
    units: list[ProductionUnit],
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        path = _stage_path(base, unit, stage)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result")
        if not isinstance(result, list):
            continue
        for entry in result:
            if not isinstance(entry, dict):
                continue
            ayah = entry.get("ayah")
            for finding in entry.get("findings", []):
                if isinstance(finding, dict):
                    rows.append(
                        {
                            "stage": stage,
                            "ref": f"{unit.surah}:{ayah}",
                            **finding,
                        }
                    )
    return rows


def _review_flags(
    conn: sqlite3.Connection,
    run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in conn.execute(
        """
        SELECT verse_key, raw_translation_json
        FROM translations
        WHERE run_id = ? AND status = 'complete'
        ORDER BY verse_key
        """,
        (run_id,),
    ):
        try:
            payload = json.loads(row["raw_translation_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        for flag in payload.get("review_flags", []):
            rows.append(
                {
                    "stage": "translator_review_flag",
                    "ref": row["verse_key"],
                    "flag": flag,
                }
            )
    return rows


def _usage_summary(base: Path) -> dict[str, Any]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    artifacts = list((base / "units").glob("*/*.json")) + list(
        (base / "refrains").glob("*.json")
    )
    for path in artifacts:
        if "FAILED" in path.name or path.name in {"INPUT.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            continue
        if isinstance(usage.get("usage"), dict):
            usage = usage["usage"]
        model = str(payload.get("model") or "unknown")
        for key, value in usage.items():
            if isinstance(value, int | float):
                totals[model][str(key)] += value
    return {
        model: dict(sorted(values.items()))
        for model, values in sorted(totals.items())
    }


def _markdown_report(report: dict[str, Any]) -> str:
    verdict = "PASS" if report["passed"] else "BLOCKED"
    lines = [
        "# Production v2.4 Quality Report",
        "",
        f"- Verdict: **{verdict}**",
        f"- Run: `{report['run_id']}`",
        f"- Source: {report['source']['surahs']} surahs, {report['source']['ayahs']} ayahs",
        f"- Persisted translations: {report['translations']['complete']}",
        f"- Production units: {report['units']}",
        f"- Repeated-Arabic groups: {report['refrains']['groups']} "
        f"({report['refrains']['ayahs']} ayahs)",
        f"- Fidelity review findings remaining: {report['review']['fidelity_findings']}",
        f"- Spoken-English suggestions: {report['review']['spoken_findings']}",
        f"- Translator review flags: {report['review']['translator_flags']}",
        "",
        "## Gate issues",
        "",
    ]
    if not report["issues"]:
        lines.append("None.")
    else:
        for issue in report["issues"]:
            ref = f" ({issue['ref']})" if issue.get("ref") else ""
            lines.append(
                f"- **{issue['severity'].upper()}** `{issue['code']}`{ref}: "
                f"{issue['message']}"
            )
    lines.extend(
        [
            "",
            "## Governance",
            "",
            "The listening text remains note-free. Model spoken-English findings and "
            "translator review flags are review leads, never automatic rewrites. Any "
            "reader-facing note requires separate evidence adjudication.",
            "",
        ]
    )
    return "\n".join(lines)


def run_production_quality_gate(
    conn: sqlite3.Connection,
    *,
    base: Path,
    units: list[ProductionUnit],
    run_id: str,
    verses: dict[tuple[int, int], str],
    refrain_report: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    def issue(code: str, severity: str, message: str, ref: str | None = None) -> None:
        record: dict[str, Any] = {
            "code": code,
            "severity": severity,
            "message": message,
        }
        if ref:
            record["ref"] = ref
        issues.append(record)

    source_issues = validate_source(conn)
    for source_issue in source_issues:
        issue(
            "source_validation",
            source_issue.severity,
            source_issue.message,
            source_issue.ref,
        )

    db_rows = list(
        conn.execute(
            """
            SELECT verse_key, translation, raw_translation_json
            FROM translations
            WHERE run_id = ? AND status = 'complete'
            ORDER BY verse_key
            """,
            (run_id,),
        )
    )
    english = {str(row["verse_key"]): str(row["translation"]) for row in db_rows}
    expected_refs = {f"{surah}:{ayah}" for surah, ayah in verses}
    actual_refs = set(english)
    if actual_refs != expected_refs:
        missing = sorted(expected_refs - actual_refs)[:10]
        unexpected = sorted(actual_refs - expected_refs)[:10]
        issue(
            "translation_coverage",
            "error",
            f"Reference mismatch; missing={missing}, unexpected={unexpected}",
        )
    empty = [ref for ref, text in english.items() if not text.strip()]
    if empty:
        issue("empty_translation", "error", f"Empty English at {empty[:10]}")

    for run_issue in validate_run(conn, run_id):
        severity = run_issue.severity
        if run_issue.message.startswith("Banned/jargon term"):
            severity = "error"
        issue("run_validation", severity, run_issue.message, run_issue.ref)

    groups = repeated_ayah_groups(verses)
    refrain_ayahs = 0
    for group in groups.values():
        refs = [f"{surah}:{ayah}" for surah, ayah in group["refs"]]
        refrain_ayahs += len(refs)
        renderings = {english.get(ref) for ref in refs}
        if len(renderings) != 1 or None in renderings:
            issue(
                "refrain_divergence",
                "error",
                f"Identical Arabic has {len(renderings)} English renderings",
                ", ".join(refs),
            )

    repair_units = [
        unit for unit in units if _stage_path(base, unit, "repair").exists()
    ]
    missing_final_audits = [
        unit.unit_id
        for unit in repair_units
        if not _stage_path(base, unit, "final_verification").exists()
    ]
    if missing_final_audits:
        issue(
            "missing_final_verification",
            "error",
            f"Repaired units lack final verification: {missing_final_audits[:10]}",
        )

    governed_refs = {
        f"{surah}:{ayah}"
        for group in refrain_report.get("groups", {}).values()
        if group.get("source") in {"policy", "opus_resolution"}
        for surah, ayah in group.get("refs", [])
    }
    refrain_units = [
        unit
        for unit in units
        if any(f"{unit.surah}:{ayah}" in governed_refs for ayah in unit.expected_ayahs)
    ]
    missing_refrain_audits = [
        unit.unit_id
        for unit in refrain_units
        if not _stage_path(base, unit, "refrain_verification").exists()
    ]
    if missing_refrain_audits:
        issue(
            "missing_refrain_verification",
            "error",
            f"Governed refrain units lack verification: {missing_refrain_audits[:10]}",
        )

    missing_spoken = [
        unit.unit_id
        for unit in units
        if not _stage_path(base, unit, "spoken").exists()
    ]
    if missing_spoken:
        issue(
            "missing_spoken_audit",
            "error",
            f"Units lack spoken-English audit: {missing_spoken[:10]}",
        )

    fidelity_findings = [
        finding
        for stage in FINAL_FIDELITY_STAGES
        for finding in _stage_findings(base, units, stage)
    ]
    for finding in fidelity_findings:
        if finding.get("severity") in MAJOR_SEVERITIES:
            issue(
                "unresolved_fidelity_finding",
                "error",
                f"{finding.get('type')}: {finding.get('explanation')}",
                str(finding.get("ref")),
            )

    spoken_findings = _stage_findings(base, units, "spoken")
    translator_flags = _review_flags(conn, run_id)
    deterministic_spoken = [
        {"ref": ref, **finding}
        for ref, text in english.items()
        for finding in signature_phrase_findings(int(ref.split(":")[1]), text)
    ]
    review_queue = {
        "version": "production-review-queue-v1",
        "run_id": run_id,
        "policy": (
            "Review leads are not automatic defects or rewrite instructions. "
            "Reader notes require cited evidence adjudication."
        ),
        "fidelity_findings": fidelity_findings,
        "spoken_findings": spoken_findings,
        "deterministic_spoken_leads": deterministic_spoken,
        "translator_review_flags": translator_flags,
    }
    atomic_json(base / "REVIEW_QUEUE.json", review_queue)

    error_count = sum(item["severity"] == "error" for item in issues)
    warning_count = sum(item["severity"] == "warning" for item in issues)
    report = {
        "version": "production-quality-v1",
        "run_id": run_id,
        "passed": error_count == 0,
        "source": {"surahs": 114, "ayahs": len(verses)},
        "units": len(units),
        "translations": {"complete": len(db_rows), "empty": len(empty)},
        "refrains": {"groups": len(groups), "ayahs": refrain_ayahs},
        "review": {
            "fidelity_findings": len(fidelity_findings),
            "major_fidelity_findings": sum(
                finding.get("severity") in MAJOR_SEVERITIES
                for finding in fidelity_findings
            ),
            "spoken_findings": len(spoken_findings),
            "deterministic_spoken_leads": len(deterministic_spoken),
            "translator_flags": len(translator_flags),
        },
        "jobs": len(list((base / "jobs").glob("*.json"))),
        "usage": _usage_summary(base),
        "issue_counts": {"error": error_count, "warning": warning_count},
        "issues": issues,
    }
    atomic_json(base / "QA_REPORT.json", report)
    atomic_text(base / "QA_REPORT.md", _markdown_report(report))
    return report
