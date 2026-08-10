"""Post-production adjudication, QA, and release packaging."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from .config import DEFAULT_SOURCE_XML, OUTPUT_DIR, PROJECT_ROOT, WORK_DIR, file_sha256
from .db import utc_now
from .production_packets import atomic_json, atomic_text
from .publication import validate_publication
from .refrains import load_quran_xml, repeated_ayah_groups
from .spoken_english_v1 import signature_phrase_findings
from .validation import validate_run, validate_source


RELEASE_VERSION = "v2.4.1"
ADJUDICATIONS_PATH = PROJECT_ROOT / "data" / "evidence" / "release-adjudications-v2.4.1.json"
READING_NOTES_PATH = PROJECT_ROOT / "data" / "evidence" / "reading-notes-v2.4.1.json"
FORMULA_PATTERNS = (
    re.compile(r"\bglory\s+(?:be\s+)?to\b", re.IGNORECASE),
    re.compile(r"\bdeclar(?:e|es|ed|ing)\s+glory\b", re.IGNORECASE),
)
MAJOR_SEVERITIES = {"blocking", "significant"}


def run_base(run_id: str) -> Path:
    return WORK_DIR / "production-v2.4" / run_id


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _text_hash(rows: list[sqlite3.Row]) -> str:
    material = "\n".join(
        f"{row['verse_key']}\t{row['translation']}" for row in rows
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def translation_rows(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT s.global_ayah_number, s.verse_key, s.arabic_uthmani_min,
                   t.translation, t.raw_translation_json
            FROM source_ayahs s
            JOIN translations t USING (verse_key)
            WHERE t.run_id = ? AND t.status = 'complete'
            ORDER BY s.global_ayah_number
            """,
            (run_id,),
        )
    )


def validate_adjudications(payload: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    if payload.get("run_id") != run_id:
        raise ValueError(
            f"Adjudications target {payload.get('run_id')}, not requested run {run_id}"
        )
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Release adjudications must contain a non-empty decisions list")
    refs: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Every release adjudication must be an object")
        ref = decision.get("ref")
        action = decision.get("action")
        if not isinstance(ref, str) or ref in refs:
            raise ValueError(f"Duplicate or invalid adjudication ref: {ref}")
        refs.add(ref)
        if action == "change":
            if not all(
                isinstance(decision.get(field), str) and decision[field]
                for field in ("before", "after", "rationale")
            ):
                raise ValueError(f"Incomplete change adjudication at {ref}")
            if decision["before"] == decision["after"]:
                raise ValueError(f"No-op change adjudication at {ref}")
        elif action == "retain":
            if not all(
                isinstance(decision.get(field), str) and decision[field]
                for field in ("expected", "rationale")
            ):
                raise ValueError(f"Incomplete retain adjudication at {ref}")
        else:
            raise ValueError(f"Unknown adjudication action at {ref}: {action}")
    return decisions


def apply_release_adjudications(
    conn: sqlite3.Connection,
    run_id: str,
    adjudications_path: Path = ADJUDICATIONS_PATH,
) -> dict[str, Any]:
    payload = _read_json(adjudications_path)
    decisions = validate_adjudications(payload, run_id)
    decision_hash = file_sha256(adjudications_path)
    changed: list[str] = []
    already_applied: list[str] = []
    retained: list[str] = []
    now = utc_now()

    conn.execute("BEGIN")
    try:
        for decision in decisions:
            ref = decision["ref"]
            row = conn.execute(
                """
                SELECT translation, raw_translation_json
                FROM translations
                WHERE run_id = ? AND verse_key = ? AND status = 'complete'
                """,
                (run_id, ref),
            ).fetchone()
            if row is None:
                raise ValueError(f"Missing completed translation for adjudication {ref}")
            current = str(row["translation"])
            try:
                raw = json.loads(row["raw_translation_json"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid raw translation JSON at {ref}") from exc
            action = decision["action"]
            updated = current
            if action == "change":
                before = decision["before"]
                after = decision["after"]
                before_count = current.count(before)
                if before_count == 1:
                    updated = current.replace(before, after, 1)
                    changed.append(ref)
                elif before_count == 0 and after in current:
                    prior = raw.get("release_hardening", {})
                    if (
                        prior.get("adjudications_sha256") != decision_hash
                        or prior.get("translation") != current
                    ):
                        raise ValueError(
                            f"After-text exists without matching release provenance at {ref}"
                        )
                    already_applied.append(ref)
                else:
                    raise ValueError(
                        f"Exact before-text guard failed at {ref}: "
                        f"expected one occurrence of {before!r}, found {before_count}"
                    )
            else:
                expected = decision["expected"]
                if expected not in current:
                    raise ValueError(
                        f"Retain guard failed at {ref}: missing {expected!r}"
                    )
                retained.append(ref)

            raw["release_hardening"] = {
                "version": payload["version"],
                "action": action,
                "category": decision["category"],
                "source_translation": current if updated != current else raw.get(
                    "release_hardening", {}
                ).get("source_translation", current),
                "translation": updated,
                "rationale": decision["rationale"],
                "adjudications_sha256": decision_hash,
            }
            conn.execute(
                """
                UPDATE translations
                SET translation = ?, raw_translation_json = ?, updated_at = ?
                WHERE run_id = ? AND verse_key = ?
                """,
                (
                    updated,
                    json.dumps(raw, ensure_ascii=False, sort_keys=True),
                    now,
                    run_id,
                    ref,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    marker = {
        "version": payload["version"],
        "run_id": run_id,
        "adjudications_sha256": decision_hash,
        "decisions": len(decisions),
        "changed": len(changed),
        "already_applied": len(already_applied),
        "retained": len(retained),
        "changed_refs": changed,
        "already_applied_refs": already_applied,
        "retained_refs": retained,
        "completed_at": now,
    }
    atomic_json(run_base(run_id) / "RELEASE_HARDENING_COMPLETE.json", marker)
    return marker


def formula_violations(rows: list[sqlite3.Row]) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    for row in rows:
        text = str(row["translation"])
        for pattern in FORMULA_PATTERNS:
            match = pattern.search(text)
            if match:
                violations.append(
                    {
                        "ref": str(row["verse_key"]),
                        "match": match.group(0),
                        "translation": text,
                    }
                )
    return violations


def validate_reading_notes(
    payload: dict[str, Any],
    valid_refs: set[str],
) -> list[dict[str, Any]]:
    sources = payload.get("sources")
    notes = payload.get("notes")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Reading notes require a source registry")
    if not isinstance(notes, list) or not notes:
        raise ValueError("Reading notes require a non-empty notes list")
    seen: set[str] = set()
    for entry in notes:
        if not isinstance(entry, dict):
            raise ValueError("Every reading note must be an object")
        ref = entry.get("ref")
        if ref not in valid_refs or ref in seen:
            raise ValueError(f"Invalid or duplicate reading-note ref: {ref}")
        seen.add(ref)
        if not isinstance(entry.get("note"), str) or not entry["note"].strip():
            raise ValueError(f"Reading note is empty at {ref}")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"Reading note lacks evidence at {ref}")
        for evidence_ref in evidence:
            source_id = str(evidence_ref).split(":", 1)[0]
            if source_id not in sources:
                raise ValueError(
                    f"Unknown evidence source {source_id!r} in note {ref}"
                )
    return notes


def _finding_still_applies(finding: dict[str, Any], english: dict[str, str]) -> bool:
    ref = str(finding.get("ref") or "")
    where = finding.get("where")
    if ref not in english:
        return False
    return not isinstance(where, str) or not where or where in english[ref]


def build_final_review_queue(
    conn: sqlite3.Connection,
    run_id: str,
    notes_payload: dict[str, Any],
    notes_path: Path = READING_NOTES_PATH,
) -> dict[str, Any]:
    base = run_base(run_id)
    original_path = base / "REVIEW_QUEUE.json"
    original = _read_json(original_path)
    rows = translation_rows(conn, run_id)
    english = {str(row["verse_key"]): str(row["translation"]) for row in rows}

    fidelity = [
        finding
        for finding in original.get("fidelity_findings", [])
        if isinstance(finding, dict)
        and finding.get("severity") not in MAJOR_SEVERITIES
        and _finding_still_applies(finding, english)
    ]
    spoken = [
        finding
        for finding in original.get("spoken_findings", [])
        if isinstance(finding, dict) and _finding_still_applies(finding, english)
    ]
    deterministic = [
        {"ref": ref, **finding}
        for ref, text in english.items()
        for finding in signature_phrase_findings(int(ref.split(":", 1)[1]), text)
    ]
    translator_flags = [
        flag
        for flag in original.get("translator_review_flags", [])
        if isinstance(flag, dict) and str(flag.get("ref") or "") in english
    ]
    note_refs = [entry["ref"] for entry in notes_payload["notes"]]
    try:
        catalog = str(notes_path.relative_to(PROJECT_ROOT))
    except ValueError:
        catalog = str(notes_path)
    payload = {
        "version": "production-review-queue-v2.4.1-final",
        "run_id": run_id,
        "source_review_queue_sha256": file_sha256(original_path),
        "policy": (
            "This is the post-adjudication queue. Resolved major findings and stale "
            "exact-text findings are excluded. Remaining items are review leads, not "
            "automatic defects or rewrite instructions."
        ),
        "fidelity_findings": fidelity,
        "spoken_findings": spoken,
        "deterministic_spoken_leads": deterministic,
        "translator_review_flags": translator_flags,
        "reading_notes": {
            "status": "selectively_adjudicated",
            "catalog": catalog,
            "count": len(note_refs),
            "refs": note_refs,
            "scope": notes_payload["edition_scope"],
        },
        "counts": {
            "fidelity_findings": len(fidelity),
            "spoken_findings": len(spoken),
            "deterministic_spoken_leads": len(deterministic),
            "translator_review_flags": len(translator_flags),
            "reading_notes": len(note_refs),
        },
    }
    atomic_json(base / "FINAL_REVIEW_QUEUE.json", payload)
    return payload


def _final_qa_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Final Release Quality Report",
        "",
        f"- Verdict: **{'PASS' if report['passed'] else 'BLOCKED'}**",
        f"- Release: `{report['release_version']}`",
        f"- Run: `{report['run_id']}`",
        f"- Final text SHA-256: `{report['final_text_sha256']}`",
        f"- Source coverage: {report['source']['surahs']} surahs, {report['source']['ayahs']} ayahs",
        f"- Repeated-Arabic groups checked: {report['refrains']['groups_checked']}",
        f"- Release adjudications: {report['adjudications']['decisions']} decisions "
        f"({report['adjudications']['changed_or_previously_applied']} changed, "
        f"{report['adjudications']['retained']} retained)",
        f"- Evidence-adjudicated reading notes: {report['reading_notes']['count']}",
        "",
        "## Gate Issues",
        "",
    ]
    if report["issues"]:
        for issue in report["issues"]:
            ref = f" ({issue['ref']})" if issue.get("ref") else ""
            lines.append(f"- **{issue['severity'].upper()}** `{issue['code']}`{ref}: {issue['message']}")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Status Of Earlier Artifacts",
            "",
            "`QA_REPORT.json` and `QA_REPORT.md` are immutable pre-remediation records. "
            "They correctly describe the earlier blocked checkpoint, not the released text. "
            "This final report, the production completion marker, and the remediation record "
            "are the authoritative release status.",
            "",
            "## Governance",
            "",
            "The listening and reader editions remain note-free. The annotated reading edition "
            "contains only the selectively adjudicated notes in the tracked evidence catalog. "
            "Open review leads remain preserved in `FINAL_REVIEW_QUEUE.json`; they are not "
            "silently presented as defects.",
            "",
        ]
    )
    return "\n".join(lines)


def run_final_quality_gate(
    conn: sqlite3.Connection,
    run_id: str,
    marker: dict[str, Any],
    notes_payload: dict[str, Any],
    notes_path: Path = READING_NOTES_PATH,
) -> dict[str, Any]:
    base = run_base(run_id)
    rows = translation_rows(conn, run_id)
    valid_refs = {str(row["verse_key"]) for row in rows}
    notes = validate_reading_notes(notes_payload, valid_refs)
    queue = build_final_review_queue(
        conn,
        run_id,
        notes_payload,
        notes_path,
    )
    issues: list[dict[str, str]] = []

    def add_issue(code: str, message: str, ref: str | None = None) -> None:
        record = {"code": code, "severity": "error", "message": message}
        if ref:
            record["ref"] = ref
        issues.append(record)

    complete_path = base / "PRODUCTION_COMPLETE.json"
    if not complete_path.exists():
        add_issue("production_completion", "PRODUCTION_COMPLETE.json is missing")
    else:
        complete = _read_json(complete_path)
        if not complete.get("quality", {}).get("passed"):
            add_issue("production_completion", "Production completion is not quality-passed")

    for validation_issue in validate_source(conn) + validate_run(conn, run_id):
        if validation_issue.severity == "error" or validation_issue.message.startswith(
            "Banned/jargon term"
        ):
            add_issue(
                "database_validation",
                validation_issue.message,
                validation_issue.ref,
            )

    for violation in formula_violations(rows):
        add_issue(
            "inherited_subhana_formula",
            f"Inherited formula remains: {violation['match']}",
            violation["ref"],
        )

    verses = load_quran_xml(DEFAULT_SOURCE_XML)
    english = {str(row["verse_key"]): str(row["translation"]) for row in rows}
    checked_groups = 0
    for group in repeated_ayah_groups(verses).values():
        refs = [f"{surah}:{ayah}" for surah, ayah in group["refs"]]
        renderings = {english.get(ref) for ref in refs}
        checked_groups += 1
        if len(renderings) != 1 or None in renderings:
            add_issue(
                "refrain_divergence",
                f"Identical Arabic has {len(renderings)} English renderings",
                ", ".join(refs),
            )

    for publication_issue in validate_publication(conn, run_id):
        if publication_issue.severity == "error" or publication_issue.message.startswith(
            "Banned/jargon term"
        ):
            add_issue(
                "publication_validation",
                publication_issue.message,
                publication_issue.ref,
            )

    report = {
        "version": "final-release-quality-v2.4.1",
        "release_version": RELEASE_VERSION,
        "run_id": run_id,
        "passed": not issues,
        "final_text_sha256": _text_hash(rows),
        "source": {"surahs": 114, "ayahs": len(rows)},
        "refrains": {"groups_checked": checked_groups},
        "formula_gate": {"violations": len(formula_violations(rows))},
        "adjudications": {
            "decisions": marker["decisions"],
            "changed_or_previously_applied": marker["changed"] + marker["already_applied"],
            "retained": marker["retained"],
            "sha256": marker["adjudications_sha256"],
        },
        "reading_notes": {
            "count": len(notes),
            "sha256": file_sha256(notes_path),
            "scope": notes_payload["edition_scope"],
        },
        "review_queue": queue["counts"],
        "historical_qa": {
            "status": "immutable_pre_remediation_record",
            "qa_report_sha256": file_sha256(base / "QA_REPORT.json"),
            "production_complete_sha256": file_sha256(complete_path),
        },
        "issues": issues,
        "completed_at": utc_now(),
    }
    atomic_json(base / "FINAL_QA_REPORT.json", report)
    atomic_text(base / "FINAL_QA_REPORT.md", _final_qa_markdown(report))
    atomic_text(
        base / "CURRENT_STATUS.md",
        "# Current Release Status\n\n"
        f"The authoritative release QA is `FINAL_QA_REPORT.md`: "
        f"**{'PASS' if report['passed'] else 'BLOCKED'}**.\n\n"
        "`QA_REPORT.md` is retained unchanged as the historical pre-remediation checkpoint.\n",
    )
    return report


def _copy(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required release artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def create_release_package(
    run_id: str,
    qa_report: dict[str, Any],
    *,
    release_version: str = RELEASE_VERSION,
    notes_path: Path = READING_NOTES_PATH,
) -> dict[str, Any]:
    if not qa_report.get("passed"):
        raise ValueError("Refusing to package a release that failed final QA")
    base = run_base(run_id)
    release_name = f"quran-translation-{release_version}"
    release_dir = OUTPUT_DIR / "release" / release_name
    release_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "quran-translation.json": OUTPUT_DIR / "quran-translation.json",
        "quran-translation.md": OUTPUT_DIR / "quran-translation.md",
        "quran-bilingual.md": OUTPUT_DIR / "quran-bilingual.md",
        "quran-listening-edition.json": OUTPUT_DIR / "publication" / "quran-publication.json",
        "quran-listening-edition.md": OUTPUT_DIR / "publication" / "quran-publication.md",
        "quran-translation-book.pdf": OUTPUT_DIR / "book" / "quran-translation-book.pdf",
        "quran-reader-edition.pdf": OUTPUT_DIR / "book" / "quran-translation-reader-edition.pdf",
        "quran-annotated-reading-edition.pdf": OUTPUT_DIR / "book" / "quran-translation-annotated-reading-edition.pdf",
        "quran-translation-book.inspection.json": OUTPUT_DIR / "book" / "quran-translation-book.inspection.json",
        "quran-reader-edition.inspection.json": OUTPUT_DIR / "book" / "quran-translation-reader-edition.inspection.json",
        "quran-annotated-reading-edition.inspection.json": OUTPUT_DIR / "book" / "quran-translation-annotated-reading-edition.inspection.json",
        "FINAL_QA_REPORT.json": base / "FINAL_QA_REPORT.json",
        "FINAL_QA_REPORT.md": base / "FINAL_QA_REPORT.md",
        "FINAL_REVIEW_QUEUE.json": base / "FINAL_REVIEW_QUEUE.json",
        "release-adjudications.json": ADJUDICATIONS_PATH,
        "reading-notes.json": notes_path,
    }
    expected_names = {
        *artifacts,
        "README.md",
        "MANIFEST.json",
        "SHA256SUMS.txt",
    }
    unexpected = {
        path.name for path in release_dir.iterdir() if path.name not in expected_names
    }
    if unexpected:
        raise ValueError(
            f"Release directory contains unexpected stale artifacts: {sorted(unexpected)}"
        )
    for name, source in artifacts.items():
        _copy(source, release_dir / name)

    readme = """# The Quran - Evidence-Audited Modern English Translation

This is the v2.4.1 translation release. It contains three English reading formats:

- `quran-translation-book.pdf`: printable ayah-numbered book.
- `quran-reader-edition.pdf`: note-free paragraph reading edition.
- `quran-annotated-reading-edition.pdf`: ayah-numbered edition with selectively adjudicated evidence notes.

The source text is Tanzil Quran Text, Uthmani Minimal, Version 1.1. Quranic Arabic
Corpus morphology supports source analysis. Claude Opus 4.6 produced the principal
English drafts and revisions; Gemini 3.1 Pro provided source-grounded criticism and
verification. Deterministic gates checked corpus coverage, repeated-Arabic invariance,
registered terminology, and release-specific formula rules. Final QA and editorial
adjudications are included in this directory.

The annotated edition is not an exhaustive commentary. It contains only notes that
passed the project's evidence-adjudication policy. The listening and reader editions
remain note-free.

The older `output/release/quran-translation-v2` directory is a legacy April/May build
and is not the source for this release or for future audio generation.
"""
    atomic_text(release_dir / "README.md", readme)

    artifact_hashes = {
        path.name: file_sha256(path)
        for path in sorted(release_dir.iterdir())
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS.txt"}
    }
    manifest = {
        "version": f"translation-release-{release_version}",
        "run_id": run_id,
        "release_name": release_name,
        "created_at": utc_now(),
        "qa_passed": True,
        "final_text_sha256": qa_report["final_text_sha256"],
        "source": "Tanzil Quran Text, Uthmani Minimal, Version 1.1",
        "source_sha256": file_sha256(DEFAULT_SOURCE_XML),
        "artifacts": artifact_hashes,
    }
    atomic_json(release_dir / "MANIFEST.json", manifest)
    checksum_paths = sorted(
        path
        for path in release_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    checksum_text = "".join(
        f"{file_sha256(path)}  {path.name}\n" for path in checksum_paths
    )
    atomic_text(release_dir / "SHA256SUMS.txt", checksum_text)

    record = {
        "version": f"translation-release-record-{release_version}",
        "run_id": run_id,
        "release_path": str(release_dir.relative_to(PROJECT_ROOT)),
        "qa_passed": True,
        "final_text_sha256": qa_report["final_text_sha256"],
        "manifest_sha256": file_sha256(release_dir / "MANIFEST.json"),
        "checksums_sha256": file_sha256(release_dir / "SHA256SUMS.txt"),
        "artifact_count": len(checksum_paths),
        "release_adjudications_sha256": file_sha256(ADJUDICATIONS_PATH),
        "reading_notes_sha256": file_sha256(notes_path),
        "created_at": utc_now(),
    }
    releases_dir = PROJECT_ROOT / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(releases_dir / f"quran-translation-{release_version}.json", record)
    return {**record, "release_dir": str(release_dir)}
