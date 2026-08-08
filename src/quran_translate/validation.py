"""Source and translation-run validation."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from .db import utc_now
from .metadata import SURAHS, SURAH_BY_NUMBER


BANNED_TERMS = (
    "prayer",
    "piety",
    "sin",
    "heaven",
    "hell",
    "verse",
    "charity",
    "hypocrite",
    "infidel",
    "lord",
    "religion",
    "angel",
    "angels",
    "messenger",
    "praise",
    "worship",
    "ingrate",
    "verily",
    "lest",
    "lo",
    "imminent",
    "chastisement",
    "recompense",
    "thus",
    "sovereign",
    "dominion",
    "decree",
    "bounty",
    "grace",
    "compassionate",
    "caring",
    "loving",
    "gracious",
    "anxiety",
    "depression",
    "groin",
    "genitals",
)

PRODUCTION_V24_BANNED_TERMS = (
    "verily",
    "lo",
    "thus",
    "lest",
    "chastisement",
    "recompense",
    "ingrate",
)


@dataclass(frozen=True)
class ValidationIssue:
    scope: str
    severity: str
    message: str
    ref: str | None = None


def validate_source(conn: sqlite3.Connection) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    total = conn.execute("SELECT COUNT(*) AS count FROM source_ayahs").fetchone()["count"]
    surahs = conn.execute("SELECT COUNT(DISTINCT surah_number) AS count FROM source_ayahs").fetchone()["count"]

    if int(surahs) != 114:
        issues.append(ValidationIssue("source", "error", f"Expected 114 surahs, found {surahs}"))
    if int(total) != 6236:
        issues.append(ValidationIssue("source", "error", f"Expected 6236 ayahs, found {total}"))

    first = conn.execute("SELECT verse_key FROM source_ayahs ORDER BY global_ayah_number LIMIT 1").fetchone()
    last = conn.execute("SELECT verse_key FROM source_ayahs ORDER BY global_ayah_number DESC LIMIT 1").fetchone()
    if not first or first["verse_key"] != "1:1":
        issues.append(ValidationIssue("source", "error", "First ayah is not 1:1"))
    if not last or last["verse_key"] != "114:6":
        issues.append(ValidationIssue("source", "error", "Last ayah is not 114:6"))

    counts = {
        int(row["surah_number"]): int(row["count"])
        for row in conn.execute(
            """
            SELECT surah_number, COUNT(*) AS count
            FROM source_ayahs
            GROUP BY surah_number
            """
        )
    }
    for info in SURAHS:
        actual = counts.get(info.number)
        if actual != info.ayah_count:
            issues.append(
                ValidationIssue(
                    "source",
                    "error",
                    f"Expected {info.ayah_count} ayahs in surah {info.number}, found {actual}",
                    ref=str(info.number),
                )
            )

    bismillah_rows = list(
        conn.execute(
            """
            SELECT surah_number, ayah_number, bismillah
            FROM source_ayahs
            WHERE bismillah IS NOT NULL
            ORDER BY surah_number, ayah_number
            """
        )
    )
    bismillah_surahs = {int(row["surah_number"]) for row in bismillah_rows}
    expected_bismillah_surahs = set(range(2, 115)) - {9}
    if bismillah_surahs != expected_bismillah_surahs or any(
        int(row["ayah_number"]) != 1 for row in bismillah_rows
    ):
        issues.append(
            ValidationIssue(
                "source",
                "error",
                "Opening Bismillah markers do not match the Tanzil policy "
                "(surahs 2-8 and 10-114, ayah 1 attributes only)",
            )
        )

    return issues


def validate_run(conn: sqlite3.Connection, run_id: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    source_total = conn.execute("SELECT COUNT(*) AS count FROM source_ayahs").fetchone()["count"]
    translated_total = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM translations
        WHERE run_id = ? AND status = 'complete'
        """,
        (run_id,),
    ).fetchone()["count"]
    if int(translated_total) != int(source_total):
        issues.append(
            ValidationIssue(
                "run",
                "error",
                f"Expected {source_total} complete translations, found {translated_total}",
            )
        )

    failed = list(
        conn.execute(
            """
            SELECT batch_id, last_error
            FROM translation_batches
            WHERE run_id = ? AND status = 'failed'
            ORDER BY batch_index
            """,
            (run_id,),
        )
    )
    for row in failed:
        issues.append(
            ValidationIssue(
                "batch",
                "error",
                f"Batch failed: {row['last_error']}",
                ref=row["batch_id"],
            )
        )

    missing = list(
        conn.execute(
            """
            SELECT s.verse_key
            FROM source_ayahs s
            LEFT JOIN translations t
              ON t.verse_key = s.verse_key
             AND t.run_id = ?
             AND t.status = 'complete'
            WHERE t.verse_key IS NULL
            ORDER BY s.global_ayah_number
            LIMIT 100
            """,
            (run_id,),
        )
    )
    for row in missing:
        issues.append(ValidationIssue("translation", "error", "Missing translation", ref=row["verse_key"]))

    run = conn.execute(
        "SELECT prompt_version FROM translation_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    banned_terms = (
        PRODUCTION_V24_BANNED_TERMS
        if run and str(run["prompt_version"]).startswith("production-v2.4")
        else BANNED_TERMS
    )
    banned_re = re.compile(
        r"\b(" + "|".join(re.escape(term) for term in banned_terms) + r")\b", re.I
    )
    bracket_re = re.compile(r"\[[^\]]+\]")
    for row in conn.execute(
        """
        SELECT verse_key, translation
        FROM translations
        WHERE run_id = ? AND status = 'complete'
        """,
        (run_id,),
    ):
        text = row["translation"]
        banned = sorted({match.group(0).lower() for match in banned_re.finditer(text)})
        if banned:
            issues.append(
                ValidationIssue(
                    "translation",
                    "warning",
                    "Banned/jargon term(s): " + ", ".join(banned),
                    ref=row["verse_key"],
                )
            )
        if bracket_re.search(text):
            issues.append(
                ValidationIssue(
                    "translation",
                    "warning",
                    "Bracketed text appears inside translation",
                    ref=row["verse_key"],
                )
            )

    return issues


def persist_issues(
    conn: sqlite3.Connection,
    issues: list[ValidationIssue],
    run_id: str | None = None,
    scope_prefix: str | None = None,
) -> None:
    now = utc_now()
    with conn:
        if scope_prefix:
            conn.execute(
                "DELETE FROM validation_issues WHERE run_id IS ? AND scope LIKE ?",
                (run_id, f"{scope_prefix}%"),
            )
        elif run_id is not None:
            conn.execute(
                "DELETE FROM validation_issues WHERE run_id = ?",
                (run_id,),
            )
        else:
            conn.execute("DELETE FROM validation_issues WHERE run_id IS NULL")
        for issue in issues:
            conn.execute(
                """
                INSERT INTO validation_issues (run_id, scope, ref, severity, message, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, issue.scope, issue.ref, issue.severity, issue.message, now),
            )
