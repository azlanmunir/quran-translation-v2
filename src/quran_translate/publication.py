"""Publication-layer cleanup and export helpers."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import OUTPUT_DIR
from .db import utc_now
from .metadata import SURAHS
from .validation import BANNED_TERMS, PRODUCTION_V24_BANNED_TERMS, ValidationIssue


PUBLICATION_DIR = OUTPUT_DIR / "publication"
PUBLICATION_BANNED_TERMS = tuple(sorted(set(BANNED_TERMS + ("messengers",))))


def is_production_v24(conn: sqlite3.Connection, run_id: str) -> bool:
    row = conn.execute(
        "SELECT prompt_version FROM translation_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return bool(row and str(row["prompt_version"]).startswith("production-v2.4"))


@dataclass(frozen=True)
class CleanupResult:
    text: str
    edits: tuple[str, ...]


def _case_word(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_word(text: str, word: str, replacement: str, label: str, edits: list[str]) -> str:
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.I)

    def repl(match: re.Match[str]) -> str:
        return _case_word(replacement, match.group(0))

    new_text, count = pattern.subn(repl, text)
    if count:
        edits.append(f"{label}: {count}")
    return new_text


def _replace_phrase(text: str, old: str, new: str, label: str, edits: list[str]) -> str:
    count = text.count(old)
    if not count:
        return text
    edits.append(f"{label}: {count}")
    return text.replace(old, new)


LEST_PHRASES: tuple[tuple[str, str], ...] = (
    ("lest you become", "so you do not become"),
    ("lest you swerve", "so you do not swerve"),
    ("lest you stray", "so you do not stray"),
    ("Lest you say", "So you do not say"),
    ("lest you say", "so you do not say"),
    ("Or lest you say", "Or so you do not say"),
    ("lest they lure", "so they do not lure"),
    ("lest a soul be locked away", "so no soul is locked away"),
    ("lest they hurl", "so they do not hurl"),
    ("lest they set a trap", "so they do not set a trap"),
    ("lest a walled enclosure be dropped", "against a walled enclosure being dropped"),
    ("lest an agonizing strike seize you", "so no agonizing strike seizes you"),
    ("lest you lose your nerve", "so you do not lose your nerve"),
    ("lest he burn them", "so he does not burn them"),
    ("lest a near strike seize you", "so no near strike seizes you"),
    ("lest it capsize with you", "so it does not capsize with you"),
    ("lest a foot slip", "so no foot slips"),
    ("lest you be thrown", "so you are not thrown"),
    ("lest you plummet", "so you do not plummet"),
    ("lest He scrape", "so He does not scrape"),
    ("lest My rage descends", "so My rage does not descend"),
    ("lest you break", "so you do not break"),
    ("lest it fall", "so it does not fall"),
    ("lest they crowd", "so they do not crowd"),
    ("lest you ever return", "so you never return"),
    ("lest the crushing strike of a massive day grip you", "so the crushing strike of a massive day does not grip you"),
    ("lest a smelting fire strike them, or a crushing agony strike them", "so no smelting fire strikes them, and no crushing agony strikes them"),
    ("lest a smelting fire strike them", "so no smelting fire strikes them"),
    ("lest Solomon and his armies crush you", "so Solomon and his armies do not crush you"),
    ("lest the one with rot in his core stretches his neck", "so the one with rot in his core does not stretch his neck"),
    ("lest they slip away", "so they do not slip away"),
    ("lest it make you lose", "so it does not make you lose"),
    ("Lest a breathing self should say", "So no breathing self says"),
    ("Or lest it should say", "Or so it does not say"),
    ("lest you crush me", "so you do not crush me"),
    ("lest you trample them", "so you do not trample them"),
    ("lest your actions collapse", "so your actions do not collapse"),
    ("lest you strike", "so you do not strike"),
)


SPECIFIC_PHRASES: tuple[tuple[str, str, str], ...] = (
    ("Ta, Sin, Mim.", "Ta, Seen, Mim.", "letter-name Sin"),
    ("Ta-Sin.", "Ta-Seen.", "letter-name Sin"),
    ("Ya-Sin.", "Ya-Seen.", "letter-name Sin"),
    ("Ayn. Sin. Qaf.", "Ayn. Seen. Qaf.", "letter-name Sin"),
    ("Written Decree", "Written Command", "decree"),
    ("written decree", "written command", "decree"),
    ("carved the decree", "carved the command", "decree"),
    ("cut the decree", "cut the command", "decree"),
    ("cuts the decree", "cuts the command", "decree"),
    ("Hell a tight confinement", "Jahannam a tight confinement", "hell"),
    ("any depression or any mound", "any dip or any mound", "depression"),
    ("from the depression in the ground", "from the low hollow in the ground", "depression"),
    ("despite loving it", "even while clinging to it", "loving"),
)


WORD_REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    ("messengers", "envoys", "messenger"),
    ("messenger", "envoy", "messenger"),
    ("thus", "so", "thus"),
    ("decree", "command", "decree"),
    ("bounty", "surplus", "bounty"),
    ("grace", "fierce shelter", "grace"),
    ("compassionate", "womb-bound", "therapy/greeting-card word"),
    ("caring", "protective", "therapy/greeting-card word"),
    ("loving", "bound close", "therapy/greeting-card word"),
    ("gracious", "open-handed", "therapy/greeting-card word"),
)


def cleanup_translation(text: str) -> CleanupResult:
    edits: list[str] = []
    cleaned = text
    for old, new, label in SPECIFIC_PHRASES:
        cleaned = _replace_phrase(cleaned, old, new, label, edits)
    for old, new in LEST_PHRASES:
        cleaned = _replace_phrase(cleaned, old, new, "lest", edits)
    for word, replacement, label in WORD_REPLACEMENTS:
        cleaned = _replace_word(cleaned, word, replacement, label, edits)

    if re.search(r"\blest\b", cleaned, re.I):
        cleaned, count = re.subn(r"\b[Ll]est\b", "so that not", cleaned)
        if count:
            edits.append(f"lest fallback: {count}")

    return CleanupResult(cleaned, tuple(edits))


def source_translation_rows(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT
                s.*,
                t.translation
            FROM source_ayahs s
            JOIN translations t
              ON t.verse_key = s.verse_key
             AND t.run_id = ?
             AND t.status = 'complete'
            ORDER BY s.global_ayah_number
            """,
            (run_id,),
        )
    )


def publication_rows(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT
                s.*,
                p.publication_translation AS translation,
                p.source_translation,
                p.changed,
                p.edits_json
            FROM source_ayahs s
            JOIN publication_translations p
              ON p.verse_key = s.verse_key
             AND p.run_id = ?
            ORDER BY s.global_ayah_number
            """,
            (run_id,),
        )
    )


def build_publication_layer(conn: sqlite3.Connection, run_id: str) -> dict[str, object]:
    rows = source_translation_rows(conn, run_id)
    identity_policy = is_production_v24(conn, run_id)
    now = utc_now()
    edit_counter: Counter[str] = Counter()
    changed_count = 0

    with conn:
        conn.execute("DELETE FROM publication_translations WHERE run_id = ?", (run_id,))
        for row in rows:
            result = (
                CleanupResult(str(row["translation"]), ())
                if identity_policy
                else cleanup_translation(str(row["translation"]))
            )
            changed = int(result.text != row["translation"])
            if changed:
                changed_count += 1
            for edit in result.edits:
                edit_counter[edit.split(":", 1)[0]] += 1
            conn.execute(
                """
                INSERT INTO publication_translations (
                    run_id,
                    verse_key,
                    source_translation,
                    publication_translation,
                    changed,
                    edits_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    row["verse_key"],
                    row["translation"],
                    result.text,
                    changed,
                    json.dumps(result.edits, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    return {
        "run_id": run_id,
        "ayahs": len(rows),
        "changed_ayahs": changed_count,
        "edit_categories": dict(sorted(edit_counter.items())),
        "policy": (
            "identity_from_audited_production_text"
            if identity_policy
            else "legacy_audited_cleanup"
        ),
    }


def validate_publication(conn: sqlite3.Connection, run_id: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    source_total = int(conn.execute("SELECT COUNT(*) AS count FROM source_ayahs").fetchone()["count"])
    publication_total = int(
        conn.execute(
            "SELECT COUNT(*) AS count FROM publication_translations WHERE run_id = ?",
            (run_id,),
        ).fetchone()["count"]
    )
    if publication_total != source_total:
        issues.append(
            ValidationIssue(
                "publication",
                "error",
                f"Expected {source_total} publication translations, found {publication_total}",
            )
        )

    banned_terms = (
        PRODUCTION_V24_BANNED_TERMS
        if is_production_v24(conn, run_id)
        else PUBLICATION_BANNED_TERMS
    )
    banned_re = re.compile(
        r"\b(" + "|".join(re.escape(term) for term in banned_terms) + r")\b", re.I
    )
    bracket_re = re.compile(r"\[[^\]]+\]")
    for row in publication_rows(conn, run_id):
        text = row["translation"]
        banned = sorted({match.group(0).lower() for match in banned_re.finditer(text)})
        if banned:
            issues.append(
                ValidationIssue(
                    "publication",
                    "warning",
                    "Banned/jargon term(s): " + ", ".join(banned),
                    ref=row["verse_key"],
                )
            )
        if bracket_re.search(text):
            issues.append(
                ValidationIssue(
                    "publication",
                    "warning",
                    "Bracketed text appears inside publication translation",
                    ref=row["verse_key"],
                )
            )
    return issues


def export_publication_json(conn: sqlite3.Connection, run_id: str, output_dir: Path = PUBLICATION_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = publication_rows(conn, run_id)
    payload = {
        "run_id": run_id,
        "source": "Tanzil Quran Text, Uthmani Minimal, Version 1.1",
        "publication_policy": (
            "Audited v2.4 production text is preserved verbatim in the publication layer."
            if is_production_v24(conn, run_id)
            else "Raw Gemini output is preserved in translations; this publication layer applies audited style cleanup only."
        ),
        "ayahs": [
            {
                "ref": row["verse_key"],
                "surah": int(row["surah_number"]),
                "ayah": int(row["ayah_number"]),
                "surah_name_ar": row["surah_name_ar"],
                "surah_name_en": row["surah_name_en"],
                "surah_meaning_en": row["surah_meaning_en"],
                "translation": row["translation"],
            }
            for row in rows
        ],
    }
    path = output_dir / "quran-publication.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def export_publication_markdown(conn: sqlite3.Connection, run_id: str, output_dir: Path = PUBLICATION_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = publication_rows(conn, run_id)
    by_surah: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_surah.setdefault(int(row["surah_number"]), []).append(row)

    path = output_dir / "quran-publication.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# The Quran - Historical Philological Translation\n\n")
        handle.write(f"*Publication layer for translation run: `{run_id}`*\n\n")
        handle.write("---\n\n")
        for info in SURAHS:
            handle.write(f"## {info.number:03d}. {info.transliteration}\n\n")
            for row in by_surah.get(info.number, []):
                handle.write(f"**{row['verse_key']}** {row['translation']}\n\n")
            handle.write("---\n\n")
    return path


def export_publication_edits(conn: sqlite3.Connection, run_id: str, output_dir: Path = PUBLICATION_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(
        conn.execute(
            """
            SELECT verse_key, source_translation, publication_translation, edits_json
            FROM publication_translations
            WHERE run_id = ? AND changed = 1
            ORDER BY CAST(substr(verse_key, 1, instr(verse_key, ':') - 1) AS INTEGER),
                     CAST(substr(verse_key, instr(verse_key, ':') + 1) AS INTEGER)
            """,
            (run_id,),
        )
    )
    payload = {
        "run_id": run_id,
        "changed_ayahs": len(rows),
        "edits": [
            {
                "ref": row["verse_key"],
                "source_translation": row["source_translation"],
                "publication_translation": row["publication_translation"],
                "edits": json.loads(row["edits_json"]),
            }
            for row in rows
        ],
    }
    path = output_dir / "publication-edits.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def export_publication_all(conn: sqlite3.Connection, run_id: str, output_dir: Path = PUBLICATION_DIR) -> list[Path]:
    return [
        export_publication_json(conn, run_id, output_dir),
        export_publication_markdown(conn, run_id, output_dir),
        export_publication_edits(conn, run_id, output_dir),
    ]
