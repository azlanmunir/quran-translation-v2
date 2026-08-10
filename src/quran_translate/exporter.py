"""Export validated translation data into reader-facing artifacts."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .config import OUTPUT_DIR
from .metadata import SURAHS


def slugify(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def translation_rows(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT
                s.*,
                t.translation,
                t.status AS translation_status
            FROM source_ayahs s
            LEFT JOIN translations t
              ON t.verse_key = s.verse_key
             AND t.run_id = ?
            ORDER BY s.global_ayah_number
            """,
            (run_id,),
        )
    )


def export_json(conn: sqlite3.Connection, run_id: str, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = translation_rows(conn, run_id)
    payload = {
        "run_id": run_id,
        "source": "Tanzil Quran Text, Uthmani Minimal, Version 1.1",
        "ayahs": [
            {
                "ref": row["verse_key"],
                "surah": int(row["surah_number"]),
                "ayah": int(row["ayah_number"]),
                "surah_name_ar": row["surah_name_ar"],
                "surah_name_en": row["surah_name_en"],
                "surah_meaning_en": row["surah_meaning_en"],
                "arabic_uthmani_min": row["arabic_uthmani_min"],
                "bismillah": row["bismillah"],
                "translation": row["translation"],
            }
            for row in rows
        ],
    }
    path = output_dir / "quran-translation.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def export_markdown(conn: sqlite3.Connection, run_id: str, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = translation_rows(conn, run_id)
    by_surah: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_surah.setdefault(int(row["surah_number"]), []).append(row)

    path = output_dir / "quran-translation.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# The Quran - Evidence-Audited Modern English Translation\n\n")
        handle.write(f"*Translation run: `{run_id}`*\n\n")
        handle.write("---\n\n")
        for info in SURAHS:
            handle.write(f"## {info.heading}\n\n")
            for row in by_surah.get(info.number, []):
                ref = row["verse_key"]
                translation = row["translation"] or f"[UNTRANSLATED {ref}]"
                handle.write(f'<a id="{ref.replace(":", "-")}"></a>\n\n')
                handle.write(f"**{ref}** {translation}\n\n")
            handle.write("---\n\n")
    return path


def export_bilingual_markdown(conn: sqlite3.Connection, run_id: str, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = translation_rows(conn, run_id)
    by_surah: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_surah.setdefault(int(row["surah_number"]), []).append(row)

    path = output_dir / "quran-bilingual.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# The Quran - Bilingual Evidence-Audited Translation\n\n")
        handle.write(f"*Translation run: `{run_id}`*\n\n")
        handle.write("---\n\n")
        for info in SURAHS:
            handle.write(f"## {info.heading}\n\n")
            surah_rows = by_surah.get(info.number, [])
            if surah_rows and surah_rows[0]["bismillah"] and info.number != 1:
                handle.write(f'<p dir="rtl" lang="ar">{surah_rows[0]["bismillah"]}</p>\n\n')
            for row in surah_rows:
                ref = row["verse_key"]
                translation = row["translation"] or f"[UNTRANSLATED {ref}]"
                handle.write(f'<a id="{ref.replace(":", "-")}"></a>\n\n')
                handle.write(f"**{ref}**\n\n")
                handle.write(f'<p dir="rtl" lang="ar">{row["arabic_uthmani_min"]}</p>\n\n')
                handle.write(f"{translation}\n\n")
            handle.write("---\n\n")
    return path


def export_surah_markdown(conn: sqlite3.Connection, run_id: str, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    surah_dir = output_dir / "surahs"
    surah_dir.mkdir(parents=True, exist_ok=True)
    rows = translation_rows(conn, run_id)
    by_surah: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_surah.setdefault(int(row["surah_number"]), []).append(row)

    paths: list[Path] = []
    for info in SURAHS:
        path = surah_dir / f"{info.number:03d}-{slugify(info.transliteration)}.md"
        paths.append(path)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"# {info.heading}\n\n")
            handle.write(f"*Translation run: `{run_id}`*\n\n")
            for row in by_surah.get(info.number, []):
                ref = row["verse_key"]
                translation = row["translation"] or f"[UNTRANSLATED {ref}]"
                handle.write(f"**{ref}** {translation}\n\n")
    return paths


def export_glossary(conn: sqlite3.Connection, run_id: str, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(
        conn.execute(
            """
            SELECT
                lower(trim(term)) AS term_key,
                term,
                root,
                rendering,
                physical_reality,
                definition,
                COUNT(*) AS count,
                group_concat(DISTINCT verse_key) AS refs
            FROM word_bank_entries
            WHERE run_id = ?
            GROUP BY term_key, root, rendering, physical_reality, definition
            ORDER BY term_key
            """,
            (run_id,),
        )
    )
    path = output_dir / "quran-glossary.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Quran Translation Glossary\n\n")
        handle.write(f"*Translation run: `{run_id}`*\n\n")
        handle.write(f"**Entries:** {len(rows)}\n\n")
        handle.write("---\n\n")
        for row in rows:
            rendering = f" -> {row['rendering']}" if row["rendering"] else ""
            handle.write(f"## {row['term']}{rendering}\n\n")
            if row["root"]:
                handle.write(f"**Root:** {row['root']}\n\n")
            if row["physical_reality"]:
                handle.write(f"**Editorial background:** {row['physical_reality']}\n\n")
            else:
                handle.write(f"{row['definition']}\n\n")
            refs = (row["refs"] or "").split(",")
            sample_refs = ", ".join(refs[:12])
            if len(refs) > 12:
                sample_refs += ", ..."
            handle.write(f"*Refs:* {sample_refs}\n\n")
    return path


def export_all(conn: sqlite3.Connection, run_id: str, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    paths = [
        export_json(conn, run_id, output_dir),
        export_markdown(conn, run_id, output_dir),
        export_bilingual_markdown(conn, run_id, output_dir),
        export_glossary(conn, run_id, output_dir),
    ]
    paths.extend(export_surah_markdown(conn, run_id, output_dir))
    return paths
