"""Import Tanzil XML without invalidating dependent translation runs."""

from __future__ import annotations

import hashlib
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import DEFAULT_SOURCE_XML
from .db import utc_now
from .metadata import SURAHS, surah_info


def import_tanzil_xml(conn: sqlite3.Connection, xml_path: Path = DEFAULT_SOURCE_XML) -> dict[str, int | str]:
    material = xml_path.read_bytes()
    digest = hashlib.sha256(material).hexdigest()
    root = ET.fromstring(material)
    if root.tag != "quran":
        raise ValueError(f"Expected <quran> root, got <{root.tag}>")
    rows = []
    suras = root.findall("sura")
    if [int(node.attrib["index"]) for node in suras] != list(range(1, 115)):
        raise ValueError("Source must contain all 114 surahs in canonical order")
    for sura in suras:
        number = int(sura.attrib["index"])
        info = surah_info(number)
        ayahs = sura.findall("aya")
        if [int(aya.attrib["index"]) for aya in ayahs] != list(range(1, info.ayah_count + 1)):
            raise ValueError(f"Source ayah coverage failed for surah {number}")
        for aya in ayahs:
            text = aya.attrib["text"]
            if not text.strip():
                raise ValueError("Source contains an empty ayah")
            ayah = int(aya.attrib["index"])
            rows.append((f"{number}:{ayah}", len(rows) + 1, number, ayah,
                         sura.attrib["name"], info.transliteration, info.meaning,
                         text, aya.attrib.get("bismillah")))
    if len(rows) != sum(surah.ayah_count for surah in SURAHS):
        raise ValueError("Source corpus coverage failed")

    # Compare actual rows, not only the import receipt, before permitting a no-op.
    with conn:
        conn.execute("SAVEPOINT source_import")
        try:
            existing = [tuple(row) for row in conn.execute(
                "SELECT verse_key, global_ayah_number, surah_number, ayah_number, "
                "surah_name_ar, surah_name_en, surah_meaning_en, arabic_uthmani_min, bismillah "
                "FROM source_ayahs ORDER BY global_ayah_number"
            )]
            if existing != rows:
                if conn.execute("SELECT 1 FROM translation_runs LIMIT 1").fetchone():
                    raise ValueError(
                        "Source differs from a database with translation runs; "
                        "use a new database or an explicitly backed-up source migration"
                    )
                conn.execute("DELETE FROM source_ayahs")
                conn.executemany(
                    "INSERT INTO source_ayahs (verse_key, global_ayah_number, surah_number, "
                    "ayah_number, surah_name_ar, surah_name_en, surah_meaning_en, "
                    "arabic_uthmani_min, bismillah) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
                )
            prior = conn.execute("SELECT sha256 FROM source_files WHERE id = 1").fetchone()
            if not prior or prior[0] != digest:
                conn.execute(
                    "INSERT INTO source_files (id, path, sha256, imported_at) VALUES (1, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET path=excluded.path, sha256=excluded.sha256, "
                    "imported_at=excluded.imported_at", (str(xml_path), digest, utc_now())
                )
            conn.execute("RELEASE source_import")
        except BaseException:
            conn.execute("ROLLBACK TO source_import")
            conn.execute("RELEASE source_import")
            raise
    return {"source_path": str(xml_path), "surahs": len(suras), "ayahs": len(rows)}
