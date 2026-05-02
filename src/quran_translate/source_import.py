"""Import Tanzil XML into the local database."""

from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import DEFAULT_SOURCE_XML, file_sha256
from .db import utc_now
from .metadata import surah_info


def import_tanzil_xml(conn: sqlite3.Connection, xml_path: Path = DEFAULT_SOURCE_XML) -> dict[str, int | str]:
    if not xml_path.exists():
        raise FileNotFoundError(f"Source XML not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    if root.tag != "quran":
        raise ValueError(f"Expected <quran> root, got <{root.tag}>")

    now = utc_now()
    global_index = 0
    surah_count = 0
    ayah_count = 0

    with conn:
        conn.execute("DELETE FROM source_ayahs")
        for sura in root.findall("sura"):
            surah_count += 1
            surah_number = int(sura.attrib["index"])
            surah_name_ar = sura.attrib["name"]
            info = surah_info(surah_number)

            for aya in sura.findall("aya"):
                global_index += 1
                ayah_count += 1
                ayah_number = int(aya.attrib["index"])
                verse_key = f"{surah_number}:{ayah_number}"
                conn.execute(
                    """
                    INSERT INTO source_ayahs (
                        verse_key,
                        global_ayah_number,
                        surah_number,
                        ayah_number,
                        surah_name_ar,
                        surah_name_en,
                        surah_meaning_en,
                        arabic_uthmani_min,
                        bismillah
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        verse_key,
                        global_index,
                        surah_number,
                        ayah_number,
                        surah_name_ar,
                        info.transliteration,
                        info.meaning,
                        aya.attrib["text"],
                        aya.attrib.get("bismillah"),
                    ),
                )

        conn.execute(
            """
            INSERT INTO source_files (id, path, sha256, imported_at)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                path = excluded.path,
                sha256 = excluded.sha256,
                imported_at = excluded.imported_at
            """,
            (str(xml_path), file_sha256(xml_path), now),
        )

    return {
        "source_path": str(xml_path),
        "surahs": surah_count,
        "ayahs": ayah_count,
    }

