"""Build prompts and parse Gemini JSON responses."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from .config import read_prompt_files


def _ayah_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "ref": row["verse_key"],
        "surah": int(row["surah_number"]),
        "ayah": int(row["ayah_number"]),
        "arabic": row["arabic_uthmani_min"],
    }


def build_batch_payload(conn: sqlite3.Connection, batch: sqlite3.Row) -> dict[str, Any]:
    target_refs = json.loads(batch["target_refs_json"])
    placeholders = ",".join("?" for _ in target_refs)
    targets = list(
        conn.execute(
            f"""
            SELECT *
            FROM source_ayahs
            WHERE verse_key IN ({placeholders})
            ORDER BY surah_number, ayah_number
            """,
            target_refs,
        )
    )
    if len(targets) != len(target_refs):
        raise ValueError(f"Batch {batch['batch_id']} has missing source ayahs")

    run = conn.execute(
        "SELECT * FROM translation_runs WHERE run_id = ?",
        (batch["run_id"],),
    ).fetchone()
    context_before = int(run["context_before"])
    context_after = int(run["context_after"])
    first = targets[0]
    last = targets[-1]

    before_rows = []
    if context_before:
        before_rows = list(
            conn.execute(
                """
                SELECT *
                FROM source_ayahs
                WHERE surah_number = ?
                  AND ayah_number < ?
                ORDER BY ayah_number DESC
                LIMIT ?
                """,
                (first["surah_number"], first["ayah_number"], context_before),
            )
        )
        before_rows.reverse()

    after_rows = []
    if context_after:
        after_rows = list(
            conn.execute(
                """
                SELECT *
                FROM source_ayahs
                WHERE surah_number = ?
                  AND ayah_number > ?
                ORDER BY ayah_number ASC
                LIMIT ?
                """,
                (last["surah_number"], last["ayah_number"], context_after),
            )
        )

    opening_bismillah = None
    first_ayah = conn.execute(
        """
        SELECT bismillah
        FROM source_ayahs
        WHERE surah_number = ? AND ayah_number = 1
        """,
        (first["surah_number"],),
    ).fetchone()
    if first_ayah:
        opening_bismillah = first_ayah["bismillah"]

    return {
        "surah": {
            "number": int(first["surah_number"]),
            "name_ar": first["surah_name_ar"],
            "name_en": first["surah_name_en"],
            "meaning_en": first["surah_meaning_en"],
            "opening_bismillah": opening_bismillah,
        },
        "context_before": [_ayah_payload(row) for row in before_rows],
        "targets": [_ayah_payload(row) for row in targets],
        "context_after": [_ayah_payload(row) for row in after_rows],
    }


def build_prompt(payload: dict[str, Any]) -> str:
    philological, contract, _ = read_prompt_files()
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        f"{philological}\n\n"
        f"{contract}\n\n"
        "Input JSON:\n"
        f"{payload_text}\n"
    )


def strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_translation_response(raw_response: str, expected_refs: list[str]) -> list[dict[str, Any]]:
    data = json.loads(strip_json_fences(raw_response))
    if isinstance(data, list):
        translations = data
    elif isinstance(data, dict) and isinstance(data.get("translations"), list):
        translations = data["translations"]
    else:
        raise ValueError("Response must be a list or an object with a translations list")

    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    expected = set(expected_refs)

    for item in translations:
        if not isinstance(item, dict):
            raise ValueError("Each translation item must be an object")
        ref = item.get("ref")
        translation = item.get("translation")
        if ref not in expected:
            raise ValueError(f"Unexpected ref in response: {ref!r}")
        if ref in seen:
            raise ValueError(f"Duplicate ref in response: {ref}")
        if not isinstance(translation, str) or not translation.strip():
            raise ValueError(f"Missing translation text for {ref}")
        word_bank = item.get("word_bank", [])
        if word_bank is None:
            word_bank = []
        if not isinstance(word_bank, list):
            raise ValueError(f"word_bank must be a list for {ref}")
        for entry in word_bank:
            if not isinstance(entry, dict):
                raise ValueError(f"Each word_bank entry must be an object for {ref}")
            term = str(entry.get("term") or "").strip()
            root = str(entry.get("root") or "").strip()
            rendering = str(entry.get("rendering") or "").strip()
            physical_reality = str(
                entry.get("physical_reality")
                or entry.get("630_ce_physical_reality")
                or entry.get("definition")
                or ""
            ).strip()
            if not term or not root or not rendering or not physical_reality:
                raise ValueError(
                    f"word_bank entries must include term, root, rendering, and physical_reality for {ref}"
                )
        item["translation"] = translation.strip()
        item["word_bank"] = word_bank
        parsed.append(item)
        seen.add(ref)

    missing = expected - seen
    if missing:
        raise ValueError(f"Missing refs in response: {', '.join(sorted(missing))}")

    return sorted(parsed, key=lambda item: expected_refs.index(item["ref"]))
