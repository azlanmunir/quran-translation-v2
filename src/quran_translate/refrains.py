#!/usr/bin/env python3
"""Mechanical repeated-ayah enumeration and translation-invariance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


def normalize_arabic(text: str) -> str:
    """Normalize encoding and spacing without erasing phonemic distinctions."""
    normalized = unicodedata.normalize("NFC", text).replace("ـ", "")
    return " ".join(normalized.split())


def refrain_id(arabic: str) -> str:
    return hashlib.sha256(normalize_arabic(arabic).encode("utf-8")).hexdigest()


def load_quran_xml(path: Path) -> dict[tuple[int, int], str]:
    root = ET.parse(path).getroot()
    verses: dict[tuple[int, int], str] = {}
    for surah in root.iter("sura"):
        surah_number = int(surah.attrib["index"])
        for ayah in surah.findall("aya"):
            verses[(surah_number, int(ayah.attrib["index"]))] = ayah.attrib["text"]
    return verses


def repeated_ayah_groups(
    verses: dict[tuple[int, int], str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    arabic_by_id: dict[str, str] = {}
    for ref, arabic in verses.items():
        group_id = refrain_id(arabic)
        grouped[group_id].append(ref)
        arabic_by_id[group_id] = normalize_arabic(arabic)
    return {
        group_id: {"arabic": arabic_by_id[group_id], "refs": sorted(refs)}
        for group_id, refs in grouped.items()
        if len(refs) > 1
    }


def validate_refrain_invariance(
    translations: list[dict[str, Any]],
    verses: dict[tuple[int, int], str],
    canonical: dict[str, str] | None = None,
) -> dict[str, Any]:
    english = {(row["surah"], row["ayah"]): row["english"] for row in translations}
    checked: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    for group_id, group in repeated_ayah_groups(verses).items():
        present = [tuple(ref) for ref in group["refs"] if tuple(ref) in english]
        if len(present) < 2:
            continue
        renderings = sorted({english[ref] for ref in present})
        record = {
            "group_id": group_id,
            "refs": [list(ref) for ref in present],
            "renderings": renderings,
        }
        checked.append(record)
        expected = canonical.get(group_id) if canonical else None
        if len(renderings) != 1 or (expected is not None and renderings[0] != expected):
            if expected is not None:
                record["expected"] = expected
            divergences.append(record)
    if divergences:
        raise ValueError(f"Repeated Arabic has divergent English: {divergences}")
    return {"groups_checked": len(checked), "groups": checked}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if identical Quran Arabic has divergent English."
    )
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--quran", type=Path, required=True)
    parser.add_argument("--canonical", type=Path)
    args = parser.parse_args()

    translations = json.loads(args.translations.read_text(encoding="utf-8"))
    canonical: dict[str, str] | None = None
    if args.canonical is not None:
        policy = json.loads(args.canonical.read_text(encoding="utf-8"))
        canonical = policy.get("canonical", policy)
    result = validate_refrain_invariance(
        translations, load_quran_xml(args.quran), canonical
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
