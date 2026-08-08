"""Deterministic production units and compact evidence packets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONCORDANCE_MAX_GLOBAL = 30
XREF_LIMIT = 3
EXEMPLAR_LIMIT = 1


@dataclass(frozen=True)
class ProductionUnit:
    unit_id: str
    unit_index: int
    surah: int
    first_ayah: int
    last_ayah: int
    context_first: int
    context_last: int

    @property
    def expected_ayahs(self) -> list[int]:
        return list(range(self.first_ayah, self.last_ayah + 1))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def source_verses(conn: sqlite3.Connection) -> dict[tuple[int, int], str]:
    return {
        (int(row["surah_number"]), int(row["ayah_number"])): str(
            row["arabic_uthmani_min"]
        )
        for row in conn.execute(
            """
            SELECT surah_number, ayah_number, arabic_uthmani_min
            FROM source_ayahs
            ORDER BY surah_number, ayah_number
            """
        )
    }


def build_units(
    conn: sqlite3.Connection,
    *,
    max_ayahs: int,
    max_arabic_chars: int,
    context_ayahs: int,
) -> list[ProductionUnit]:
    if max_ayahs <= 0 or max_arabic_chars <= 0 or context_ayahs < 0:
        raise ValueError("Invalid production unit limits")

    rows = list(
        conn.execute(
            """
            SELECT surah_number, ayah_number, arabic_uthmani_min
            FROM source_ayahs
            ORDER BY surah_number, ayah_number
            """
        )
    )
    if len(rows) != 6236:
        raise ValueError(f"Expected 6236 source ayahs, found {len(rows)}")

    by_surah: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        by_surah[int(row["surah_number"])].append(row)

    units: list[ProductionUnit] = []
    unit_index = 0
    for surah in range(1, 115):
        surah_rows = by_surah[surah]
        if not surah_rows:
            raise ValueError(f"Source lacks surah {surah}")
        chunks: list[list[sqlite3.Row]] = []
        current: list[sqlite3.Row] = []
        current_chars = 0
        for row in surah_rows:
            row_chars = len(str(row["arabic_uthmani_min"]))
            would_overflow = current and (
                len(current) >= max_ayahs
                or current_chars + row_chars > max_arabic_chars
            )
            if would_overflow:
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(row)
            current_chars += row_chars
        if current:
            chunks.append(current)

        surah_last = int(surah_rows[-1]["ayah_number"])
        for chunk in chunks:
            unit_index += 1
            first = int(chunk[0]["ayah_number"])
            last = int(chunk[-1]["ayah_number"])
            units.append(
                ProductionUnit(
                    unit_id=f"s{surah:03d}_{first:03d}_{last:03d}",
                    unit_index=unit_index,
                    surah=surah,
                    first_ayah=first,
                    last_ayah=last,
                    context_first=max(1, first - context_ayahs),
                    context_last=min(surah_last, last + context_ayahs),
                )
            )
    return units


def load_morphology(
    path: Path,
) -> tuple[
    dict[tuple[int, int], dict[int, list[tuple[str, str, str]]]],
    dict[str, list[tuple[int, int]]],
]:
    segments: dict[tuple[int, int], dict[int, list[tuple[str, str, str]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        location, form, pos, features = line.split("\t")
        surah, ayah, word, _segment = (int(value) for value in location.split(":"))
        segments[(surah, ayah)][word].append((form, pos, features))

    lemma_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ref, words in segments.items():
        for parts in words.values():
            lemmas = {
                match.group(1)
                for _form, _pos, features in parts
                if (match := re.search(r"LEM:([^|]+)", features))
            }
            for lemma in lemmas:
                if not lemma_index[lemma] or lemma_index[lemma][-1] != ref:
                    lemma_index[lemma].append(ref)
    return segments, lemma_index


def _feature(features: str, key: str) -> str:
    match = re.search(key + r":([^|]+)", features)
    return match.group(1) if match else ""


def build_packet(
    unit: ProductionUnit,
    *,
    segments: dict[tuple[int, int], dict[int, list[tuple[str, str, str]]]],
    lemma_index: dict[str, list[tuple[int, int]]],
    verses: dict[tuple[int, int], str],
    provenance: str,
) -> str:
    lines = [
        f"# Evidence packet - {unit.surah}:{unit.first_ayah}-{unit.last_ayah}",
        "",
        "QAC morphology is scholarly annotation, not raw fact. It constrains form,",
        "not contextual sense. Concordance examples are mechanically selected by",
        "shortest ayah, not interpretive relevance. Counts are distinct ayahs.",
        f"Provenance: {provenance}",
        "",
    ]
    lemmas: list[str] = []
    for ayah in unit.expected_ayahs:
        lines.append(f"## {unit.surah}:{ayah} morphology")
        words = segments.get((unit.surah, ayah), {})
        for word_index in sorted(words):
            parts = words[word_index]
            surface = "".join(part[0] for part in parts)
            descriptors: list[str] = []
            for _form, pos, features in parts:
                lemma = _feature(features, "LEM")
                root = _feature(features, "ROOT")
                descriptor = pos
                if root:
                    descriptor += f" root={root}"
                if lemma:
                    descriptor += f" lemma={lemma}"
                    lemmas.append(lemma)
                descriptors.append(descriptor)
            lines.append(f"- {surface}: " + " + ".join(descriptors))
        lines.append("")

    lines.append(
        f"## Compact concordance (lemmas in fewer than {CONCORDANCE_MAX_GLOBAL} ayahs)"
    )
    seen: set[str] = set()
    for lemma in lemmas:
        if lemma in seen:
            continue
        seen.add(lemma)
        occurrences = lemma_index.get(lemma, [])
        if not 1 <= len(occurrences) < CONCORDANCE_MAX_GLOBAL:
            continue
        outside = [
            ref
            for ref in occurrences
            if not (
                ref[0] == unit.surah
                and unit.first_ayah <= ref[1] <= unit.last_ayah
            )
        ]
        refs = ", ".join(f"{surah}:{ayah}" for surah, ayah in outside[:XREF_LIMIT])
        lines.append(
            f"- **{lemma}**: {len(occurrences)} ayahs"
            + (f"; elsewhere {refs}" if refs else "; only in target")
        )
        for ref in sorted(outside, key=lambda item: len(verses[item]))[:EXEMPLAR_LIMIT]:
            lines.append(f"  - {ref[0]}:{ref[1]} - {verses[ref]}")
    lines.append("")
    return "\n".join(lines)


def write_packets(
    units: list[ProductionUnit],
    *,
    output_dir: Path,
    morphology_path: Path,
    source_path: Path,
    verses: dict[tuple[int, int], str],
) -> None:
    segments, lemma_index = load_morphology(morphology_path)
    source_refs = set(verses)
    morphology_refs = set(segments)
    if morphology_refs != source_refs:
        missing = sorted(source_refs - morphology_refs)[:10]
        unexpected = sorted(morphology_refs - source_refs)[:10]
        raise ValueError(
            "Morphology/source reference mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    provenance = (
        "Quranic Arabic Corpus morphology via github.com/mustafa0x/quran-morphology "
        "(downloaded 2026-08-03, GPL-licensed; "
        f"sha256 {hashlib.sha256(morphology_path.read_bytes()).hexdigest()}); "
        "Tanzil Uthmani Minimal XML "
        f"(sha256 {hashlib.sha256(source_path.read_bytes()).hexdigest()})"
    )
    for unit in units:
        path = output_dir / f"{unit.unit_id}.md"
        packet = build_packet(
            unit,
            segments=segments,
            lemma_index=lemma_index,
            verses=verses,
            provenance=provenance,
        )
        if path.exists() and path.read_text(encoding="utf-8") == packet:
            continue
        atomic_text(path, packet)
