#!/usr/bin/env python3
"""Contract for a narrow, source-aware spoken-English quality checker."""

from __future__ import annotations

from typing import Any


FINDING_TYPES = {
    "ungrammatical",
    "source_language_calque",
    "archaic_or_obscure",
    "unclear_referent",
    "broken_parallelism",
    "read_aloud_failure",
    "register_violation",
    "signature_translation_phrase",
    "softened_force",
    "flattened_oath",
    "dropped_emphasis",
}
SEVERITIES = {"material", "minor"}

SPOKEN_ENGLISH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "ayah": {"type": "integer"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": sorted(FINDING_TYPES)},
                        "severity": {"type": "string", "enum": sorted(SEVERITIES)},
                        "where": {"type": "string"},
                        "explanation": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": [
                        "type",
                        "severity",
                        "where",
                        "explanation",
                        "suggestion",
                    ],
                },
            },
            "verdict": {"type": "string", "enum": ["pass", "revise"]},
        },
        "required": ["ayah", "findings", "verdict"],
    },
}


SPOKEN_ENGLISH_SYSTEM = """You are a narrow spoken-English quality checker for a
Quran translation. The translation has already passed a separate fidelity critic.
Your job is not to reinterpret, beautify, modernize ideas, or change theology. Find
only English that a modern listener cannot parse cleanly in real time.

Read each line aloud silently and check:

1. Is it grammatical English?
2. Does Arabic-shaped word order or a literal preposition make it opaque?
3. Does an archaic or specialist English word violate the supplied register policy?
4. Is a pronoun or referent accidentally unclear in English?
5. Is a list or purpose clause grammatically broken?
6. Does the line fail aloud because its syntax cannot be followed at normal speed?
7. Does it fall into a curated signature phrase or cadence from a major published
   translation where the Arabic permits a fresher, equally faithful rhythm?
8. Has warning, threat, punishment, woe, law, or judgment been softened into calm,
   therapeutic, or merely reflective English?
9. Has an oath opening been flattened, or has material emphasis disappeared from the
   English clause rather than being carried by syntax, focus, or punctuation?

Do not flag:

- content merely because it is unfamiliar, severe, repetitive, or uncomfortable;
- a source-mandated image, abrupt turn, repetition, proper name, or ambiguity;
- a valid sentence merely because you prefer another style;
- semantic or philological disputes assigned to the fidelity critic.

Positive targets for evidence-neutral register choices include "cool your eye,"
"my head is ablaze with white hair," and "We can restore even his fingertips."
Their value is source-grounded physical force, not ornament. Conventional words are
fine; conventional rhythm is what this check should notice.

Every suggestion must be the smallest plain-English repair and must preserve the
same claim, agents, negation, scope, imagery, and uncertainty. If no meaning-neutral
repair is possible, say so in the explanation and suggest routing the line back to
the source-aware reviser rather than guessing.

Return only a JSON array with exactly one object per requested ayah, in order:
[{"ayah": 1, "findings": [{"type": "source_language_calque",
  "severity": "material", "where": "exact contiguous English quote",
  "explanation": "specific and falsifiable explanation",
  "suggestion": "minimal replacement"}], "verdict": "revise"}]

`where` must be a non-empty exact contiguous quote from the supplied English.
Every finding requires a non-empty suggestion. Any finding requires `revise`; a pass
must have no findings. Do not return a rewritten ayah or any text outside the JSON.
"""


SIGNATURE_PHRASES = (
    {
        "phrase": "be mindful of God",
        "source": "M. A. S. Abdel Haleem house rendering",
        "guidance": "Adjudicate by context; warning, law, threat, and judgment normally require fear or guard language.",
    },
    {
        "phrase": "those mindful of God",
        "source": "M. A. S. Abdel Haleem house rendering",
        "guidance": "Retain only for descriptive piety; route active warning contexts for force review.",
    },
)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def signature_phrase_findings(ayah: int, english: str) -> list[dict[str, str | int]]:
    lowered = english.casefold()
    findings: list[dict[str, str | int]] = []
    for record in SIGNATURE_PHRASES:
        phrase = record["phrase"]
        start = lowered.find(phrase.casefold())
        if start < 0:
            continue
        exact = english[start : start + len(phrase)]
        findings.append(
            {
                "ayah": ayah,
                "type": "signature_translation_phrase",
                "severity": "minor",
                "where": exact,
                "explanation": (
                    f"Matches the curated signature phrase from {record['source']}. "
                    "This is a review lead, not an automatic defect."
                ),
                "suggestion": record["guidance"],
            }
        )
    return findings


def validate_spoken_english(
    document: Any,
    expected_ayahs: list[int],
    english_by_ayah: dict[int, str],
) -> list[dict[str, Any]] | None:
    if not isinstance(document, list) or len(document) != len(expected_ayahs):
        return None
    if [entry.get("ayah") for entry in document if isinstance(entry, dict)] != expected_ayahs:
        return None

    for entry in document:
        if not isinstance(entry, dict) or set(entry) != {"ayah", "findings", "verdict"}:
            return None
        findings = entry.get("findings")
        verdict = entry.get("verdict")
        if not isinstance(findings, list) or verdict not in {"pass", "revise"}:
            return None
        if (verdict == "pass") != (len(findings) == 0):
            return None
        source = _normalize(english_by_ayah.get(entry["ayah"], ""))
        for finding in findings:
            if not isinstance(finding, dict) or set(finding) != {
                "type",
                "severity",
                "where",
                "explanation",
                "suggestion",
            }:
                return None
            if finding.get("type") not in FINDING_TYPES:
                return None
            if finding.get("severity") not in SEVERITIES:
                return None
            for field in ("where", "explanation", "suggestion"):
                if not isinstance(finding.get(field), str) or not finding[field].strip():
                    return None
            quote = _normalize(finding["where"])
            if quote not in source or quote in {"translation", "english", "line"}:
                return None
    return document
