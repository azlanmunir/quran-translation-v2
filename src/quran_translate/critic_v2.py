#!/usr/bin/env python3
"""High-recall fidelity critic contract and validation helpers."""

from __future__ import annotations

from typing import Any


FINDING_TYPES = {
    "addition",
    "omission",
    "altered_agency",
    "altered_negation",
    "altered_scope",
    "collapsed_ambiguity",
    "deadened_metaphor",
    "invented_vividness",
    "sense_error",
    "ledger_error",
    "refrain_error",
    "register_error",
    "invented_evidence_in_note",
}
SEVERITIES = {"blocking", "significant", "minor"}

CRITIC_JSON_SCHEMA = {
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
                        "arabic_ground": {"type": "string"},
                        "explanation": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": [
                        "type",
                        "severity",
                        "where",
                        "arabic_ground",
                        "explanation",
                    ],
                },
            },
            "verdict": {"type": "string", "enum": ["pass", "revise"]},
        },
        "required": ["ayah", "findings", "verdict"],
    },
}


CRITIC_SYSTEM = """You are a forensic fidelity auditor for a Quran translation.
Accuracy matters more than elegance. Audit the supplied English against the supplied
Arabic one ayah at a time. The evidence packet and sense ledger are fallible aids;
they never override the Arabic, syntax, or immediate context.

Work silently through this checklist before returning JSON:

1. CLAUSE COVERAGE: align every Arabic clause, particle with semantic force,
   modifier, restriction, negation, emphasis, comparison, and named participant to
   the English. Flag content omitted or added. A conjunction need not be translated
   as "and" when punctuation or parallel syntax preserves its discourse function;
   do not flag merely formal non-equivalence.
   Do not create a standalone finding merely because emphatic particles such as
   inna, la-, or qad lack a one-word English equivalent. English syntax, focus,
   tense, or punctuation normally carries them. Flag only a material change to the
   clause's certainty or contrast, and explain that change rather than listing the
   untranslated particle.
2. RELATIONS: verify agent, patient, possession, pronoun referent, preposition,
   causality, sequence, number, and scope. In particular, do not flatten distinct
   constructions merely because they share a root.
3. SENSE IN CONTEXT: test the chosen lexical sense against syntax, immediate
   context, concordance, and applicable ledger conditions. Root imagery is not a
   definition. A familiar project rendering can itself be wrong. For a rare word,
   explicitly compare every semantic component in its supplied sense record with
   the English; do not reduce a specific lexical item to a generic category.
4. AMBIGUITY: flag a materially live ambiguity collapsed without contextual support
   or an appropriate note. Do not invent ambiguity merely because a dictionary lists
   several senses.
5. IMAGERY AND FORCE: preserve images the Arabic activates, but flag invented
   imagery, intensity, specificity, emotional color, moral judgment, or universal
   scope. Ordinary English "ever" in a negative clause is an acceptable way to
   express the reach of Arabic negation and is not, by itself, added scope. Likewise,
   "all-knowing" is not an addition where the context itself asserts exhaustive
   knowledge. Context, not morphology alone, decides scope.
6. NOTES: every historical, etymological, cultural, audience-reception, and
   philological claim must be supported by an identified supplied evidence record.
   Morphology or a root label alone is not historical evidence.
7. SECOND PASS: re-read Arabic and English after the first audit. A pass verdict is
   permitted only after all seven checks.

Finding types:
- addition: English asserts semantic content, a modifier, specificity, or judgment
  the Arabic does not assert.
- omission: Arabic semantic content is absent from English; use where "<missing>"
  when no English span can be quoted.
- altered_agency: subject, agent, patient, possessor, or responsibility changes.
- altered_negation: polarity or the scope of negation changes.
- altered_scope: quantifier, intensifier, exception, condition, or logical scope
  changes.
- collapsed_ambiguity: a materially live source ambiguity is silently narrowed.
- deadened_metaphor: an image active in context is replaced by an abstraction.
- invented_vividness: an image or emotional force is introduced without Arabic
  support.
- sense_error: a contextual lexical or constructional sense is wrong.
- ledger_error: the translation mechanically follows or violates a ledger entry in
  a way that conflicts with the Arabic/context.
- refrain_error: an actually invariant formula is inconsistent.
- register_error: a named project register rule is broken; taste is not a finding.
- invented_evidence_in_note: a note states evidence or historical explanation not
  present in an identified supplied evidence record.

Severity:
- blocking: reverses or seriously corrupts the assertion.
- significant: materially changes or loses meaning.
- minor: real but local loss, overstatement, or unsupported narrowing.

A ledger prohibition does not determine severity. Judge the semantic damage. A
plain-English alternative to religious jargon is not a register error merely because
an established religious term also exists.

Return ONLY a JSON array with exactly one object per requested ayah, in order:
[{"ayah": 1, "findings": [{"type": "omission", "severity": "significant",
  "where": "<missing>", "arabic_ground": "exact Arabic span",
  "explanation": "specific, falsifiable explanation",
  "suggestion": "optional minimal repair"}], "verdict": "revise"}]

Every finding requires a non-empty exact Arabic quote and an exact, contiguous
English quote copied verbatim from the supplied translation or note. Never write a
label such as "Note" and never splice a quote with ellipses.
Only an omission with no corresponding English may use "<missing>". Blocking or
significant findings require "revise". Any finding may
justify "revise"; "revise" with no findings is invalid. Do not rewrite for taste."""


def _normalize_quote(text: str) -> str:
    return " ".join(text.casefold().split())


def canonicalize_critic(
    document: Any, notes_by_ayah: dict[int, str]
) -> tuple[Any, list[dict[str, Any]]]:
    """Apply only semantic-preserving, explicitly logged interface normalizations."""
    if not isinstance(document, list):
        return document, []
    normalizations: list[dict[str, Any]] = []
    for entry in document:
        if not isinstance(entry, dict):
            continue
        ayah = entry.get("ayah")
        for index, finding in enumerate(entry.get("findings", [])):
            if not isinstance(finding, dict):
                continue
            if (
                _normalize_quote(str(finding.get("where", ""))) == "note"
                and finding.get("type") == "invented_evidence_in_note"
                and notes_by_ayah.get(ayah)
            ):
                finding["where"] = notes_by_ayah[ayah]
                normalizations.append(
                    {
                        "ayah": ayah,
                        "finding_index": index,
                        "operation": "expand_note_label_to_full_note",
                    }
                )
    return document, normalizations


def validate_critic(
    document: Any,
    expected_ayahs: list[int],
    reader_text_by_ayah: dict[int, str] | None = None,
    arabic_text_by_ayah: dict[int, str] | None = None,
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
        for finding in findings:
            if not isinstance(finding, dict):
                return None
            required = {"type", "severity", "where", "arabic_ground", "explanation"}
            if not required.issubset(finding) or set(finding) - (required | {"suggestion"}):
                return None
            if finding.get("type") not in FINDING_TYPES:
                return None
            if finding.get("severity") not in SEVERITIES:
                return None
            for field in ("where", "arabic_ground", "explanation"):
                if not isinstance(finding.get(field), str) or not finding[field].strip():
                    return None
            if "suggestion" in finding and (
                not isinstance(finding["suggestion"], str)
                or not finding["suggestion"].strip()
            ):
                return None
            if finding["where"] == "<missing>" and finding["type"] != "omission":
                return None
            if reader_text_by_ayah is not None and finding["where"] != "<missing>":
                source = _normalize_quote(reader_text_by_ayah.get(entry["ayah"], ""))
                quote = _normalize_quote(finding["where"])
                if quote in {"note", "translation", "english", "draft"} or quote not in source:
                    return None
            if arabic_text_by_ayah is not None:
                source_arabic = _normalize_quote(
                    arabic_text_by_ayah.get(entry["ayah"], "")
                )
                arabic_quote = _normalize_quote(finding["arabic_ground"])
                if not arabic_quote or arabic_quote not in source_arabic:
                    return None
        major = any(finding["severity"] in {"blocking", "significant"} for finding in findings)
        if major and verdict != "revise":
            return None
        if verdict == "revise" and not findings:
            return None
        if verdict == "pass" and findings:
            return None
    return document
