"""Strict Urdu production contracts and deterministic release gates."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from .refrains import repeated_ayah_groups
from .urdu_translation_bakeoff import ALLOWED_FLAGS, validate_translation


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
    "hindi_drift",
    "archaic_obscurity",
    "untranslated_arabic",
    "register_error",
    "spoken_runon",
    "terminology_drift",
}
SEVERITIES = {"blocking", "significant", "minor"}
SEVERITY_RANK = {"minor": 1, "significant": 2, "blocking": 3}


FINDING_SCHEMA: dict[str, Any] = {
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
        "suggestion",
    ],
    "additionalProperties": False,
}

CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ayahs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ayah": {"type": "integer"},
                    "findings": {"type": "array", "items": FINDING_SCHEMA},
                    "verdict": {"type": "string", "enum": ["pass", "revise"]},
                },
                "required": ["ayah", "findings", "verdict"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ayahs"],
    "additionalProperties": False,
}

REVISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ayahs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ayah": {"type": "integer"},
                    "urdu": {"type": "string"},
                    "review_flags": {
                        "type": "array",
                        "items": {"type": "string", "enum": sorted(ALLOWED_FLAGS)},
                    },
                },
                "required": ["ayah", "urdu", "review_flags"],
                "additionalProperties": False,
            },
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding_id": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": ["applied", "rejected", "escalated"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["finding_id", "decision", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["ayahs", "decisions"],
    "additionalProperties": False,
}

VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ayahs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ayah": {"type": "integer"},
                    "accept": {"type": "boolean"},
                    "findings": {"type": "array", "items": FINDING_SCHEMA},
                    "reason": {"type": "string"},
                },
                "required": ["ayah", "accept", "findings", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ayahs"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    code: str
    ref: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.translate(
        str.maketrans({"ی": "ي", "ے": "ي", "ک": "ك", "ـ": ""})
    )
    return " ".join(normalized.split())


def _strip_wrapping_quotes(value: str) -> str:
    text = value.strip()
    pairs = {('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("«", "»")}
    if len(text) >= 2 and (text[0], text[-1]) in pairs:
        return text[1:-1].strip()
    return text


def _valid_ground(ground: str, source: str) -> bool:
    return bool(ground.strip()) and _normalize(_strip_wrapping_quotes(ground)) in _normalize(
        source
    )


def validate_finding(
    finding: Any,
    *,
    arabic: str,
    urdu: str,
) -> dict[str, str] | None:
    if not isinstance(finding, dict) or set(finding) != {
        "type",
        "severity",
        "where",
        "arabic_ground",
        "explanation",
        "suggestion",
    }:
        return None
    finding_type = finding.get("type")
    severity = finding.get("severity")
    where = finding.get("where")
    arabic_ground = finding.get("arabic_ground")
    explanation = finding.get("explanation")
    suggestion = finding.get("suggestion")
    if finding_type not in FINDING_TYPES or severity not in SEVERITIES:
        return None
    if not all(isinstance(value, str) for value in (where, arabic_ground, explanation, suggestion)):
        return None
    if not explanation.strip() or not _valid_ground(arabic_ground, arabic):
        return None
    if where != "<missing>" and not _valid_ground(where, urdu):
        return None
    if where == "<missing>" and finding_type != "omission":
        return None
    return {
        "type": str(finding_type),
        "severity": str(severity),
        "where": where.strip(),
        "arabic_ground": arabic_ground.strip(),
        "explanation": explanation.strip(),
        "suggestion": suggestion.strip(),
    }


def validate_critic(
    document: Any,
    *,
    expected: list[int],
    arabic_by_ayah: dict[int, str],
    urdu_by_ayah: dict[int, str],
) -> dict[str, Any] | None:
    if not isinstance(document, dict) or set(document) != {"ayahs"}:
        return None
    rows = document.get("ayahs")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None
    if [row.get("ayah") for row in rows if isinstance(row, dict)] != expected:
        return None
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"ayah", "findings", "verdict"}:
            return None
        ayah = int(row["ayah"])
        findings = row.get("findings")
        verdict = row.get("verdict")
        if not isinstance(findings, list) or verdict not in {"pass", "revise"}:
            return None
        clean_findings: list[dict[str, str]] = []
        for finding in findings:
            clean = validate_finding(
                finding,
                arabic=arabic_by_ayah[ayah],
                urdu=urdu_by_ayah[ayah],
            )
            if clean is None:
                return None
            clean_findings.append(clean)
        requires_revision = any(
            item["severity"] in {"blocking", "significant"}
            for item in clean_findings
        )
        if verdict == "revise" and not requires_revision:
            return None
        if verdict == "pass" and requires_revision:
            return None
        clean_rows.append(
            {"ayah": ayah, "findings": clean_findings, "verdict": verdict}
        )
    return {"ayahs": clean_rows}


def finding_records(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "finding_id": f"f-{row['ayah']}-{index}",
            "ayah": row["ayah"],
            **finding,
        }
        for row in document["ayahs"]
        for index, finding in enumerate(row["findings"])
    ]


def validate_revision(
    document: Any,
    *,
    expected: list[int],
    finding_ids: list[str],
) -> dict[str, Any] | None:
    if not isinstance(document, dict) or set(document) != {"ayahs", "decisions"}:
        return None
    translated = validate_translation({"ayahs": document.get("ayahs")}, expected)
    decisions = document.get("decisions")
    if translated is None or not isinstance(decisions, list):
        return None
    if len(decisions) != len(finding_ids):
        return None
    clean_decisions: list[dict[str, str]] = []
    seen: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != {
            "finding_id",
            "decision",
            "reason",
        }:
            return None
        finding_id = decision.get("finding_id")
        action = decision.get("decision")
        reason = decision.get("reason")
        if (
            finding_id not in finding_ids
            or finding_id in seen
            or action not in {"applied", "rejected", "escalated"}
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            return None
        seen.add(str(finding_id))
        clean_decisions.append(
            {
                "finding_id": str(finding_id),
                "decision": str(action),
                "reason": reason.strip(),
            }
        )
    if seen != set(finding_ids):
        return None
    return {"ayahs": translated["ayahs"], "decisions": clean_decisions}


def validate_verification(
    document: Any,
    *,
    expected: list[int],
    arabic_by_ayah: dict[int, str],
    proposed_by_ayah: dict[int, str],
) -> dict[str, Any] | None:
    if not isinstance(document, dict) or set(document) != {"ayahs"}:
        return None
    rows = document.get("ayahs")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None
    if [row.get("ayah") for row in rows if isinstance(row, dict)] != expected:
        return None
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "ayah",
            "accept",
            "findings",
            "reason",
        }:
            return None
        ayah = int(row["ayah"])
        accept = row.get("accept")
        findings = row.get("findings")
        reason = row.get("reason")
        if not isinstance(accept, bool) or not isinstance(findings, list):
            return None
        if not isinstance(reason, str) or not reason.strip():
            return None
        clean_findings: list[dict[str, str]] = []
        for finding in findings:
            clean = validate_finding(
                finding,
                arabic=arabic_by_ayah[ayah],
                urdu=proposed_by_ayah[ayah],
            )
            if clean is None:
                return None
            clean_findings.append(clean)
        severe = any(
            item["severity"] in {"blocking", "significant"}
            for item in clean_findings
        )
        if accept and severe:
            return None
        if not accept and not severe:
            return None
        clean_rows.append(
            {
                "ayah": ayah,
                "accept": accept,
                "findings": clean_findings,
                "reason": reason.strip(),
            }
        )
    return {"ayahs": clean_rows}


def _issue(code: str, ref: str, message: str, severity: str = "error") -> QualityIssue:
    return QualityIssue(severity=severity, code=code, ref=ref, message=message)


def deterministic_quality_gate(
    translations: dict[tuple[int, int], str],
    verses: dict[tuple[int, int], str],
) -> dict[str, Any]:
    issues: list[QualityIssue] = []
    source_refs = set(verses)
    translation_refs = set(translations)
    for ref in sorted(source_refs - translation_refs):
        issues.append(_issue("missing_ayah", f"{ref[0]}:{ref[1]}", "Missing Urdu ayah"))
    for ref in sorted(translation_refs - source_refs):
        issues.append(_issue("extra_ayah", f"{ref[0]}:{ref[1]}", "Unexpected Urdu ayah"))

    for (surah, ayah), text in sorted(translations.items()):
        ref = f"{surah}:{ayah}"
        if not text.strip():
            issues.append(_issue("blank_urdu", ref, "Urdu text is blank"))
        if re.search(r"[\u0900-\u097f]", text):
            issues.append(_issue("devanagari", ref, "Devanagari characters are forbidden"))
        if re.search(r"[\[\](){}（）]", text):
            issues.append(_issue("parenthetical_tafsir", ref, "Body text contains brackets or parentheses"))
        if "خدا" in text:
            issues.append(_issue("divine_name_drift", ref, "Use اللہ consistently instead of خدا"))
        if "تقویٰ اختیار" in text or "تقوی اختیار" in text:
            issues.append(
                _issue(
                    "untranslated_taqwa_verb",
                    ref,
                    "Render the active verbal force instead of 'adopt taqwa'",
                )
            )

    exact_rules = {
        (2, 177): {
            "required": ["مشرق", "مغرب", "اور"],
            "forbidden": ["مشرق یا مغرب"],
        },
        (2, 187): {
            "required": ["فجر", "مباشرت نہ کرو"],
            "forbidden": ["میل جول", "ان سے نہ ملو", "صبح کا سفید"],
        },
        (4, 12): {"required": ["کلالہ"], "forbidden": []},
        (4, 176): {"required": ["کلالہ"], "forbidden": []},
        (17, 32): {"required": ["زنا"], "forbidden": ["ناجائز جنسی تعلق"]},
        (24, 2): {"required": ["زنا"], "forbidden": ["ناجائز جنسی تعلق"]},
        (24, 3): {"required": ["زنا"], "forbidden": ["ناجائز جنسی تعلق"]},
        (112, 2): {"required": ["محتاج"], "forbidden": []},
    }
    for ref_tuple, rule in exact_rules.items():
        text = translations.get(ref_tuple)
        if text is None:
            continue
        ref = f"{ref_tuple[0]}:{ref_tuple[1]}"
        for term in rule["required"]:
            if term not in text:
                issues.append(_issue("ledger_required", ref, f"Required Urdu term is absent: {term}"))
        for term in rule["forbidden"]:
            if term in text:
                issues.append(_issue("ledger_forbidden", ref, f"Forbidden Urdu wording is present: {term}"))

    checked_groups = 0
    for group_id, group in repeated_ayah_groups(verses).items():
        refs = [tuple(ref) for ref in group["refs"]]
        renderings = {translations[ref] for ref in refs if ref in translations}
        if len(renderings) < 1:
            continue
        checked_groups += 1
        if len(renderings) > 1:
            joined = ", ".join(f"{surah}:{ayah}" for surah, ayah in refs)
            issues.append(
                _issue(
                    "refrain_divergence",
                    joined,
                    f"Identical Arabic has {len(renderings)} Urdu renderings ({group_id[:12]})",
                )
            )

    errors = [issue for issue in issues if issue.severity == "error"]
    return {
        "version": "urdu-deterministic-quality-v1",
        "source_ayahs": len(source_refs),
        "translation_ayahs": len(translation_refs),
        "repeated_groups_checked": checked_groups,
        "errors": len(errors),
        "warnings": len(issues) - len(errors),
        "issues": [issue.to_dict() for issue in issues],
        "passed": not errors,
    }
