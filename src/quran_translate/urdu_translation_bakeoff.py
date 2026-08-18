"""Resumable, blinded Arabic-to-Urdu translation model bakeoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import ssl
import time
import unicodedata
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import (
    DEFAULT_SOURCE_XML,
    OUTPUT_DIR,
    PROJECT_ROOT,
    load_dotenv,
)
from .metadata import surah_info
from .production_packets import (
    ProductionUnit,
    atomic_json,
    atomic_text,
    build_packet,
    load_morphology,
)


BAKEOFF_ID = "quran-urdu-translation-models-20260818-v2"
DEFAULT_ROOT = OUTPUT_DIR / "urdu" / "bakeoffs" / BAKEOFF_ID
POLICY_PATH = PROJECT_ROOT / "prompts" / "urdu-translation-v1.md"
AUDIT_PATH = PROJECT_ROOT / "prompts" / "urdu-audit-v1.md"
LEDGER_MD_PATH = PROJECT_ROOT / "prompts" / "sense-ledger-v2.4.md"
LEDGER_JSON_PATH = PROJECT_ROOT / "data" / "evidence" / "sense-ledger-v2.4.json"
MORPHOLOGY_PATH = PROJECT_ROOT / "data" / "evidence" / "qac-morphology.txt"
NOTES_PATH = PROJECT_ROOT / "data" / "evidence" / "reading-notes-v2.4.1.json"
ADJUDICATIONS_PATH = (
    PROJECT_ROOT / "data" / "evidence" / "release-adjudications-v2.4.1.json"
)
PREREGISTRATION_PATH = PROJECT_ROOT / "URDU_TRANSLATION_BAKEOFF_v1.md"

CANONICAL_SOURCE_SHA256 = (
    "f78067cd98c51c03e450581e1e8713f4e7c352e0b62a4fe5c35811da28dd23bf"
)
CANONICAL_MORPHOLOGY_SHA256 = (
    "742bfac59941b2cb09736d5b7aae694af50792261fb8450cbf6afafcc340645f"
)
MAX_OUTPUT_TOKENS = 16_000
CONTRACT_ATTEMPTS = 2
DEFAULT_WORKERS = 4
RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504, 529}

ALLOWED_FLAGS = {
    "rare_word",
    "disputed_grammar",
    "disputed_sense",
    "legal",
    "loanword",
    "ledger_gap",
    "low_confidence",
}
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
}
SEVERITIES = {"blocking", "significant", "minor"}
SCORE_FIELDS = {
    "naturalness",
    "clarity",
    "pakistani_urdu",
    "spoken_cadence",
    "source_force",
}


class UrduBakeoffError(RuntimeError):
    """The bakeoff cannot continue without violating a frozen invariant."""


@dataclass(frozen=True)
class PassageSpec:
    passage_id: str
    surah: int
    first_ayah: int
    last_ayah: int
    focus: str

    @property
    def expected_ayahs(self) -> list[int]:
        return list(range(self.first_ayah, self.last_ayah + 1))


@dataclass(frozen=True)
class ModelSpec:
    candidate_id: str
    provider: str
    model_id: str
    reasoning: str
    private_label: str


PASSAGES = (
    PassageSpec("fatihah", 1, 1, 7, "opening prayer, titles, and agency"),
    PassageSpec("fasting", 2, 177, 187, "piety, law, fasting, and taqwa"),
    PassageSpec("inheritance", 4, 11, 12, "dense inheritance law"),
    PassageSpec("nisa-34", 4, 34, 35, "contested relational and legal vocabulary"),
    PassageSpec("yusuf-dialogue", 12, 23, 30, "narrative dialogue and agency"),
    PassageSpec("zakariya", 19, 1, 11, "prayer, bodily imagery, and signaling"),
    PassageSpec("zina-law", 24, 2, 5, "legal scope and testimony"),
    PassageSpec("rahman-opening", 55, 1, 16, "dual address and repeated refrain"),
    PassageSpec("takwir", 81, 1, 14, "eschatological imagery and rare nouns"),
    PassageSpec("duha", 93, 1, 11, "oaths and restrained consolation"),
    PassageSpec("ikhlas", 112, 1, 4, "theological compression and al-samad"),
)

CANDIDATES = (
    ModelSpec("anthropic-fable-5", "anthropic", "claude-fable-5", "high", "Claude Fable 5"),
    ModelSpec("anthropic-opus-4-8", "anthropic", "claude-opus-4-8", "high", "Claude Opus 4.8"),
    ModelSpec("google-gemini-31-pro", "google", "gemini-3.1-pro-preview", "high", "Gemini 3.1 Pro"),
    ModelSpec("google-gemini-37-flash", "google", "gemini-3.7-flash", "high", "Gemini 3.7 Flash"),
    ModelSpec("openai-gpt-56-sol", "openai", "gpt-5.6-sol", "high", "GPT-5.6 Sol"),
    ModelSpec("openai-gpt-56-terra", "openai", "gpt-5.6-terra", "high", "GPT-5.6 Terra"),
    ModelSpec("openrouter-muse-spark-12", "openrouter", "meta/muse-spark-1.2", "high", "Muse Spark 1.2"),
)

AUDITORS = (
    ModelSpec("auditor-gemini-37", "google", "gemini-3.7-flash", "high", "Gemini 3.7 Flash auditor"),
    ModelSpec("auditor-gpt-56-terra", "openai", "gpt-5.6-terra", "high", "GPT-5.6 Terra auditor"),
)


TRANSLATION_SCHEMA: dict[str, Any] = {
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
        }
    },
    "required": ["ayahs"],
    "additionalProperties": False,
}


def audit_schema(codes: list[str]) -> dict[str, Any]:
    finding = {
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
    ayah = {
        "type": "object",
        "properties": {
            "ayah": {"type": "integer"},
            "findings": {"type": "array", "items": finding},
            "verdict": {"type": "string", "enum": ["pass", "revise"]},
        },
        "required": ["ayah", "findings", "verdict"],
        "additionalProperties": False,
    }
    scores = {
        "type": "object",
        "properties": {
            field: {"type": "integer", "minimum": 1, "maximum": 5}
            for field in sorted(SCORE_FIELDS)
        },
        "required": sorted(SCORE_FIELDS),
        "additionalProperties": False,
    }
    candidate = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "enum": codes},
            "ayahs": {"type": "array", "items": ayah},
            "scores": scores,
            "summary": {"type": "string"},
        },
        "required": ["code", "ayahs", "scores", "summary"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"candidates": {"type": "array", "items": candidate}},
        "required": ["candidates"],
        "additionalProperties": False,
    }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path.home() / "Downloads" / "quran-translation" / ".env")


def _source_verses() -> tuple[dict[tuple[int, int], str], dict[int, int]]:
    if file_hash(DEFAULT_SOURCE_XML) != CANONICAL_SOURCE_SHA256:
        raise UrduBakeoffError("Pinned Tanzil source hash changed")
    root = ET.parse(DEFAULT_SOURCE_XML).getroot()
    verses: dict[tuple[int, int], str] = {}
    last_by_surah: dict[int, int] = {}
    for sura in root.findall("sura"):
        surah = int(sura.attrib["index"])
        for aya in sura.findall("aya"):
            ayah = int(aya.attrib["index"])
            verses[(surah, ayah)] = aya.attrib["text"]
            last_by_surah[surah] = ayah
    if len(verses) != 6236 or set(last_by_surah) != set(range(1, 115)):
        raise UrduBakeoffError("Pinned source does not contain 114 surahs and 6236 ayahs")
    return verses, last_by_surah


def _ref_in_passage(ref: str, passage: PassageSpec) -> bool:
    match = re.fullmatch(r"(\d+):(\d+)", ref)
    if not match:
        return False
    return (
        int(match.group(1)) == passage.surah
        and passage.first_ayah <= int(match.group(2)) <= passage.last_ayah
    )


def _applicable_records(passage: PassageSpec) -> dict[str, list[dict[str, Any]]]:
    notes_doc = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    adjudications_doc = json.loads(ADJUDICATIONS_PATH.read_text(encoding="utf-8"))
    return {
        "reading_notes": [
            note
            for note in notes_doc.get("notes", [])
            if isinstance(note, dict)
            and _ref_in_passage(str(note.get("ref", "")), passage)
        ],
        "release_adjudications": [
            decision
            for decision in adjudications_doc.get("decisions", [])
            if isinstance(decision, dict)
            and _ref_in_passage(str(decision.get("ref", "")), passage)
        ],
    }


def build_passage_payloads() -> list[dict[str, Any]]:
    if file_hash(MORPHOLOGY_PATH) != CANONICAL_MORPHOLOGY_SHA256:
        raise UrduBakeoffError("Pinned QAC morphology hash changed")
    verses, last_by_surah = _source_verses()
    segments, lemma_index = load_morphology(MORPHOLOGY_PATH)
    provenance = (
        "Quranic Arabic Corpus morphology via github.com/mustafa0x/quran-morphology "
        f"(sha256 {file_hash(MORPHOLOGY_PATH)}); Tanzil Uthmani Minimal XML "
        f"(sha256 {file_hash(DEFAULT_SOURCE_XML)})"
    )
    payloads: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, passage in enumerate(PASSAGES, start=1):
        refs = {(passage.surah, ayah) for ayah in passage.expected_ayahs}
        if not refs.issubset(verses) or seen & refs:
            raise UrduBakeoffError(f"Invalid or overlapping passage: {passage.passage_id}")
        seen.update(refs)
        unit = ProductionUnit(
            unit_id=passage.passage_id,
            unit_index=index,
            surah=passage.surah,
            first_ayah=passage.first_ayah,
            last_ayah=passage.last_ayah,
            context_first=max(1, passage.first_ayah - 2),
            context_last=min(last_by_surah[passage.surah], passage.last_ayah + 2),
        )
        payload = {
            "passage": asdict(passage),
            "surah_name": asdict(surah_info(passage.surah)),
            "local_context": [
                {"ayah": ayah, "arabic": verses[(passage.surah, ayah)]}
                for ayah in range(unit.context_first, unit.context_last + 1)
            ],
            "target": [
                {"ayah": ayah, "arabic": verses[(passage.surah, ayah)]}
                for ayah in passage.expected_ayahs
            ],
            "evidence_packet": build_packet(
                unit,
                segments=segments,
                lemma_index=lemma_index,
                verses=verses,
                provenance=provenance,
            ),
            "adjudicated_semantic_records": _applicable_records(passage),
        }
        payload["input_sha256"] = stable_hash(payload)
        payloads.append(payload)
    if len(seen) != 90:
        raise UrduBakeoffError(f"Bakeoff corpus must contain exactly 90 ayahs, found {len(seen)}")
    return payloads


def _private_blind_key(root: Path) -> dict[str, str]:
    path = root / "PRIVATE_BLIND_KEY.json"
    candidate_ids = [candidate.candidate_id for candidate in CANDIDATES]
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        mapping = document.get("mapping")
        if not isinstance(mapping, dict) or set(mapping) != set(candidate_ids):
            raise UrduBakeoffError("Existing blind key does not match the frozen roster")
        if len(set(mapping.values())) != len(candidate_ids):
            raise UrduBakeoffError("Existing blind key contains duplicate codes")
        path.chmod(0o600)
        return {str(key): str(value) for key, value in mapping.items()}
    shuffled = list(candidate_ids)
    secrets.SystemRandom().shuffle(shuffled)
    codes = [f"Candidate {chr(ord('A') + index)}" for index in range(len(shuffled))]
    mapping = {candidate_id: code for candidate_id, code in zip(shuffled, codes)}
    atomic_json(
        path,
        {
            "version": "urdu-translation-blind-key-v1",
            "bakeoff_id": BAKEOFF_ID,
            "mapping": mapping,
        },
    )
    path.chmod(0o600)
    return mapping


def _manifest(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "urdu-translation-bakeoff-v1",
        "bakeoff_id": BAKEOFF_ID,
        "corpus": {
            "passages": [asdict(passage) for passage in PASSAGES],
            "passage_count": len(PASSAGES),
            "ayah_count": sum(len(passage.expected_ayahs) for passage in PASSAGES),
            "payload_hash": stable_hash(payloads),
        },
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "auditors": [asdict(auditor) for auditor in AUDITORS],
        "generation": {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "contract_attempts": CONTRACT_ATTEMPTS,
            "temperature": 0,
        },
        "inputs": {
            "source_xml": file_hash(DEFAULT_SOURCE_XML),
            "morphology": file_hash(MORPHOLOGY_PATH),
            "policy": file_hash(POLICY_PATH),
            "audit_policy": file_hash(AUDIT_PATH),
            "ledger_md": file_hash(LEDGER_MD_PATH),
            "ledger_json": file_hash(LEDGER_JSON_PATH),
            "reading_notes": file_hash(NOTES_PATH),
            "release_adjudications": file_hash(ADJUDICATIONS_PATH),
            "preregistration": file_hash(PREREGISTRATION_PATH),
            "harness": file_hash(Path(__file__)),
        },
    }


def prepare_bakeoff(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    required = (
        DEFAULT_SOURCE_XML,
        MORPHOLOGY_PATH,
        POLICY_PATH,
        AUDIT_PATH,
        LEDGER_MD_PATH,
        LEDGER_JSON_PATH,
        NOTES_PATH,
        ADJUDICATIONS_PATH,
        PREREGISTRATION_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise UrduBakeoffError(f"Missing frozen bakeoff inputs: {missing}")
    payloads = build_passage_payloads()
    root.mkdir(parents=True, exist_ok=True)
    inputs_dir = root / "inputs"
    for payload in payloads:
        path = inputs_dir / f"{payload['passage']['passage_id']}.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
            raise UrduBakeoffError(f"Frozen passage payload changed: {path}")
        atomic_json(path, payload)
    manifest = _manifest(payloads)
    manifest_path = root / "MANIFEST.json"
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != manifest:
            raise UrduBakeoffError(
                "Bakeoff manifest changed; use a new root rather than mixing versions"
            )
    else:
        atomic_json(manifest_path, manifest)
    _private_blind_key(root)
    state = bakeoff_status(root, verify_manifest=False)
    atomic_json(root / "RUN.json", state)
    return state


def _assert_manifest(root: Path) -> dict[str, Any]:
    path = root / "MANIFEST.json"
    if not path.is_file():
        raise UrduBakeoffError("Bakeoff is not prepared")
    current = json.loads(path.read_text(encoding="utf-8"))
    payloads = [
        json.loads((root / "inputs" / f"{passage.passage_id}.json").read_text(encoding="utf-8"))
        for passage in PASSAGES
    ]
    if current != _manifest(payloads):
        raise UrduBakeoffError("Manifest or frozen inputs changed; refusing mixed-version resume")
    return current


def extract_json(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _normalize_quoted_urdu(text: str) -> str:
    return _normalize(text.strip().strip("\"'“”‘’«»"))


def _normalize_arabic_ground(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
        and character != "ـ"
        and not "\u06d6" <= character <= "\u06ed"
    )
    orthographic_folds = str.maketrans(
        {
            "ٱ": "ا",
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ى": "ي",
            "ی": "ي",
            "ئ": "ي",
            "ؤ": "و",
        }
    )
    return _normalize(without_marks.translate(orthographic_folds))


def validate_translation(document: Any, expected: list[int]) -> dict[str, Any] | None:
    if not isinstance(document, dict) or set(document) != {"ayahs"}:
        return None
    rows = document.get("ayahs")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None
    if [row.get("ayah") for row in rows if isinstance(row, dict)] != expected:
        return None
    clean: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"ayah", "urdu", "review_flags"}:
            return None
        urdu = row.get("urdu")
        flags = row.get("review_flags")
        if not isinstance(urdu, str) or not urdu.strip():
            return None
        if not re.search(r"[\u0600-\u06ff]", urdu):
            return None
        if re.search(r"[\u0900-\u097f]", urdu):
            return None
        if (
            not isinstance(flags, list)
            or any(not isinstance(flag, str) or flag not in ALLOWED_FLAGS for flag in flags)
        ):
            return None
        clean.append(
            {
                "ayah": int(row["ayah"]),
                "urdu": urdu.strip(),
                "review_flags": list(dict.fromkeys(flags)),
            }
        )
    return {"ayahs": clean}


def validate_audit(
    document: Any,
    *,
    codes: list[str],
    expected: list[int],
    urdu_by_code: dict[str, dict[int, str]],
    arabic_by_ayah: dict[int, str],
) -> dict[str, Any] | None:
    if not isinstance(document, dict) or set(document) != {"candidates"}:
        return None
    candidates = document.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(codes):
        return None
    if [item.get("code") for item in candidates if isinstance(item, dict)] != codes:
        return None
    clean_candidates: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict) or set(item) != {"code", "ayahs", "scores", "summary"}:
            return None
        code = item["code"]
        ayahs = item.get("ayahs")
        scores = item.get("scores")
        summary = item.get("summary")
        if not isinstance(ayahs, list) or len(ayahs) != len(expected):
            return None
        if [row.get("ayah") for row in ayahs if isinstance(row, dict)] != expected:
            return None
        if not isinstance(scores, dict) or set(scores) != SCORE_FIELDS:
            return None
        if any(not isinstance(value, int) or not 1 <= value <= 5 for value in scores.values()):
            return None
        if not isinstance(summary, str) or not summary.strip():
            return None
        clean_ayahs: list[dict[str, Any]] = []
        for row in ayahs:
            if not isinstance(row, dict) or set(row) != {"ayah", "findings", "verdict"}:
                return None
            findings = row.get("findings")
            verdict = row.get("verdict")
            if not isinstance(findings, list) or verdict not in {"pass", "revise"}:
                return None
            clean_findings: list[dict[str, str]] = []
            for finding in findings:
                required = {
                    "type",
                    "severity",
                    "where",
                    "arabic_ground",
                    "explanation",
                    "suggestion",
                }
                if not isinstance(finding, dict) or set(finding) != required:
                    return None
                if finding["type"] not in FINDING_TYPES or finding["severity"] not in SEVERITIES:
                    return None
                if any(
                    not isinstance(finding[field], str)
                    or (field != "suggestion" and not finding[field].strip())
                    for field in required - {"type", "severity"}
                ):
                    return None
                if finding["where"] == "<missing>":
                    if finding["type"] != "omission":
                        return None
                elif _normalize_quoted_urdu(finding["where"]) not in _normalize(
                    urdu_by_code[code][int(row["ayah"])]
                ):
                    return None
                if _normalize_arabic_ground(
                    finding["arabic_ground"]
                ) not in _normalize_arabic_ground(
                    arabic_by_ayah[int(row["ayah"])]
                ):
                    return None
                clean_findings.append({key: str(finding[key]).strip() for key in required})
            major = any(
                finding["severity"] in {"blocking", "significant"}
                for finding in clean_findings
            )
            if major and verdict != "revise":
                return None
            if not clean_findings and verdict != "pass":
                return None
            clean_ayahs.append(
                {"ayah": int(row["ayah"]), "findings": clean_findings, "verdict": verdict}
            )
        clean_candidates.append(
            {
                "code": code,
                "ayahs": clean_ayahs,
                "scores": {field: int(scores[field]) for field in sorted(SCORE_FIELDS)},
                "summary": summary.strip(),
            }
        )
    return {"candidates": clean_candidates}


def _translation_system() -> str:
    return (
        POLICY_PATH.read_text(encoding="utf-8").strip()
        + "\n\n=== MODEL-FACING SEMANTIC LEDGER ===\n"
        + LEDGER_MD_PATH.read_text(encoding="utf-8").strip()
        + "\n\n=== STRUCTURED SENSE RECORDS ===\n"
        + LEDGER_JSON_PATH.read_text(encoding="utf-8").strip()
    )


def _translation_user(payload: dict[str, Any], retry_error: str | None = None) -> str:
    passage = payload["passage"]
    context = "\n".join(
        f"({row['ayah']}) {row['arabic']}" for row in payload["local_context"]
    )
    target = "\n".join(f"({row['ayah']}) {row['arabic']}" for row in payload["target"])
    records = json.dumps(
        payload["adjudicated_semantic_records"], ensure_ascii=False, indent=2
    )
    retry = (
        "\n\nThe previous response failed the registered contract: "
        f"{retry_error}. Return a corrected JSON object only."
        if retry_error
        else ""
    )
    return (
        f"=== EVIDENCE PACKET ===\n{payload['evidence_packet']}\n\n"
        f"=== ADJUDICATED SEMANTIC RECORDS ===\n{records}\n\n"
        f"=== LOCAL ARABIC CONTEXT: SURAH {passage['surah']} ===\n{context}\n\n"
        f"=== TARGET AYAHS: {passage['surah']}:{passage['first_ayah']}-"
        f"{passage['last_ayah']} ===\n{target}\n\n"
        "Translate only the target ayahs directly from Arabic. Return the registered "
        "JSON object with one Urdu row per target ayah."
        + retry
    )


def _request_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int = 600,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise UrduBakeoffError("Provider returned non-object JSON")
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:3000]
            if exc.code in RETRYABLE_HTTP and attempt < 3:
                time.sleep(min(10 * (attempt + 1), 30))
                continue
            raise UrduBakeoffError(f"Provider HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ssl.SSLError) as exc:
            if attempt == 3:
                raise UrduBakeoffError(f"Provider request failed: {exc}") from exc
            time.sleep(min(10 * (attempt + 1), 30))
    raise AssertionError("unreachable")


def _call_anthropic(
    model: ModelSpec, system: str, user: str, schema: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise UrduBakeoffError("ANTHROPIC_API_KEY is missing")
    response = _request_json(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        payload={
            "model": model.model_id,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": model.reasoning,
                "format": {"type": "json_schema", "schema": schema},
            },
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ],
            "messages": [{"role": "user", "content": user}],
        },
    )
    text = "".join(
        str(block.get("text") or "")
        for block in response.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    if not text.strip():
        raise UrduBakeoffError("Anthropic response contains no text")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return text, usage, response


def _call_google(
    model: ModelSpec, system: str, user: str, schema: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise UrduBakeoffError("GOOGLE_API_KEY is missing")
    try:
        from google import genai
    except ImportError as exc:
        raise UrduBakeoffError("google-genai is not installed") from exc
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model.model_id,
        contents=user,
        config={
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "thinking_config": {"thinking_level": "HIGH"},
        },
    )
    text = str(getattr(response, "text", "") or "")
    if not text.strip():
        raise UrduBakeoffError("Google response contains no text")
    raw = (
        response.model_dump(mode="json")
        if hasattr(response, "model_dump")
        else {"repr": repr(response)}
    )
    usage_obj = getattr(response, "usage_metadata", None)
    usage = (
        usage_obj.model_dump(mode="json")
        if hasattr(usage_obj, "model_dump")
        else {}
    )
    return text, usage, raw


def _openai_output_text(response: dict[str, Any]) -> str:
    values: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                values.append(str(content.get("text") or ""))
    return "".join(values)


def _call_openai(
    model: ModelSpec, system: str, user: str, schema: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise UrduBakeoffError("OPENAI_API_KEY is missing")
    response = _request_json(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {key}"},
        payload={
            "model": model.model_id,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            "reasoning": {"effort": model.reasoning},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "quran_urdu_bakeoff",
                    "strict": True,
                    "schema": schema,
                }
            },
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )
    text = _openai_output_text(response)
    if not text.strip():
        raise UrduBakeoffError("OpenAI response contains no output text")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return text, usage, response


def _call_openrouter(
    model: ModelSpec, system: str, user: str, schema: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise UrduBakeoffError("OPENROUTER_API_KEY is missing")
    response = _request_json(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://localhost/quran-translation-v2",
            "X-Title": "Quran Urdu Translation Bakeoff",
        },
        payload={
            "model": model.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "reasoning": {"effort": model.reasoning, "exclude": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "quran_urdu_bakeoff",
                    "strict": True,
                    "schema": schema,
                },
            },
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
        },
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UrduBakeoffError("OpenRouter response contains no choices")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, list):
        text = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    else:
        text = str(content or "")
    if not text.strip():
        raise UrduBakeoffError("OpenRouter response contains no text")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return text, usage, response


PROVIDER_CALLS: dict[
    str,
    Callable[[ModelSpec, str, str, dict[str, Any]], tuple[str, dict[str, Any], dict[str, Any]]],
] = {
    "anthropic": _call_anthropic,
    "google": _call_google,
    "openai": _call_openai,
    "openrouter": _call_openrouter,
}


def _private_result_path(root: Path, candidate: ModelSpec, passage: PassageSpec) -> Path:
    return root / "private" / "translations" / candidate.candidate_id / f"{passage.passage_id}.json"


def _translation_input_hash(candidate: ModelSpec, payload: dict[str, Any]) -> str:
    return stable_hash(
        {
            "candidate": asdict(candidate),
            "policy": file_hash(POLICY_PATH),
            "ledger_md": file_hash(LEDGER_MD_PATH),
            "ledger_json": file_hash(LEDGER_JSON_PATH),
            "payload": payload,
            "schema": TRANSLATION_SCHEMA,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "contract_attempts": CONTRACT_ATTEMPTS,
        }
    )


def _load_complete_translation(
    path: Path, input_hash: str, expected: list[int]
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("input_hash") != input_hash:
        raise UrduBakeoffError(f"Cached translation input mismatch: {path}")
    if document.get("status") != "complete":
        return None
    result = validate_translation(document.get("result"), expected)
    if result is None:
        raise UrduBakeoffError(f"Cached translation fails contract: {path}")
    return document


def _generate_translation_job(
    root: Path,
    candidate: ModelSpec,
    passage: PassageSpec,
    payload: dict[str, Any],
) -> dict[str, Any]:
    path = _private_result_path(root, candidate, passage)
    input_hash = _translation_input_hash(candidate, payload)
    existing = _load_complete_translation(path, input_hash, passage.expected_ayahs)
    if existing is not None:
        return {"candidate_id": candidate.candidate_id, "passage_id": passage.passage_id, "status": "reused"}
    if path.exists():
        failed = json.loads(path.read_text(encoding="utf-8"))
        if failed.get("input_hash") != input_hash:
            raise UrduBakeoffError(f"Cached failed translation input mismatch: {path}")
        if int(failed.get("attempts", 0)) >= CONTRACT_ATTEMPTS:
            return {"candidate_id": candidate.candidate_id, "passage_id": passage.passage_id, "status": "failed"}

    errors: list[str] = []
    started = time.monotonic()
    raw_response: dict[str, Any] | None = None
    usage: dict[str, Any] = {}
    raw_text = ""
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        try:
            user = _translation_user(payload, errors[-1] if errors else None)
            raw_text, usage, raw_response = PROVIDER_CALLS[candidate.provider](
                candidate,
                _translation_system(),
                user,
                TRANSLATION_SCHEMA,
            )
            parsed = extract_json(raw_text)
            result = validate_translation(parsed, passage.expected_ayahs)
            if result is None:
                errors.append("response failed exact ayah, field, Urdu-script, or flag contract")
                continue
            atomic_json(
                path,
                {
                    "version": "urdu-translation-result-v1",
                    "input_hash": input_hash,
                    "status": "complete",
                    "candidate": asdict(candidate),
                    "passage": asdict(passage),
                    "attempts": attempt,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "usage": usage,
                    "result": result,
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                    "errors_before_success": errors,
                },
            )
            return {"candidate_id": candidate.candidate_id, "passage_id": passage.passage_id, "status": "complete"}
        except Exception as exc:  # Provider classes vary; persist a bounded diagnostic.
            errors.append(f"{type(exc).__name__}: {exc}"[:3000])
    atomic_json(
        path,
        {
            "version": "urdu-translation-result-v1",
            "input_hash": input_hash,
            "status": "failed",
            "candidate": asdict(candidate),
            "passage": asdict(passage),
            "attempts": CONTRACT_ATTEMPTS,
            "latency_seconds": round(time.monotonic() - started, 3),
            "usage": usage,
            "raw_text": raw_text,
            "raw_response": raw_response,
            "errors": errors,
        },
    )
    return {"candidate_id": candidate.candidate_id, "passage_id": passage.passage_id, "status": "failed"}


def run_generation(
    root: Path = DEFAULT_ROOT,
    *,
    workers: int = DEFAULT_WORKERS,
    candidate_ids: set[str] | None = None,
) -> dict[str, Any]:
    prepare_bakeoff(root)
    _assert_manifest(root)
    load_environment()
    known_ids = {candidate.candidate_id for candidate in CANDIDATES}
    requested_ids = known_ids if candidate_ids is None else set(candidate_ids)
    unknown_ids = requested_ids - known_ids
    if unknown_ids:
        raise UrduBakeoffError(
            "Unknown candidate IDs: " + ", ".join(sorted(unknown_ids))
        )
    selected_candidates = tuple(
        candidate for candidate in CANDIDATES if candidate.candidate_id in requested_ids
    )
    if not selected_candidates:
        raise UrduBakeoffError("At least one candidate must be selected")
    payloads = {
        passage.passage_id: json.loads(
            (root / "inputs" / f"{passage.passage_id}.json").read_text(encoding="utf-8")
        )
        for passage in PASSAGES
    }
    pending: list[tuple[ModelSpec, PassageSpec, dict[str, Any]]] = []
    for passage in PASSAGES:
        for candidate in selected_candidates:
            path = _private_result_path(root, candidate, passage)
            input_hash = _translation_input_hash(candidate, payloads[passage.passage_id])
            if _load_complete_translation(path, input_hash, passage.expected_ayahs) is None:
                if path.exists() and int(
                    json.loads(path.read_text(encoding="utf-8")).get("attempts", 0)
                ) >= CONTRACT_ATTEMPTS:
                    continue
                pending.append((candidate, passage, payloads[passage.passage_id]))

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(_generate_translation_job, root, candidate, passage, payload): (
                    candidate,
                    passage,
                )
                for candidate, passage, payload in pending
            }
            for future in as_completed(futures):
                candidate, passage = futures[future]
                outcome = future.result()
                print(
                    f"translation {outcome['status']}: {candidate.candidate_id} "
                    f"{passage.passage_id}",
                    flush=True,
                )
    status = bakeoff_status(root)
    if status["generation"]["failed"] == 0 and status["generation"]["complete"] == status["generation"]["total"]:
        package_blind_workbook(root)
        status = bakeoff_status(root)
    atomic_json(root / "RUN.json", status)
    return status


def seed_generation(
    source_root: Path,
    destination_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    prepare_bakeoff(destination_root)
    _assert_manifest(destination_root)
    payloads = {
        passage.passage_id: json.loads(
            (destination_root / "inputs" / f"{passage.passage_id}.json").read_text(
                encoding="utf-8"
            )
        )
        for passage in PASSAGES
    }
    for passage in PASSAGES:
        for candidate in CANDIDATES:
            input_hash = _translation_input_hash(
                candidate, payloads[passage.passage_id]
            )
            source_path = _private_result_path(source_root, candidate, passage)
            source_document = _load_complete_translation(
                source_path, input_hash, passage.expected_ayahs
            )
            if source_document is None:
                raise UrduBakeoffError(
                    "Cannot seed incomplete translation: "
                    f"{candidate.candidate_id}/{passage.passage_id}"
                )
            destination_path = _private_result_path(
                destination_root, candidate, passage
            )
            if destination_path.exists():
                destination_document = _load_complete_translation(
                    destination_path, input_hash, passage.expected_ayahs
                )
                if destination_document != source_document:
                    raise UrduBakeoffError(
                        f"Seed destination differs from source: {destination_path}"
                    )
            else:
                atomic_json(destination_path, source_document)
    package_blind_workbook(destination_root)
    status = bakeoff_status(destination_root)
    atomic_json(destination_root / "RUN.json", status)
    return status


def _workbook_html(manifest: dict[str, Any]) -> str:
    data = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arabic-to-Urdu Quran Translation Bakeoff</title>
<style>
:root{{--ink:#171918;--muted:#626761;--paper:#f8f8f5;--white:#fff;--green:#174f3b;--burgundy:#7b2639;--gold:#b68a36;--line:#d9ddd7}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,system-ui,-apple-system,sans-serif;letter-spacing:0}}
header{{background:var(--green);color:white;padding:22px 5vw;border-bottom:5px solid var(--gold)}}
header h1{{margin:0 0 6px;font-size:28px;font-weight:750}} header p{{margin:0;max-width:850px;color:#eef6f1}}
nav{{position:sticky;top:0;z-index:2;background:var(--white);border-bottom:1px solid var(--line);padding:12px 5vw;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
select,button,input,textarea{{font:inherit}} select,button{{min-height:40px;border:1px solid #aeb6ad;border-radius:6px;background:white;padding:7px 11px}}
button{{cursor:pointer;background:var(--burgundy);border-color:var(--burgundy);color:white;font-weight:700}} button.secondary{{background:white;color:var(--green);border-color:var(--green)}}
main{{max-width:1280px;margin:0 auto;padding:24px 5vw 60px}} .focus{{color:var(--muted);margin:4px 0 22px}}
.candidate{{background:var(--white);border:1px solid var(--line);border-left:5px solid var(--gold);border-radius:6px;margin:0 0 22px;padding:18px}}
.candidate h2{{margin:0 0 14px;color:var(--green);font-size:21px}} .ayah{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px;border-top:1px solid var(--line);padding:15px 0}}
.arabic,.urdu{{font-family:"Noto Naskh Arabic","Geeza Pro",serif;direction:rtl;text-align:right;font-size:24px;line-height:1.8;overflow-wrap:anywhere}}
.ref{{display:block;font-family:Inter,system-ui,sans-serif;direction:ltr;text-align:left;color:var(--muted);font-size:12px;margin-bottom:4px}}
.scores{{display:grid;grid-template-columns:repeat(3,minmax(130px,1fr));gap:10px;margin-top:14px}} label{{display:grid;gap:5px;color:var(--muted);font-size:13px}}
input[type=number]{{width:100%;min-height:38px;border:1px solid #bcc3bb;border-radius:5px;padding:6px}} textarea{{width:100%;min-height:75px;border:1px solid #bcc3bb;border-radius:5px;padding:8px;resize:vertical}}
.note{{margin-top:12px}} .muted{{color:var(--muted)}}
@media(max-width:760px){{header h1{{font-size:23px}} .ayah{{grid-template-columns:1fr}} .scores{{grid-template-columns:1fr 1fr}} .arabic,.urdu{{font-size:21px}}}}
</style></head><body>
<header><h1>Arabic-to-Urdu Quran Translation Bakeoff</h1><p>Read without guessing the model. Fidelity questions belong in notes; score the Urdu for clarity, naturalness, force, and how well it would work aloud.</p></header>
<nav><label>Passage<select id="passage"></select></label><button id="export">Export scores</button><button class="secondary" id="clear">Clear local scores</button><span id="saved" class="muted"></span></nav>
<main><h1 id="title"></h1><p id="focus" class="focus"></p><div id="candidates"></div></main>
<script>
const DATA={data}; const KEY='urdu-bakeoff-v1-scores'; let saved=JSON.parse(localStorage.getItem(KEY)||'{{}}');
const passageSelect=document.getElementById('passage');
DATA.passages.forEach((p,i)=>{{const o=document.createElement('option');o.value=p.passage_id;o.textContent=`${{p.ref}} · ${{p.label}}`;passageSelect.appendChild(o)}});
function scoreKey(p,c,f){{return `${{p}}|${{c}}|${{f}}`}} function persist(){{localStorage.setItem(KEY,JSON.stringify(saved));document.getElementById('saved').textContent='Saved locally';}}
function render(){{const p=DATA.passages.find(x=>x.passage_id===passageSelect.value)||DATA.passages[0];passageSelect.value=p.passage_id;document.getElementById('title').textContent=`${{p.ref}} · ${{p.label}}`;document.getElementById('focus').textContent=p.focus;const host=document.getElementById('candidates');host.innerHTML='';
p.candidates.forEach(c=>{{const section=document.createElement('section');section.className='candidate';const h=document.createElement('h2');h.textContent=c.code;section.appendChild(h);c.ayahs.forEach((row,i)=>{{const a=document.createElement('div');a.className='ayah';const ar=document.createElement('div');ar.className='arabic';const arRef=document.createElement('span');arRef.className='ref';arRef.textContent=`${{p.ref.split(':')[0]}}:${{row.ayah}}`;ar.appendChild(arRef);ar.appendChild(document.createTextNode(p.arabic[i].arabic));const ur=document.createElement('div');ur.className='urdu';const urRef=document.createElement('span');urRef.className='ref';urRef.textContent='Urdu';ur.appendChild(urRef);ur.appendChild(document.createTextNode(row.urdu));a.appendChild(ar);a.appendChild(ur);section.appendChild(a)}});const scores=document.createElement('div');scores.className='scores';['naturalness','clarity','source_force','spoken_quality','overall'].forEach(f=>{{const label=document.createElement('label');label.textContent=f.replaceAll('_',' ');const input=document.createElement('input');input.type='number';input.min=1;input.max=5;input.step=1;input.value=saved[scoreKey(p.passage_id,c.code,f)]||'';input.onchange=()=>{{saved[scoreKey(p.passage_id,c.code,f)]=input.value;persist()}};label.appendChild(input);scores.appendChild(label)}});section.appendChild(scores);const label=document.createElement('label');label.className='note';label.textContent='Notes';const ta=document.createElement('textarea');ta.value=saved[scoreKey(p.passage_id,c.code,'notes')]||'';ta.oninput=()=>{{saved[scoreKey(p.passage_id,c.code,'notes')]=ta.value;persist()}};label.appendChild(ta);section.appendChild(label);host.appendChild(section)}})}}
passageSelect.onchange=render;document.getElementById('clear').onclick=()=>{{if(confirm('Clear all locally saved scores?')){{saved={{}};localStorage.removeItem(KEY);render()}}}};
document.getElementById('export').onclick=()=>{{const payload={{bakeoff_id:DATA.bakeoff_id,exported_at:new Date().toISOString(),scores:saved}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='quran-urdu-translation-bakeoff-scores.json';a.click();URL.revokeObjectURL(a.href)}};
render();
</script></body></html>"""


def package_blind_workbook(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    _assert_manifest(root)
    mapping = _private_blind_key(root)
    passages_out: list[dict[str, Any]] = []
    for passage in PASSAGES:
        payload = json.loads(
            (root / "inputs" / f"{passage.passage_id}.json").read_text(encoding="utf-8")
        )
        candidates_out: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            path = _private_result_path(root, candidate, passage)
            input_hash = _translation_input_hash(candidate, payload)
            document = _load_complete_translation(path, input_hash, passage.expected_ayahs)
            if document is None:
                raise UrduBakeoffError(f"Cannot blind incomplete translation: {candidate.candidate_id}/{passage.passage_id}")
            candidates_out.append(
                {"code": mapping[candidate.candidate_id], "ayahs": document["result"]["ayahs"]}
            )
        candidates_out.sort(key=lambda item: item["code"])
        info = surah_info(passage.surah)
        passages_out.append(
            {
                "passage_id": passage.passage_id,
                "ref": f"{passage.surah}:{passage.first_ayah}-{passage.last_ayah}",
                "label": f"{info.transliteration} ({info.meaning})",
                "focus": passage.focus,
                "arabic": payload["target"],
                "candidates": candidates_out,
            }
        )
    manifest = {
        "version": "urdu-translation-blind-workbook-v1",
        "bakeoff_id": BAKEOFF_ID,
        "candidate_codes": sorted(mapping.values()),
        "passages": passages_out,
    }
    blind = root / "blind"
    atomic_json(blind / "BLIND_MANIFEST.json", manifest)
    atomic_text(blind / "review.html", _workbook_html(manifest))
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in blind.rglob("*") if path.is_file()
    )
    private_terms = [
        value
        for candidate in CANDIDATES
        for value in (candidate.candidate_id, candidate.model_id, candidate.private_label)
    ]
    leaked = [term for term in private_terms if term and term in serialized]
    if leaked:
        raise UrduBakeoffError(f"Private model identity leaked into blind package: {leaked}")
    return manifest


def _audit_user(
    payload: dict[str, Any],
    passage_entry: dict[str, Any],
    retry_error: str | None = None,
) -> str:
    context = "\n".join(
        f"({row['ayah']}) {row['arabic']}" for row in payload["local_context"]
    )
    candidates = json.dumps(
        [
            {"code": candidate["code"], "ayahs": candidate["ayahs"]}
            for candidate in passage_entry["candidates"]
        ],
        ensure_ascii=False,
        indent=2,
    )
    records = json.dumps(
        payload["adjudicated_semantic_records"], ensure_ascii=False, indent=2
    )
    retry = (
        "\n\nThe previous response failed the registered audit contract: "
        f"{retry_error}. Return corrected JSON only."
        if retry_error
        else ""
    )
    return (
        f"=== EVIDENCE PACKET ===\n{payload['evidence_packet']}\n\n"
        f"=== ADJUDICATED SEMANTIC RECORDS ===\n{records}\n\n"
        f"=== LOCAL ARABIC CONTEXT ===\n{context}\n\n"
        f"=== BLINDED URDU CANDIDATES ===\n{candidates}\n\n"
        "Audit every code in the supplied order and every target ayah in order. "
        "Return only the registered JSON object."
        + retry
    )


def _audit_result_path(root: Path, auditor: ModelSpec, passage: PassageSpec) -> Path:
    return root / "private" / "audits" / auditor.candidate_id / f"{passage.passage_id}.json"


def _audit_input_hash(
    auditor: ModelSpec,
    payload: dict[str, Any],
    passage_entry: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "auditor": asdict(auditor),
            "audit_policy": file_hash(AUDIT_PATH),
            "payload": payload,
            "blinded_candidates": passage_entry["candidates"],
            "schema": audit_schema([item["code"] for item in passage_entry["candidates"]]),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "contract_attempts": CONTRACT_ATTEMPTS,
        }
    )


def _load_complete_audit(
    path: Path,
    *,
    input_hash: str,
    payload: dict[str, Any],
    passage_entry: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("input_hash") != input_hash:
        raise UrduBakeoffError(f"Cached audit input mismatch: {path}")
    if document.get("status") != "complete":
        return None
    codes = [item["code"] for item in passage_entry["candidates"]]
    expected = payload["passage"]["first_ayah"], payload["passage"]["last_ayah"]
    expected_ayahs = list(range(int(expected[0]), int(expected[1]) + 1))
    urdu_by_code = {
        item["code"]: {int(row["ayah"]): str(row["urdu"]) for row in item["ayahs"]}
        for item in passage_entry["candidates"]
    }
    arabic_by_ayah = {
        int(row["ayah"]): str(row["arabic"]) for row in payload["target"]
    }
    result = validate_audit(
        document.get("result"),
        codes=codes,
        expected=expected_ayahs,
        urdu_by_code=urdu_by_code,
        arabic_by_ayah=arabic_by_ayah,
    )
    if result is None:
        raise UrduBakeoffError(f"Cached audit fails contract: {path}")
    return document


def _generate_audit_job(
    root: Path,
    auditor: ModelSpec,
    passage: PassageSpec,
    payload: dict[str, Any],
    passage_entry: dict[str, Any],
) -> dict[str, Any]:
    path = _audit_result_path(root, auditor, passage)
    input_hash = _audit_input_hash(auditor, payload, passage_entry)
    existing = _load_complete_audit(
        path, input_hash=input_hash, payload=payload, passage_entry=passage_entry
    )
    if existing is not None:
        return {"auditor": auditor.candidate_id, "passage_id": passage.passage_id, "status": "reused"}
    if path.exists():
        failed = json.loads(path.read_text(encoding="utf-8"))
        if failed.get("input_hash") != input_hash:
            raise UrduBakeoffError(f"Cached failed audit input mismatch: {path}")
        if int(failed.get("attempts", 0)) >= CONTRACT_ATTEMPTS:
            return {"auditor": auditor.candidate_id, "passage_id": passage.passage_id, "status": "failed"}

    codes = [item["code"] for item in passage_entry["candidates"]]
    expected = passage.expected_ayahs
    urdu_by_code = {
        item["code"]: {int(row["ayah"]): str(row["urdu"]) for row in item["ayahs"]}
        for item in passage_entry["candidates"]
    }
    arabic_by_ayah = {int(row["ayah"]): str(row["arabic"]) for row in payload["target"]}
    schema = audit_schema(codes)
    errors: list[str] = []
    started = time.monotonic()
    raw_text = ""
    usage: dict[str, Any] = {}
    raw_response: dict[str, Any] | None = None
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        try:
            user = _audit_user(payload, passage_entry, errors[-1] if errors else None)
            raw_text, usage, raw_response = PROVIDER_CALLS[auditor.provider](
                auditor,
                AUDIT_PATH.read_text(encoding="utf-8"),
                user,
                schema,
            )
            parsed = extract_json(raw_text)
            result = validate_audit(
                parsed,
                codes=codes,
                expected=expected,
                urdu_by_code=urdu_by_code,
                arabic_by_ayah=arabic_by_ayah,
            )
            if result is None:
                errors.append("response failed exact code, ayah, quote, verdict, or score contract")
                continue
            atomic_json(
                path,
                {
                    "version": "urdu-translation-audit-result-v1",
                    "input_hash": input_hash,
                    "status": "complete",
                    "auditor": asdict(auditor),
                    "passage": asdict(passage),
                    "attempts": attempt,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "usage": usage,
                    "result": result,
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                    "errors_before_success": errors,
                },
            )
            return {"auditor": auditor.candidate_id, "passage_id": passage.passage_id, "status": "complete"}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}"[:3000])
    atomic_json(
        path,
        {
            "version": "urdu-translation-audit-result-v1",
            "input_hash": input_hash,
            "status": "failed",
            "auditor": asdict(auditor),
            "passage": asdict(passage),
            "attempts": CONTRACT_ATTEMPTS,
            "latency_seconds": round(time.monotonic() - started, 3),
            "usage": usage,
            "raw_text": raw_text,
            "raw_response": raw_response,
            "errors": errors,
        },
    )
    return {"auditor": auditor.candidate_id, "passage_id": passage.passage_id, "status": "failed"}


def _severity_weight(severity: str) -> float:
    return {"blocking": 3.0, "significant": 1.0, "minor": 0.25}[severity]


def build_audit_summary(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    workbook = json.loads((root / "blind" / "BLIND_MANIFEST.json").read_text(encoding="utf-8"))
    passage_by_id = {entry["passage_id"]: entry for entry in workbook["passages"]}
    findings_by_key: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    score_values: dict[str, dict[str, list[int]]] = {
        code: {field: [] for field in SCORE_FIELDS} for code in workbook["candidate_codes"]
    }
    auditor_reports: list[dict[str, Any]] = []
    for auditor_index, auditor in enumerate(AUDITORS, start=1):
        auditor_label = f"Auditor {auditor_index}"
        counts = {code: {severity: 0 for severity in sorted(SEVERITIES)} for code in workbook["candidate_codes"]}
        for passage in PASSAGES:
            payload = json.loads((root / "inputs" / f"{passage.passage_id}.json").read_text(encoding="utf-8"))
            passage_entry = passage_by_id[passage.passage_id]
            path = _audit_result_path(root, auditor, passage)
            input_hash = _audit_input_hash(auditor, payload, passage_entry)
            document = _load_complete_audit(
                path,
                input_hash=input_hash,
                payload=payload,
                passage_entry=passage_entry,
            )
            if document is None:
                raise UrduBakeoffError(f"Cannot summarize incomplete audit: {auditor.candidate_id}/{passage.passage_id}")
            for candidate in document["result"]["candidates"]:
                code = candidate["code"]
                for field, value in candidate["scores"].items():
                    score_values[code][field].append(int(value))
                for ayah in candidate["ayahs"]:
                    for finding in ayah["findings"]:
                        counts[code][finding["severity"]] += 1
                        key = (code, passage.passage_id, int(ayah["ayah"]), finding["type"])
                        findings_by_key.setdefault(key, []).append(
                            {"auditor": auditor_label, **finding}
                        )
        auditor_reports.append({"auditor": auditor_label, "finding_counts": counts})

    consensus: list[dict[str, Any]] = []
    singletons: list[dict[str, Any]] = []
    weighted = {code: 0.0 for code in workbook["candidate_codes"]}
    for (code, passage_id, ayah, finding_type), records in sorted(findings_by_key.items()):
        severity = max(records, key=lambda row: _severity_weight(row["severity"]))["severity"]
        item = {
            "code": code,
            "passage_id": passage_id,
            "ayah": ayah,
            "type": finding_type,
            "severity": severity,
            "records": records,
        }
        if len({record["auditor"] for record in records}) == len(AUDITORS):
            consensus.append(item)
            weighted[code] += _severity_weight(severity)
        else:
            singletons.append(item)

    mean_scores = {
        code: {
            field: round(sum(values) / len(values), 3) if values else None
            for field, values in fields.items()
        }
        for code, fields in score_values.items()
    }
    summary = {
        "version": "urdu-translation-blind-audit-summary-v1",
        "bakeoff_id": BAKEOFF_ID,
        "auditors": [f"Auditor {index}" for index in range(1, len(AUDITORS) + 1)],
        "candidate_codes": workbook["candidate_codes"],
        "weighted_consensus_defects": weighted,
        "mean_language_scores": mean_scores,
        "consensus_findings": consensus,
        "singleton_findings": singletons,
        "auditor_reports": auditor_reports,
        "warning": "Automated audit is triage, not scholarly adjudication. Model identities remain sealed.",
    }
    atomic_json(root / "blind" / "AUDIT_SUMMARY.json", summary)
    return summary


def run_audits(
    root: Path = DEFAULT_ROOT,
    *,
    workers: int = 2,
) -> dict[str, Any]:
    _assert_manifest(root)
    if not (root / "blind" / "BLIND_MANIFEST.json").is_file():
        package_blind_workbook(root)
    load_environment()
    workbook = json.loads((root / "blind" / "BLIND_MANIFEST.json").read_text(encoding="utf-8"))
    passage_by_id = {entry["passage_id"]: entry for entry in workbook["passages"]}
    jobs: list[tuple[ModelSpec, PassageSpec, dict[str, Any], dict[str, Any]]] = []
    for passage in PASSAGES:
        payload = json.loads((root / "inputs" / f"{passage.passage_id}.json").read_text(encoding="utf-8"))
        passage_entry = passage_by_id[passage.passage_id]
        for auditor in AUDITORS:
            path = _audit_result_path(root, auditor, passage)
            input_hash = _audit_input_hash(auditor, payload, passage_entry)
            if _load_complete_audit(
                path,
                input_hash=input_hash,
                payload=payload,
                passage_entry=passage_entry,
            ) is None:
                if path.exists() and int(
                    json.loads(path.read_text(encoding="utf-8")).get("attempts", 0)
                ) >= CONTRACT_ATTEMPTS:
                    continue
                jobs.append((auditor, passage, payload, passage_entry))
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    _generate_audit_job, root, auditor, passage, payload, passage_entry
                ): (auditor, passage)
                for auditor, passage, payload, passage_entry in jobs
            }
            for future in as_completed(futures):
                auditor, passage = futures[future]
                outcome = future.result()
                print(
                    f"audit {outcome['status']}: {auditor.candidate_id} {passage.passage_id}",
                    flush=True,
                )
    status = bakeoff_status(root)
    if status["audits"]["failed"] == 0 and status["audits"]["complete"] == status["audits"]["total"]:
        build_audit_summary(root)
        status = bakeoff_status(root)
    atomic_json(root / "RUN.json", status)
    return status


def _usage_totals(root: Path) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for path in (root / "private").rglob("*.json") if (root / "private").exists() else []:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("status") != "complete" or not isinstance(document.get("usage"), dict):
            continue
        identity = document.get("candidate") or document.get("auditor") or {}
        model_id = str(identity.get("model_id") or "unknown")
        bucket = totals.setdefault(model_id, {})
        for key, value in document["usage"].items():
            if isinstance(value, (int, float)):
                bucket[key] = bucket.get(key, 0.0) + float(value)
    return totals


def bakeoff_status(root: Path = DEFAULT_ROOT, *, verify_manifest: bool = True) -> dict[str, Any]:
    if verify_manifest:
        _assert_manifest(root)
    generation = {"complete": 0, "failed": 0, "pending": 0, "total": len(PASSAGES) * len(CANDIDATES)}
    for passage in PASSAGES:
        for candidate in CANDIDATES:
            path = _private_result_path(root, candidate, passage)
            if not path.exists():
                generation["pending"] += 1
                continue
            status = json.loads(path.read_text(encoding="utf-8")).get("status")
            generation["complete" if status == "complete" else "failed"] += 1
    audits = {"complete": 0, "failed": 0, "pending": 0, "total": len(PASSAGES) * len(AUDITORS)}
    for passage in PASSAGES:
        for auditor in AUDITORS:
            path = _audit_result_path(root, auditor, passage)
            if not path.exists():
                audits["pending"] += 1
                continue
            status = json.loads(path.read_text(encoding="utf-8")).get("status")
            audits["complete" if status == "complete" else "failed"] += 1
    if generation["failed"] or audits["failed"]:
        state = "blocked"
    elif audits["complete"] == audits["total"] and (root / "blind" / "AUDIT_SUMMARY.json").is_file():
        state = "ready_for_blind_review"
    elif generation["complete"] == generation["total"]:
        state = "generation_complete"
    else:
        state = "prepared"
    return {
        "version": "urdu-translation-bakeoff-status-v1",
        "bakeoff_id": BAKEOFF_ID,
        "status": state,
        "generation": generation,
        "audits": audits,
        "workbook": str(root / "blind" / "review.html"),
        "audit_summary": str(root / "blind" / "AUDIT_SUMMARY.json"),
        "usage": _usage_totals(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "audit", "status"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidate_ids",
        help="Run only this candidate ID; repeat to select multiple candidates.",
    )
    parser.add_argument(
        "--seed-generation-from",
        type=Path,
        help="For prepare, import contract-valid completed translations from this run.",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        result = (
            seed_generation(args.seed_generation_from, args.root)
            if args.seed_generation_from
            else prepare_bakeoff(args.root)
        )
    elif args.command == "run":
        result = run_generation(
            args.root,
            workers=args.workers,
            candidate_ids=set(args.candidate_ids) if args.candidate_ids else None,
        )
    elif args.command == "audit":
        result = run_audits(args.root, workers=min(args.workers, 2))
    else:
        result = bakeoff_status(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
