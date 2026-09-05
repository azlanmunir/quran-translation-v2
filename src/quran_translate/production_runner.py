"""Resumable v2.4 full-Quran translation production pipeline.

The production path is deliberately phase-based:

1. Opus draft through Anthropic Message Batches.
2. Gemini forensic fidelity critic through Gemini Batch API.
3. Opus constrained revision only where findings exist.
4. Gemini verification of revised units.
5. One bounded Opus repair for any verification findings.

Every provider request and validated result is content-addressed on disk. No phase
regenerates a validated unit, and a restart polls existing provider jobs rather than
submitting duplicates.
"""

from __future__ import annotations

from .production_clients import submit_batch_once
from .state_safety import exclusive_lock

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import (
    DATA_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_SOURCE_XML,
    PROJECT_ROOT,
    load_dotenv,
)
from .critic_v2 import (
    CRITIC_JSON_SCHEMA,
    CRITIC_SYSTEM,
    canonicalize_critic,
    validate_critic,
)
from .db import connect, init_db, utc_now
from .production_clients import (
    AnthropicBatchClient,
    BatchState,
    GeminiBatchClient,
    GeminiSynchronousClient,
    ProviderError,
    anthropic_result_text,
)
from .production_packets import (
    ProductionUnit,
    atomic_json,
    atomic_text,
    build_units,
    source_verses,
    write_packets,
)
from .production_refrains import resolve_refrains
from .production_quality import run_production_quality_gate
from .refrains import repeated_ayah_groups
from .spoken_english_v1 import (
    SPOKEN_ENGLISH_SCHEMA,
    SPOKEN_ENGLISH_SYSTEM,
    validate_spoken_english,
)
from .validation import validate_source


PRODUCTION_ROOT = DATA_DIR / "work" / "production-v2.4"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "production-v2.4.md"
LEDGER_MD_PATH = PROJECT_ROOT / "prompts" / "sense-ledger-v2.4.md"
LEDGER_JSON_PATH = DATA_DIR / "evidence" / "sense-ledger-v2.4.json"
MORPHOLOGY_PATH = DATA_DIR / "evidence" / "qac-morphology.txt"
REFRAIN_POLICY_PATH = DATA_DIR / "evidence" / "refrain-policy-v1.json"

ANTHROPIC_MODEL = "claude-opus-4-6"
GEMINI_MODEL = "gemini-3.1-pro-preview"
PROMPT_VERSION = "production-v2.4-opus-gemini-transport-v1"
CANONICAL_SOURCE_SHA256 = (
    "f78067cd98c51c03e450581e1e8713f4e7c352e0b62a4fe5c35811da28dd23bf"
)
CANONICAL_MORPHOLOGY_SHA256 = (
    "742bfac59941b2cb09736d5b7aae694af50792261fb8450cbf6afafcc340645f"
)

DEFAULT_MAX_AYAHS = 32
DEFAULT_MAX_ARABIC_CHARS = 6_500
DEFAULT_CONTEXT_AYAHS = 3
DEFAULT_SHARD_SIZE = 254
ANTHROPIC_MAX_TOKENS = 48_000
ANTHROPIC_EFFORT = "high"
GEMINI_MAX_TOKENS = 20_000
GEMINI_TRANSPORTS = {"batch", "sync"}
GEMINI_SYNC_DELAY_SECONDS = 0.5
CONTRACT_ATTEMPTS = 2
POLL_SECONDS = 60

ALLOWED_FLAGS = {
    "rare_word",
    "disputed_grammar",
    "disputed_sense",
    "legal",
    "loanword",
    "ledger_gap",
    "low_confidence",
}

READER_ROW_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ayah": {"type": "integer"},
        "english": {"type": "string"},
        "review_flags": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(ALLOWED_FLAGS)},
        },
    },
    "required": ["ayah", "english"],
    "additionalProperties": False,
}

READER_JSON_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": READER_ROW_JSON_SCHEMA,
}

REVISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ayahs": READER_JSON_SCHEMA,
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


REVISION_SYSTEM = """You are the constrained reviser for a Quran translation.
Fidelity has priority over preserved ambiguity, natural spoken English, and literary
force, in that order. You receive Arabic, evidence, a reader draft, and a forensic
critic report. The critic is fallible: inspect each finding against the Arabic and
evidence before acting.

For every finding, return exactly one decision:
- applied: make the smallest repair that resolves a valid finding;
- rejected: preserve the draft and give a precise Arabic/contextual reason;
- escalated: evidence cannot responsibly settle the issue; preserve the strongest
  supported English and add an appropriate internal review flag.

Preserve unflagged wording unless a local grammatical repair requires a change. Do
not add imagery, intensity, agency, causality, motive, specificity, sequence, moral
judgment, or historical claims. Do not add reader-facing notes.

The only permitted review_flags are: rare_word, disputed_grammar,
disputed_sense, legal, loanword, ledger_gap, and low_confidence. Preserve applicable
existing flags, add only values from that list, and never coin a custom flag.

Return ONLY this JSON object:
{"ayahs": [{"ayah": 1, "english": "...", "review_flags": ["optional"]}],
 "decisions": [{"finding_id": "f-1-0", "decision": "applied",
  "reason": "specific reason"}]}
"""

DRAFT_SYSTEM = """You are the primary translator. Follow the supplied production
translation policy and sense ledger exactly. Translate only the requested target
ayahs, preserve their order and identifiers, and return no prose outside the
registered JSON contract."""


class ProductionError(RuntimeError):
    """A production invariant failed."""


@dataclass(frozen=True)
class ProductionConfig:
    run_id: str
    max_ayahs: int = DEFAULT_MAX_AYAHS
    max_arabic_chars: int = DEFAULT_MAX_ARABIC_CHARS
    context_ayahs: int = DEFAULT_CONTEXT_AYAHS
    shard_size: int = DEFAULT_SHARD_SIZE
    poll_seconds: int = POLL_SECONDS
    seed_drafts_from: str | None = None
    gemini_transport: str = "batch"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def extract_json(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]|\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_reader(document: Any, expected: list[int]) -> list[dict[str, Any]] | None:
    if not isinstance(document, list) or len(document) != len(expected):
        return None
    if [row.get("ayah") for row in document if isinstance(row, dict)] != expected:
        return None
    clean: list[dict[str, Any]] = []
    for row in document:
        if not isinstance(row, dict) or set(row) - {"ayah", "english", "review_flags"}:
            return None
        english = row.get("english")
        flags = row.get("review_flags", [])
        if not isinstance(english, str) or not english.strip():
            return None
        if not isinstance(flags, list) or any(flag not in ALLOWED_FLAGS for flag in flags):
            return None
        item: dict[str, Any] = {"ayah": row["ayah"], "english": english.strip()}
        if flags:
            item["review_flags"] = list(dict.fromkeys(flags))
        clean.append(item)
    return clean


def reader_text(reader: list[dict[str, Any]]) -> dict[int, str]:
    return {int(row["ayah"]): str(row["english"]) for row in reader}


def validate_critic_response(
    document: Any,
    expected: list[int],
    reader: list[dict[str, Any]],
    arabic_by_ayah: dict[int, str] | None = None,
) -> list[dict[str, Any]] | None:
    candidate, _normalizations = canonicalize_critic(document, {})
    return validate_critic(
        candidate,
        expected,
        reader_text(reader),
        arabic_by_ayah,
    )


def unit_arabic(
    verses: dict[tuple[int, int], str],
    unit: ProductionUnit,
) -> dict[int, str]:
    return {ayah: verses[(unit.surah, ayah)] for ayah in unit.expected_ayahs}


def finding_records(critic: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "finding_id": f"f-{entry['ayah']}-{index}",
            "ayah": entry["ayah"],
            **finding,
        }
        for entry in critic
        for index, finding in enumerate(entry["findings"])
    ]


def validate_revision(
    document: Any,
    expected: list[int],
    finding_ids: list[str],
) -> dict[str, Any] | None:
    if not isinstance(document, dict):
        return None
    ayahs = validate_reader(document.get("ayahs"), expected)
    decisions = document.get("decisions")
    if ayahs is None or not isinstance(decisions, list):
        return None
    if len(decisions) != len(finding_ids):
        return None
    seen: set[str] = set()
    clean_decisions: list[dict[str, str]] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            return None
        finding_id = decision.get("finding_id")
        status = decision.get("decision")
        reason = decision.get("reason")
        if (
            finding_id not in finding_ids
            or finding_id in seen
            or status not in {"applied", "rejected", "escalated"}
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            return None
        seen.add(str(finding_id))
        clean_decisions.append(
            {
                "finding_id": str(finding_id),
                "decision": str(status),
                "reason": reason.strip(),
            }
        )
    if seen != set(finding_ids):
        return None
    return {"ayahs": ayahs, "decisions": clean_decisions}


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    research_env = Path.home() / "Downloads" / "quran-translation" / ".env"
    load_dotenv(research_env)


def run_dir(run_id: str) -> Path:
    return PRODUCTION_ROOT / run_id


def unit_dir(base: Path, unit: ProductionUnit) -> Path:
    return base / "units" / unit.unit_id


def artifact_path(base: Path, unit: ProductionUnit, stage: str) -> Path:
    return unit_dir(base, unit) / f"{stage}.json"


def load_artifact(path: Path, input_hash: str, validator: Callable[[Any], Any]) -> Any:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("input_hash") != input_hash:
        raise ProductionError(f"Cached stage input mismatch: {path}")
    result = validator(payload.get("result"))
    if result is None:
        raise ProductionError(f"Cached stage fails contract: {path}")
    return result


def prompt_material() -> tuple[str, str, str]:
    return (
        PROMPT_PATH.read_text(encoding="utf-8").strip(),
        LEDGER_MD_PATH.read_text(encoding="utf-8").strip(),
        LEDGER_JSON_PATH.read_text(encoding="utf-8").strip(),
    )


SEED_COMPATIBLE_INPUTS = {
    "source_xml",
    "morphology",
    "prompt",
    "ledger_md",
    "ledger_json",
    "refrain_policy",
}


def seed_draft_artifacts(
    *,
    target_base: Path,
    source_base: Path,
    units: list[ProductionUnit],
    input_hash_for_unit: Callable[[ProductionUnit], str],
) -> dict[str, Any]:
    """Import only contract-valid drafts from a semantically identical run."""
    if source_base.resolve() == target_base.resolve():
        raise ProductionError("A production run cannot seed drafts from itself")
    source_manifest_path = source_base / "manifest.json"
    target_manifest_path = target_base / "manifest.json"
    if not source_manifest_path.is_file() or not target_manifest_path.is_file():
        raise ProductionError("Draft seeding requires both source and target manifests")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("unit_hash") != target_manifest.get("unit_hash"):
        raise ProductionError("Draft seed unit boundaries do not match the target run")
    source_model = source_manifest.get("models", {}).get("draft_revision")
    target_model = target_manifest.get("models", {}).get("draft_revision")
    if source_model != target_model or target_model != ANTHROPIC_MODEL:
        raise ProductionError("Draft seed model does not match the target run")
    source_inputs = source_manifest.get("inputs", {})
    target_inputs = target_manifest.get("inputs", {})
    mismatched = sorted(
        key
        for key in SEED_COMPATIBLE_INPUTS
        if source_inputs.get(key) != target_inputs.get(key)
    )
    if mismatched:
        raise ProductionError(
            "Draft seed semantic inputs do not match: " + ", ".join(mismatched)
        )

    imported: list[dict[str, str]] = []
    for unit in units:
        source_path = artifact_path(source_base, unit, "draft")
        if not source_path.is_file():
            continue
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        result = validate_reader(source_payload.get("result"), unit.expected_ayahs)
        if result is None:
            raise ProductionError(f"Draft seed artifact fails contract: {source_path}")
        target_path = artifact_path(target_base, unit, "draft")
        input_hash = input_hash_for_unit(unit)
        source_hash = file_hash(source_path)
        if target_path.exists():
            load_artifact(
                target_path,
                input_hash,
                lambda data, unit=unit: validate_reader(data, unit.expected_ayahs),
            )
        else:
            atomic_json(
                target_path,
                {
                    "input_hash": input_hash,
                    "model": ANTHROPIC_MODEL,
                    "attempt": source_payload.get("attempt"),
                    "usage": source_payload.get("usage", {}),
                    "result": result,
                    "raw": source_payload.get("raw", ""),
                    "imported_from": {
                        "run_id": source_base.name,
                        "artifact_sha256": source_hash,
                        "reason": "validated draft reuse after transport-contract hardening",
                    },
                },
            )
        imported.append({"unit_id": unit.unit_id, "artifact_sha256": source_hash})

    report = {
        "version": "production-draft-seed-v1",
        "source_run_id": source_base.name,
        "source_manifest_sha256": file_hash(source_manifest_path),
        "target_manifest_sha256": file_hash(target_manifest_path),
        "imported_count": len(imported),
        "imported": imported,
    }
    report_path = target_base / "DRAFT_SEED.json"
    if report_path.exists():
        current = json.loads(report_path.read_text(encoding="utf-8"))
        if current != report:
            raise ProductionError("Draft seed report changed during resume")
    else:
        atomic_json(report_path, report)
    return report


def shared_inputs(
    unit: ProductionUnit,
    *,
    verses: dict[tuple[int, int], str],
    packet: str,
    opening_bismillah: str | None = None,
) -> str:
    context = "\n".join(
        f"({ayah}) {verses[(unit.surah, ayah)]}"
        for ayah in range(unit.context_first, unit.context_last + 1)
    )
    target = "\n".join(
        f"({ayah}) {verses[(unit.surah, ayah)]}" for ayah in unit.expected_ayahs
    )
    opening = (
        "=== UNNUMBERED SURAH-OPENING MARKER (CONTEXT ONLY; DO NOT RETURN OR "
        f"TRANSLATE AS A TARGET AYAH) ===\n{opening_bismillah}\n\n"
        if opening_bismillah and unit.first_ayah == 1
        else ""
    )
    return (
        f"=== EVIDENCE PACKET FOR THE TARGET ===\n{packet}\n\n"
        f"{opening}"
        f"=== LOCAL CONTEXT: SURAH {unit.surah}, "
        f"AYAHS {unit.context_first}-{unit.context_last} ===\n{context}\n\n"
        f"=== TARGET AYAHS TO TRANSLATE: {unit.surah}:"
        f"{unit.first_ayah}-{unit.last_ayah} ===\n{target}"
    )


def cached_system_blocks(primary: str, prompt: str, ledger_md: str, ledger_json: str) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": primary},
        {"type": "text", "text": prompt},
        {
            "type": "text",
            "text": (
                "=== MODEL-FACING SENSE LEDGER ===\n"
                f"{ledger_md}\n\n=== STRUCTURED SENSE RECORDS ===\n{ledger_json}"
            ),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    ]


def prepare_production(
    conn: sqlite3.Connection,
    config: ProductionConfig,
) -> tuple[Path, list[ProductionUnit], dict[tuple[int, int], str]]:
    if config.gemini_transport not in GEMINI_TRANSPORTS:
        raise ProductionError(
            f"Unsupported Gemini transport: {config.gemini_transport}"
        )
    init_db(conn)
    source_issues = [issue for issue in validate_source(conn) if issue.severity == "error"]
    if source_issues:
        raise ProductionError(
            "Canonical source validation failed: "
            + "; ".join(issue.message for issue in source_issues)
        )
    required = [
        DEFAULT_SOURCE_XML,
        MORPHOLOGY_PATH,
        PROMPT_PATH,
        LEDGER_MD_PATH,
        LEDGER_JSON_PATH,
        REFRAIN_POLICY_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProductionError(f"Missing production inputs: {missing}")
    source_hash = file_hash(DEFAULT_SOURCE_XML)
    morphology_hash = file_hash(MORPHOLOGY_PATH)
    if source_hash != CANONICAL_SOURCE_SHA256:
        raise ProductionError(
            f"Canonical Tanzil XML hash changed: {source_hash}"
        )
    if morphology_hash != CANONICAL_MORPHOLOGY_SHA256:
        raise ProductionError(
            f"QAC morphology hash changed: {morphology_hash}"
        )
    imported_source = conn.execute(
        "SELECT sha256 FROM source_files WHERE id = 1"
    ).fetchone()
    if not imported_source or imported_source["sha256"] != source_hash:
        raise ProductionError(
            "SQLite source does not match the pinned Tanzil XML; re-import it before "
            "preparing production."
        )

    base = run_dir(config.run_id)
    base.mkdir(parents=True, exist_ok=True)
    verses = source_verses(conn)
    try:
        ledger = json.loads(LEDGER_JSON_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductionError("Structured sense ledger is invalid JSON") from exc
    evidence_sources = ledger.get("evidence_sources")
    if not isinstance(evidence_sources, dict):
        raise ProductionError("Structured sense ledger lacks evidence_sources")
    referenced_evidence = {
        evidence_id
        for entry in ledger.get("entries", [])
        if isinstance(entry, dict)
        for sense in entry.get("senses", [])
        if isinstance(sense, dict)
        for evidence_id in sense.get("evidence_ids", [])
    }
    dangling_evidence = sorted(referenced_evidence - set(evidence_sources))
    if dangling_evidence:
        raise ProductionError(
            f"Sense ledger has unresolved evidence IDs: {dangling_evidence}"
        )
    refrain_policy = json.loads(REFRAIN_POLICY_PATH.read_text(encoding="utf-8"))
    canonical_refrains = refrain_policy.get("canonical", refrain_policy)
    repeated_groups = repeated_ayah_groups(verses)
    if not isinstance(canonical_refrains, dict) or not set(canonical_refrains).issubset(
        repeated_groups
    ):
        raise ProductionError("Refrain policy references an unknown Arabic group hash")
    units = build_units(
        conn,
        max_ayahs=config.max_ayahs,
        max_arabic_chars=config.max_arabic_chars,
        context_ayahs=config.context_ayahs,
    )
    if sum(len(unit.expected_ayahs) for unit in units) != 6236:
        raise ProductionError("Production units do not cover exactly 6236 ayahs")
    if len({(unit.surah, ayah) for unit in units for ayah in unit.expected_ayahs}) != 6236:
        raise ProductionError("Production units overlap or omit ayahs")

    manifest = {
        "version": "quran-production-v2.4.1",
        "config": asdict(config),
        "models": {"draft_revision": ANTHROPIC_MODEL, "critic": GEMINI_MODEL},
        "limits": {
            "anthropic_max_tokens": ANTHROPIC_MAX_TOKENS,
            "anthropic_effort": ANTHROPIC_EFFORT,
            "anthropic_structured_outputs": True,
            "gemini_max_tokens": GEMINI_MAX_TOKENS,
            "contract_attempts": CONTRACT_ATTEMPTS,
        },
        "inputs": {
            "source_xml": source_hash,
            "morphology": morphology_hash,
            "prompt": file_hash(PROMPT_PATH),
            "ledger_md": file_hash(LEDGER_MD_PATH),
            "ledger_json": file_hash(LEDGER_JSON_PATH),
            "refrain_policy": file_hash(REFRAIN_POLICY_PATH),
            "production_runner": file_hash(Path(__file__)),
            "production_clients": file_hash(Path(__file__).with_name("production_clients.py")),
            "production_packets": file_hash(Path(__file__).with_name("production_packets.py")),
            "production_refrains": file_hash(Path(__file__).with_name("production_refrains.py")),
            "production_quality": file_hash(Path(__file__).with_name("production_quality.py")),
            "critic": file_hash(Path(__file__).with_name("critic_v2.py")),
            "spoken_english": file_hash(Path(__file__).with_name("spoken_english_v1.py")),
        },
        "unit_hash": stable_hash([unit.to_dict() for unit in units]),
    }
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != manifest:
            raise ProductionError(
                "Production manifest changed. Use a new run_id; never mix prompt, code, "
                "source, or batching versions inside one run."
            )
    else:
        atomic_json(manifest_path, manifest)

    write_packets(
        units,
        output_dir=base / "evidence",
        morphology_path=MORPHOLOGY_PATH,
        source_path=DEFAULT_SOURCE_XML,
        verses=verses,
    )
    units_payload = [unit.to_dict() for unit in units]
    units_path = base / "units.json"
    if units_path.exists() and json.loads(
        units_path.read_text(encoding="utf-8")
    ) != units_payload:
        raise ProductionError(f"Frozen production units changed: {units_path}")
    atomic_json(units_path, units_payload)

    _register_run(conn, config, units, manifest)
    return base, units, verses


def _register_run(
    conn: sqlite3.Connection,
    config: ProductionConfig,
    units: list[ProductionUnit],
    manifest: dict[str, Any],
) -> None:
    row = conn.execute(
        "SELECT * FROM translation_runs WHERE run_id = ?", (config.run_id,)
    ).fetchone()
    now = utc_now()
    prompt_hash = stable_hash(manifest["inputs"])
    if row:
        expected = {
            "model": ANTHROPIC_MODEL,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "batch_size": config.max_ayahs,
            "max_target_chars": config.max_arabic_chars,
            "context_before": config.context_ayahs,
            "context_after": config.context_ayahs,
        }
        mismatched = [key for key, value in expected.items() if row[key] != value]
        if mismatched:
            raise ProductionError(f"Existing run metadata mismatch: {mismatched}")
        return

    with conn:
        conn.execute(
            """
            INSERT INTO translation_runs (
                run_id, model, prompt_version, prompt_hash, batch_size,
                max_target_chars, context_before, context_after, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)
            """,
            (
                config.run_id,
                ANTHROPIC_MODEL,
                PROMPT_VERSION,
                prompt_hash,
                config.max_ayahs,
                config.max_arabic_chars,
                config.context_ayahs,
                config.context_ayahs,
                now,
                now,
            ),
        )
        for unit in units:
            refs = [f"{unit.surah}:{ayah}" for ayah in unit.expected_ayahs]
            conn.execute(
                """
                INSERT INTO translation_batches (
                    batch_id, run_id, batch_index, surah_number, start_ref,
                    end_ref, target_refs_json, status, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
                """,
                (
                    f"{config.run_id}_{unit.unit_id}",
                    config.run_id,
                    unit.unit_index,
                    unit.surah,
                    refs[0],
                    refs[-1],
                    json.dumps(refs),
                    now,
                    now,
                ),
            )


def _unit_packet(base: Path, unit: ProductionUnit) -> str:
    return (base / "evidence" / f"{unit.unit_id}.md").read_text(encoding="utf-8")


def draft_assignment(shared: str) -> str:
    return (
        f"{shared}\n\nTranslate only the target ayahs. Return exactly one reader object "
        "per target ayah under the JSON contract in the system instructions."
    )


def critic_assignment(shared: str, reader: list[dict[str, Any]]) -> str:
    return (
        f"{shared}\n\n=== ENGLISH TO AUDIT ===\n"
        f"{json.dumps(reader, ensure_ascii=False, indent=2)}"
    )


def revision_assignment(
    shared: str,
    reader: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> str:
    return (
        f"{shared}\n\n=== READER DRAFT ===\n"
        f"{json.dumps(reader, ensure_ascii=False, indent=2)}\n\n"
        "=== FINDINGS TO ADJUDICATE ===\n"
        f"{json.dumps(findings, ensure_ascii=False, indent=2)}"
    )


def _anthropic_params(
    *,
    system: list[dict[str, Any]],
    user: str,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": ANTHROPIC_MODEL,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": ANTHROPIC_EFFORT,
            "format": {"type": "json_schema", "schema": response_schema},
        },
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }


def anthropic_schema_for_stage(stage: str) -> dict[str, Any]:
    if stage == "draft":
        return READER_JSON_SCHEMA
    if stage in {"revision", "repair"}:
        return REVISION_JSON_SCHEMA
    raise ProductionError(f"No Anthropic response schema registered for stage: {stage}")


def _stage_input_hash(stage: str, unit: ProductionUnit, payload: Any) -> str:
    anthropic_stages = {"draft", "revision", "repair"}
    record = {
        "stage": stage,
        "unit": unit.to_dict(),
        "model": ANTHROPIC_MODEL if stage in anthropic_stages else GEMINI_MODEL,
        "payload": payload,
    }
    if stage in anthropic_stages:
        record["generation"] = {
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "effort": ANTHROPIC_EFFORT,
            "thinking": "adaptive",
            "response_schema": anthropic_schema_for_stage(stage),
        }
    return stable_hash(record)


def _shards(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _job_path(base: Path, stage: str, attempt: int, shard_index: int) -> Path:
    return base / "jobs" / f"{stage}-a{attempt}-s{shard_index:03d}.json"


def _wait_anthropic(
    client: AnthropicBatchClient,
    state: BatchState,
    *,
    poll_seconds: int,
    on_poll: Callable[[BatchState], None],
) -> BatchState:
    current = state
    while not current.ended:
        time.sleep(poll_seconds)
        current = client.retrieve(current.batch_id)
        on_poll(current)
    return current


def run_anthropic_stage(
    *,
    base: Path,
    stage: str,
    units: list[ProductionUnit],
    shard_size: int,
    poll_seconds: int,
    system: list[dict[str, Any]],
    assignment: Callable[[ProductionUnit, int], tuple[str, Any, Callable[[Any], Any]]],
    client: AnthropicBatchClient,
) -> None:
    with exclusive_lock(base / "jobs" / f".{stage}.lock"):
        _run_anthropic_stage(
            base=base, stage=stage, units=units, shard_size=shard_size,
            poll_seconds=poll_seconds, system=system, assignment=assignment, client=client,
        )


def _run_anthropic_stage(
    *,
    base: Path,
    stage: str,
    units: list[ProductionUnit],
    shard_size: int,
    poll_seconds: int,
    system: list[dict[str, Any]],
    assignment: Callable[[ProductionUnit, int], tuple[str, Any, Callable[[Any], Any]]],
    client: AnthropicBatchClient,
) -> None:
    response_schema = anthropic_schema_for_stage(stage)
    if shard_size <= 0:
        raise ProductionError("Shard size must be positive")
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        work_by_id = {}
        cached_ids = set()
        for unit in units:
            user, hash_payload, validator = assignment(unit, attempt)
            input_hash = _stage_input_hash(stage, unit, hash_payload)
            cached = load_artifact(artifact_path(base, unit, stage), input_hash, validator)
            work_by_id[unit.unit_id] = (unit, user, input_hash, validator)
            if cached is not None:
                cached_ids.add(unit.unit_id)
        plan_path = base / "jobs" / f"{stage}-a{attempt}-plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            groups = plan["groups"]
            if plan.get("shard_size") != shard_size:
                raise ProductionError("Frozen shard size changed")
        else:
            # Preserve legacy accepted shard membership exactly; append missing work.
            groups = []
            jobs = sorted((base / "jobs").glob(f"{stage}-a{attempt}-s*.json"))
            for index, job_path in enumerate(jobs, start=1):
                if job_path != _job_path(base, stage, attempt, index):
                    raise ProductionError("Non-contiguous legacy shards require reconciliation")
                job = json.loads(job_path.read_text(encoding="utf-8"))
                custom_ids = job.get("custom_ids", [])
                prefix = stage[:8] + "-"
                if not custom_ids or any(not value.startswith(prefix) for value in custom_ids):
                    raise ProductionError("Invalid existing shard membership")
                groups.append([value[len(prefix):] for value in custom_ids])
            assigned = {value for group in groups for value in group}
            pending = [
                unit.unit_id for unit in units
                if unit.unit_id not in assigned and unit.unit_id not in cached_ids
                and (attempt == 1 or (
                    unit_dir(base, unit) / f"{stage}-attempt{attempt - 1}-FAILED.json"
                ).exists())
            ]
            groups.extend(_shards(pending, shard_size))
            atomic_json(plan_path, {"shard_size": shard_size, "groups": groups})
        flat = [value for group in groups for value in group]
        if len(flat) != len(set(flat)) or any(value not in work_by_id for value in flat):
            raise ProductionError("Frozen attempt has invalid or duplicate units")
        if any(not group for group in groups):
            raise ProductionError("Frozen attempt has an empty shard")
        for shard_index, group in enumerate(groups, start=1):
            shard = [work_by_id[value] for value in group]
            if all(value in cached_ids for value in group):
                continue
            requests = [
                {
                    "custom_id": f"{stage[:8]}-{unit.unit_id}",
                    "params": _anthropic_params(
                        system=system,
                        user=user
                        + (
                            "\n\nRETRY: The prior response failed the registered JSON "
                            "contract. Return only the requested schema with exact ayah "
                            "coverage and permitted fields."
                            if attempt > 1
                            else ""
                        ),
                        response_schema=response_schema,
                    ),
                }
                for unit, user, _input_hash, _validator in shard
            ]
            request_hash = stable_hash(requests)
            job_path = _job_path(base, stage, attempt, shard_index)
            if job_path.exists():
                job = json.loads(job_path.read_text(encoding="utf-8"))
                if job.get("request_hash") != request_hash:
                    raise ProductionError(f"Provider job input mismatch: {job_path}")
                state = client.retrieve(str(job["batch_id"]))
            else:
                state = submit_batch_once(client, requests, job_path)
                atomic_json(
                    job_path,
                    {
                        "provider": "anthropic",
                        "stage": stage,
                        "attempt": attempt,
                        "shard": shard_index,
                        "request_hash": request_hash,
                        "batch_id": state.batch_id,
                        "state": state.state,
                        "submitted_at": utc_now(),
                        "custom_ids": [request["custom_id"] for request in requests],
                    },
                )

            def record_poll(current: BatchState) -> None:
                job = json.loads(job_path.read_text(encoding="utf-8"))
                job.update({"state": current.state, "last_polled_at": utc_now()})
                atomic_json(job_path, job)

            state = _wait_anthropic(
                client, state, poll_seconds=poll_seconds, on_poll=record_poll
            )
            record_poll(state)
            rows = {str(row.get("custom_id")): row for row in client.results(state.batch_id)}
            for unit, _user, input_hash, validator in shard:
                if artifact_path(base, unit, stage).exists():
                    load_artifact(artifact_path(base, unit, stage), input_hash, validator)
                    continue
                custom_id = f"{stage[:8]}-{unit.unit_id}"
                row = rows.get(custom_id)
                if row is None:
                    atomic_json(
                        unit_dir(base, unit) / f"{stage}-attempt{attempt}-FAILED.json",
                        {
                            "input_hash": input_hash,
                            "attempt": attempt,
                            "error": "Provider result lacked this custom_id",
                        },
                    )
                    continue
                try:
                    raw, usage = anthropic_result_text(row)
                    result = validator(extract_json(raw))
                except (ProviderError, json.JSONDecodeError, TypeError, ValueError):
                    result = None
                    raw = json.dumps(row, ensure_ascii=False)
                    usage = {}
                if result is None:
                    atomic_json(
                        unit_dir(base, unit) / f"{stage}-attempt{attempt}-FAILED.json",
                        {
                            "input_hash": input_hash,
                            "attempt": attempt,
                            "raw": raw,
                            "usage": usage,
                        },
                    )
                    continue
                atomic_json(
                    artifact_path(base, unit, stage),
                    {
                        "input_hash": input_hash,
                        "model": ANTHROPIC_MODEL,
                        "attempt": attempt,
                        "usage": usage,
                        "result": result,
                        "raw": raw,
                    },
                )
    pending = [unit for unit in units if not artifact_path(base, unit, stage).exists()]
    if pending:
        raise ProductionError(
            f"{stage} contract failed twice for: "
            + ", ".join(unit.unit_id for unit in pending)
        )


def _wait_gemini(
    client: GeminiBatchClient,
    state: BatchState,
    *,
    poll_seconds: int,
    on_poll: Callable[[BatchState], None],
) -> BatchState:
    current = state
    while not current.ended:
        time.sleep(poll_seconds)
        current = client.retrieve(current.batch_id)
        on_poll(current)
    return current


def _gemini_text(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    response = row.get("response")
    if not isinstance(response, dict):
        raise ProviderError(f"Gemini row failed: {row.get('error') or row}")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ProviderError("Gemini response lacks candidates")
    content = candidates[0].get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise ProviderError("Gemini response lacks content parts")
    text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
    if not text.strip():
        raise ProviderError("Gemini response contains no text")
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    return text, usage if isinstance(usage, dict) else {}


def run_gemini_sync_stage(
    *,
    base: Path,
    stage: str,
    units: list[ProductionUnit],
    system: str,
    assignment: Callable[[ProductionUnit, int], tuple[str, Any, Callable[[Any], Any]]],
    client: GeminiSynchronousClient,
    response_schema: dict[str, Any],
) -> None:
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        attempt_units = (
            list(units)
            if attempt == 1
            else [
                unit
                for unit in units
                if (
                    unit_dir(base, unit)
                    / f"{stage}-attempt{attempt - 1}-FAILED.json"
                ).exists()
                and not artifact_path(base, unit, stage).exists()
            ]
        )
        for unit in attempt_units:
            user, hash_payload, validator = assignment(unit, attempt)
            input_hash = _stage_input_hash(stage, unit, hash_payload)
            cached = load_artifact(
                artifact_path(base, unit, stage), input_hash, validator
            )
            if cached is not None:
                continue
            if attempt > 1:
                user += (
                    "\n\nRETRY: The prior response failed the registered JSON "
                    "contract. Return only the requested schema with exact ayah "
                    "coverage and permitted fields."
                )
            request_payload = {
                "model": GEMINI_MODEL,
                "system": system,
                "user": user,
                "response_schema": response_schema,
                "max_output_tokens": GEMINI_MAX_TOKENS,
                "temperature": 0,
            }
            request_hash = stable_hash(request_payload)
            job_path = (
                base / "jobs" / f"{stage}-sync-a{attempt}-{unit.unit_id}.json"
            )
            if job_path.exists():
                job = json.loads(job_path.read_text(encoding="utf-8"))
                if job.get("request_hash") != request_hash:
                    raise ProductionError(f"Provider job input mismatch: {job_path}")
                row = job.get("row")
                if not isinstance(row, dict):
                    raise ProductionError(f"Gemini sync job lacks response: {job_path}")
            else:
                try:
                    row = client.generate(
                        model=GEMINI_MODEL,
                        system=system,
                        user=user,
                        response_schema=response_schema,
                        max_output_tokens=GEMINI_MAX_TOKENS,
                    )
                except ProviderError as exc:
                    atomic_json(
                        unit_dir(base, unit) / f"{stage}-attempt{attempt}-FAILED.json",
                        {
                            "input_hash": input_hash,
                            "attempt": attempt,
                            "error": str(exc),
                        },
                    )
                    continue
                atomic_json(
                    job_path,
                    {
                        "provider": "google",
                        "transport": "sync",
                        "stage": stage,
                        "attempt": attempt,
                        "unit_id": unit.unit_id,
                        "request_hash": request_hash,
                        "state": "succeeded",
                        "completed_at": utc_now(),
                        "row": row,
                    },
                )
            try:
                raw, usage = _gemini_text(row)
                result = validator(extract_json(raw))
            except (ProviderError, json.JSONDecodeError, TypeError, ValueError):
                result = None
                raw = json.dumps(row, ensure_ascii=False)
                usage = {}
            if result is None:
                atomic_json(
                    unit_dir(base, unit) / f"{stage}-attempt{attempt}-FAILED.json",
                    {
                        "input_hash": input_hash,
                        "attempt": attempt,
                        "raw": raw,
                        "usage": usage,
                    },
                )
                continue
            atomic_json(
                artifact_path(base, unit, stage),
                {
                    "input_hash": input_hash,
                    "model": GEMINI_MODEL,
                    "transport": "sync",
                    "attempt": attempt,
                    "usage": usage,
                    "result": result,
                    "raw": raw,
                },
            )
            time.sleep(GEMINI_SYNC_DELAY_SECONDS)
    pending = [unit for unit in units if not artifact_path(base, unit, stage).exists()]
    if pending:
        raise ProductionError(
            f"{stage} contract failed twice for: "
            + ", ".join(unit.unit_id for unit in pending)
        )


def run_gemini_stage(
    *,
    base: Path,
    stage: str,
    units: list[ProductionUnit],
    shard_size: int,
    poll_seconds: int,
    system: str,
    assignment: Callable[[ProductionUnit, int], tuple[str, Any, Callable[[Any], Any]]],
    client: GeminiBatchClient | GeminiSynchronousClient,
    transport: str = "batch",
    response_schema: dict[str, Any] = CRITIC_JSON_SCHEMA,
) -> None:
    if transport == "sync":
        run_gemini_sync_stage(
            base=base,
            stage=stage,
            units=units,
            system=system,
            assignment=assignment,
            client=client,  # type: ignore[arg-type]
            response_schema=response_schema,
        )
        return
    if transport != "batch":
        raise ProductionError(f"Unsupported Gemini transport: {transport}")
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        attempt_units = (
            list(units)
            if attempt == 1
            else [
                unit
                for unit in units
                if (
                    unit_dir(base, unit)
                    / f"{stage}-attempt{attempt - 1}-FAILED.json"
                ).exists()
                and not artifact_path(base, unit, stage).exists()
            ]
        )
        if not attempt_units:
            break
        work: list[tuple[ProductionUnit, str, str, Callable[[Any], Any]]] = []
        for unit in attempt_units:
            user, hash_payload, validator = assignment(unit, attempt)
            input_hash = _stage_input_hash(stage, unit, hash_payload)
            load_artifact(artifact_path(base, unit, stage), input_hash, validator)
            work.append((unit, user, input_hash, validator))
        if not work:
            break

        for shard_index, shard in enumerate(_shards(work, shard_size), start=1):
            lines: list[str] = []
            for unit, user, _input_hash, _validator in shard:
                if attempt > 1:
                    user += (
                        "\n\nRETRY: The prior response failed the registered JSON "
                        "contract. Return only the requested schema with exact ayah "
                        "coverage and permitted fields."
                    )
                request = {
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generation_config": {
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                        "max_output_tokens": GEMINI_MAX_TOKENS,
                        "temperature": 0,
                    },
                }
                lines.append(
                    json.dumps(
                        {"key": unit.unit_id, "request": request},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            request_text = "\n".join(lines) + "\n"
            request_hash = sha256_bytes(request_text.encode("utf-8"))
            input_path = base / "jobs" / f"{stage}-a{attempt}-s{shard_index:03d}.jsonl"
            if input_path.exists() and input_path.read_text(encoding="utf-8") != request_text:
                raise ProductionError(f"Gemini batch input changed: {input_path}")
            atomic_text(input_path, request_text)
            job_path = _job_path(base, stage, attempt, shard_index)
            if job_path.exists():
                job = json.loads(job_path.read_text(encoding="utf-8"))
                if job.get("request_hash") != request_hash:
                    raise ProductionError(f"Provider job input mismatch: {job_path}")
                state = client.retrieve(str(job["batch_id"]))
            else:
                state = client.submit_file(
                    model=GEMINI_MODEL,
                    input_path=input_path,
                    display_name=f"{base.name}-{stage}-a{attempt}-s{shard_index:03d}",
                )
                atomic_json(
                    job_path,
                    {
                        "provider": "google",
                        "stage": stage,
                        "attempt": attempt,
                        "shard": shard_index,
                        "request_hash": request_hash,
                        "batch_id": state.batch_id,
                        "state": state.state,
                        "submitted_at": utc_now(),
                        "keys": [unit.unit_id for unit, *_rest in shard],
                    },
                )

            def record_poll(current: BatchState) -> None:
                job = json.loads(job_path.read_text(encoding="utf-8"))
                job.update({"state": current.state, "last_polled_at": utc_now()})
                atomic_json(job_path, job)

            state = _wait_gemini(
                client, state, poll_seconds=poll_seconds, on_poll=record_poll
            )
            record_poll(state)
            if not state.succeeded:
                for unit, _user, input_hash, _validator in shard:
                    if artifact_path(base, unit, stage).exists():
                        continue
                    atomic_json(
                        unit_dir(base, unit) / f"{stage}-attempt{attempt}-FAILED.json",
                        {
                            "input_hash": input_hash,
                            "attempt": attempt,
                            "error": f"Gemini batch ended in {state.state}",
                        },
                    )
                continue
            try:
                result_rows = client.file_results(state.batch_id)
            except ProviderError as exc:
                for unit, _user, input_hash, _validator in shard:
                    if artifact_path(base, unit, stage).exists():
                        continue
                    atomic_json(
                        unit_dir(base, unit) / f"{stage}-attempt{attempt}-FAILED.json",
                        {
                            "input_hash": input_hash,
                            "attempt": attempt,
                            "error": str(exc),
                        },
                    )
                continue
            rows = {str(row.get("key")): row for row in result_rows}
            for unit, _user, input_hash, validator in shard:
                if artifact_path(base, unit, stage).exists():
                    load_artifact(artifact_path(base, unit, stage), input_hash, validator)
                    continue
                row = rows.get(unit.unit_id)
                if row is None:
                    atomic_json(
                        unit_dir(base, unit) / f"{stage}-attempt{attempt}-FAILED.json",
                        {
                            "input_hash": input_hash,
                            "attempt": attempt,
                            "error": "Provider result lacked this key",
                        },
                    )
                    continue
                try:
                    raw, usage = _gemini_text(row)
                    result = validator(extract_json(raw))
                except (ProviderError, json.JSONDecodeError, TypeError, ValueError):
                    result = None
                    raw = json.dumps(row, ensure_ascii=False)
                    usage = {}
                if result is None:
                    atomic_json(
                        unit_dir(base, unit) / f"{stage}-attempt{attempt}-FAILED.json",
                        {
                            "input_hash": input_hash,
                            "attempt": attempt,
                            "raw": raw,
                            "usage": usage,
                        },
                    )
                    continue
                atomic_json(
                    artifact_path(base, unit, stage),
                    {
                        "input_hash": input_hash,
                        "model": GEMINI_MODEL,
                        "attempt": attempt,
                        "usage": usage,
                        "result": result,
                        "raw": raw,
                    },
                )
    pending = [unit for unit in units if not artifact_path(base, unit, stage).exists()]
    if pending:
        raise ProductionError(
            f"{stage} contract failed twice for: "
            + ", ".join(unit.unit_id for unit in pending)
        )


def _artifact_result(base: Path, unit: ProductionUnit, stage: str) -> Any:
    path = artifact_path(base, unit, stage)
    if not path.exists():
        raise ProductionError(f"Missing required stage artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["result"]


def _draft_reader(base: Path, unit: ProductionUnit) -> list[dict[str, Any]]:
    return _artifact_result(base, unit, "draft")


def _critic_findings(base: Path, unit: ProductionUnit, stage: str) -> list[dict[str, Any]]:
    return finding_records(_artifact_result(base, unit, stage))


def _reader_after_first_revision(base: Path, unit: ProductionUnit) -> list[dict[str, Any]]:
    findings = _critic_findings(base, unit, "critic")
    if not findings:
        return _draft_reader(base, unit)
    return _artifact_result(base, unit, "revision")["ayahs"]


def _final_reader(base: Path, unit: ProductionUnit) -> list[dict[str, Any]]:
    verification = artifact_path(base, unit, "verification")
    if verification.exists() and _critic_findings(base, unit, "verification"):
        return _artifact_result(base, unit, "repair")["ayahs"]
    return _reader_after_first_revision(base, unit)


def _reader_with_overrides(
    base: Path,
    unit: ProductionUnit,
    overrides: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "english": overrides.get(
                f"{unit.surah}:{row['ayah']}", str(row["english"])
            ),
        }
        for row in _final_reader(base, unit)
    ]


def persist_final_units(
    conn: sqlite3.Connection,
    *,
    base: Path,
    units: list[ProductionUnit],
    run_id: str,
    overrides: dict[str, str] | None = None,
) -> None:
    overrides = overrides or {}
    now = utc_now()
    with conn:
        for unit in units:
            reader = _final_reader(base, unit)
            for row in reader:
                ref = f"{unit.surah}:{row['ayah']}"
                final_english = overrides.get(ref, row["english"])
                raw = {
                    "source": "production-v2.4",
                    "unit_id": unit.unit_id,
                    "translation": final_english,
                    "review_flags": row.get("review_flags", []),
                    "refrain_override": ref in overrides,
                }
                conn.execute(
                    """
                    INSERT INTO translations (
                        run_id, verse_key, translation, status,
                        raw_translation_json, created_at, updated_at
                    ) VALUES (?, ?, ?, 'complete', ?, ?, ?)
                    ON CONFLICT(run_id, verse_key) DO UPDATE SET
                        translation = excluded.translation,
                        status = excluded.status,
                        raw_translation_json = excluded.raw_translation_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        run_id,
                        ref,
                        final_english,
                        json.dumps(raw, ensure_ascii=False, sort_keys=True),
                        now,
                        now,
                    ),
                )
            conn.execute(
                """
                UPDATE translation_batches
                SET status = 'complete', updated_at = ?
                WHERE run_id = ? AND batch_index = ?
                """,
                (now, run_id, unit.unit_index),
            )
        conn.execute(
            "UPDATE translation_runs SET status = 'qa_pending', updated_at = ? WHERE run_id = ?",
            (now, run_id),
        )


def production_status(base: Path, units: list[ProductionUnit]) -> dict[str, Any]:
    stages = [
        "draft",
        "critic",
        "revision",
        "verification",
        "repair",
        "final_verification",
        "refrain_verification",
        "spoken",
    ]
    counts = {
        stage: sum(artifact_path(base, unit, stage).exists() for unit in units)
        for stage in stages
    }
    final_ayahs = 0
    for unit in units:
        try:
            final_ayahs += len(_final_reader(base, unit))
        except (ProductionError, KeyError, TypeError):
            continue
    return {
        "run_id": base.name,
        "units": len(units),
        "ayahs": 6236,
        "stage_counts": counts,
        "final_ayahs_ready": final_ayahs,
    }


def dry_run_report(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    opening_bismillah: dict[int, str],
    shard_size: int,
) -> dict[str, Any]:
    prompt, ledger_md, ledger_json = prompt_material()
    shared_chars = len(prompt) + len(ledger_md) + len(ledger_json)
    user_chars: list[int] = []
    output_word_budget = 0
    for unit in units:
        shared = shared_inputs(
            unit,
            verses=verses,
            packet=_unit_packet(base, unit),
            opening_bismillah=opening_bismillah.get(unit.surah),
        )
        user_chars.append(len(draft_assignment(shared)))
        output_word_budget += sum(
            len(verses[(unit.surah, ayah)].split()) for ayah in unit.expected_ayahs
        )
    report = {
        "version": "production-dry-run-v1",
        "run_id": base.name,
        "unit_count": len(units),
        "ayah_count": sum(len(unit.expected_ayahs) for unit in units),
        "shared_cached_chars_per_opus_request": shared_chars,
        "unit_prompt_chars": {
            "min": min(user_chars),
            "max": max(user_chars),
            "mean": round(sum(user_chars) / len(user_chars), 1),
            "total": sum(user_chars),
        },
        "arabic_word_count": output_word_budget,
        "batch_shards_per_full_phase": len(_shards(units, shard_size)),
        "api_requests_submitted": 0,
    }
    atomic_json(base / "DRY_RUN.json", report)
    return report


def run_production(
    conn: sqlite3.Connection,
    config: ProductionConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    base, units, verses = prepare_production(conn, config)
    opening_bismillah = {
        int(row["surah_number"]): str(row["bismillah"])
        for row in conn.execute(
            """
            SELECT surah_number, bismillah
            FROM source_ayahs
            WHERE ayah_number = 1 AND bismillah IS NOT NULL
            """
        )
    }
    if dry_run:
        return dry_run_report(
            base, units, verses, opening_bismillah, config.shard_size
        )

    with conn:
        conn.execute(
            "UPDATE translation_runs SET status = 'in_progress', updated_at = ? "
            "WHERE run_id = ?",
            (utc_now(), config.run_id),
        )

    load_environment()
    anthropic = AnthropicBatchClient(os.environ.get("ANTHROPIC_API_KEY", ""))
    gemini: GeminiBatchClient | GeminiSynchronousClient
    if config.gemini_transport == "sync":
        gemini = GeminiSynchronousClient(os.environ.get("GOOGLE_API_KEY", ""))
    else:
        gemini = GeminiBatchClient(os.environ.get("GOOGLE_API_KEY", ""))
    prompt, ledger_md, ledger_json = prompt_material()
    draft_system = cached_system_blocks(DRAFT_SYSTEM, prompt, ledger_md, ledger_json)
    revision_system = cached_system_blocks(
        REVISION_SYSTEM, prompt, ledger_md, ledger_json
    )
    critic_system = (
        f"{CRITIC_SYSTEM}\n\n=== PROJECT TRANSLATION POLICY ===\n{prompt}\n\n"
        f"=== MODEL-FACING SENSE LEDGER ===\n{ledger_md}\n\n"
        f"=== STRUCTURED SENSE RECORDS ===\n{ledger_json}"
    )

    shared_by_unit = {
        unit.unit_id: shared_inputs(
            unit,
            verses=verses,
            packet=_unit_packet(base, unit),
            opening_bismillah=opening_bismillah.get(unit.surah),
        )
        for unit in units
    }

    if config.seed_drafts_from:
        source_base = run_dir(config.seed_drafts_from)
        seed_draft_artifacts(
            target_base=base,
            source_base=source_base,
            units=units,
            input_hash_for_unit=lambda unit: _stage_input_hash(
                "draft",
                unit,
                {
                    "system": draft_system,
                    "user": draft_assignment(shared_by_unit[unit.unit_id]),
                },
            ),
        )

    run_anthropic_stage(
        base=base,
        stage="draft",
        units=units,
        shard_size=config.shard_size,
        poll_seconds=config.poll_seconds,
        system=draft_system,
        client=anthropic,
        assignment=lambda unit, _attempt: (
            draft_assignment(shared_by_unit[unit.unit_id]),
            {
                "system": draft_system,
                "user": draft_assignment(shared_by_unit[unit.unit_id]),
            },
            lambda data, unit=unit: validate_reader(data, unit.expected_ayahs),
        ),
    )

    run_gemini_stage(
        base=base,
        stage="critic",
        units=units,
        shard_size=config.shard_size,
        poll_seconds=config.poll_seconds,
        system=critic_system,
        client=gemini,
        transport=config.gemini_transport,
        assignment=lambda unit, _attempt: (
            critic_assignment(
                shared_by_unit[unit.unit_id], _draft_reader(base, unit)
            ),
            {
                "system": critic_system,
                "reader": _draft_reader(base, unit),
                "shared": shared_by_unit[unit.unit_id],
            },
            lambda data, unit=unit: validate_critic_response(
                data,
                unit.expected_ayahs,
                _draft_reader(base, unit),
                unit_arabic(verses, unit),
            ),
        ),
    )

    revision_units = [
        unit for unit in units if _critic_findings(base, unit, "critic")
    ]
    run_anthropic_stage(
        base=base,
        stage="revision",
        units=revision_units,
        shard_size=config.shard_size,
        poll_seconds=config.poll_seconds,
        system=revision_system,
        client=anthropic,
        assignment=lambda unit, _attempt: (
            revision_assignment(
                shared_by_unit[unit.unit_id],
                _draft_reader(base, unit),
                _critic_findings(base, unit, "critic"),
            ),
            {
                "system": revision_system,
                "reader": _draft_reader(base, unit),
                "findings": _critic_findings(base, unit, "critic"),
            },
            lambda data, unit=unit: validate_revision(
                data,
                unit.expected_ayahs,
                [
                    finding["finding_id"]
                    for finding in _critic_findings(base, unit, "critic")
                ],
            ),
        ),
    )

    run_gemini_stage(
        base=base,
        stage="verification",
        units=revision_units,
        shard_size=config.shard_size,
        poll_seconds=config.poll_seconds,
        system=critic_system,
        client=gemini,
        transport=config.gemini_transport,
        assignment=lambda unit, _attempt: (
            critic_assignment(
                shared_by_unit[unit.unit_id],
                _reader_after_first_revision(base, unit),
            )
            + "\n\nThis is a post-revision verification pass. Look especially for "
            "defects introduced during revision. Do not assume the earlier critic "
            "or reviser was correct.",
            {
                "system": critic_system,
                "reader": _reader_after_first_revision(base, unit),
                "shared": shared_by_unit[unit.unit_id],
                "mode": "post_revision_verification",
            },
            lambda data, unit=unit: validate_critic_response(
                data,
                unit.expected_ayahs,
                _reader_after_first_revision(base, unit),
                unit_arabic(verses, unit),
            ),
        ),
    )

    repair_units = [
        unit
        for unit in revision_units
        if _critic_findings(base, unit, "verification")
    ]
    run_anthropic_stage(
        base=base,
        stage="repair",
        units=repair_units,
        shard_size=config.shard_size,
        poll_seconds=config.poll_seconds,
        system=revision_system,
        client=anthropic,
        assignment=lambda unit, _attempt: (
            revision_assignment(
                shared_by_unit[unit.unit_id],
                _reader_after_first_revision(base, unit),
                _critic_findings(base, unit, "verification"),
            ),
            {
                "system": revision_system,
                "reader": _reader_after_first_revision(base, unit),
                "findings": _critic_findings(base, unit, "verification"),
                "mode": "bounded_second_revision",
            },
            lambda data, unit=unit: validate_revision(
                data,
                unit.expected_ayahs,
                [
                    finding["finding_id"]
                    for finding in _critic_findings(base, unit, "verification")
                ],
            ),
        ),
    )

    run_gemini_stage(
        base=base,
        stage="final_verification",
        units=repair_units,
        shard_size=config.shard_size,
        poll_seconds=config.poll_seconds,
        system=critic_system,
        client=gemini,
        transport=config.gemini_transport,
        assignment=lambda unit, _attempt: (
            critic_assignment(shared_by_unit[unit.unit_id], _final_reader(base, unit))
            + "\n\nThis is the final fidelity check after the bounded repair. "
            "Report every remaining defect; no further automatic rewrite follows.",
            {
                "system": critic_system,
                "reader": _final_reader(base, unit),
                "shared": shared_by_unit[unit.unit_id],
                "mode": "final_post_repair_verification",
            },
            lambda data, unit=unit: validate_critic_response(
                data,
                unit.expected_ayahs,
                _final_reader(base, unit),
                unit_arabic(verses, unit),
            ),
        ),
    )

    model_translations = {
        (unit.surah, int(row["ayah"])): str(row["english"])
        for unit in units
        for row in _final_reader(base, unit)
    }
    refrain_shared_system = [
        {"type": "text", "text": prompt},
        {
            "type": "text",
            "text": (
                "=== MODEL-FACING SENSE LEDGER ===\n"
                f"{ledger_md}\n\n=== STRUCTURED SENSE RECORDS ===\n{ledger_json}"
            ),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    ]
    refrain_report = resolve_refrains(
        base=base,
        verses=verses,
        translations=model_translations,
        policy_path=REFRAIN_POLICY_PATH,
        shared_system=refrain_shared_system,
        client=anthropic,
        poll_seconds=config.poll_seconds,
    )

    overrides = refrain_report["overrides"]
    governed_refs = {
        f"{surah}:{ayah}"
        for group in refrain_report["groups"].values()
        if group.get("source") in {"policy", "opus_resolution"}
        for surah, ayah in group["refs"]
    }
    refrain_units = [
        unit
        for unit in units
        if any(f"{unit.surah}:{ayah}" in governed_refs for ayah in unit.expected_ayahs)
    ]
    run_gemini_stage(
        base=base,
        stage="refrain_verification",
        units=refrain_units,
        shard_size=config.shard_size,
        poll_seconds=config.poll_seconds,
        system=critic_system,
        client=gemini,
        transport=config.gemini_transport,
        assignment=lambda unit, _attempt: (
            critic_assignment(
                shared_by_unit[unit.unit_id],
                _reader_with_overrides(base, unit, overrides),
            )
            + "\n\nThis audit follows a mechanically enforced repeated-ayah "
            "rendering. Verify that the invariant English remains faithful in every "
            "local context and that no surrounding ayah was damaged.",
            {
                "system": critic_system,
                "reader": _reader_with_overrides(base, unit, overrides),
                "shared": shared_by_unit[unit.unit_id],
                "mode": "post_refrain_verification",
            },
            lambda data, unit=unit: validate_critic_response(
                data,
                unit.expected_ayahs,
                _reader_with_overrides(base, unit, overrides),
                unit_arabic(verses, unit),
            ),
        ),
    )

    spoken_system = (
        f"{SPOKEN_ENGLISH_SYSTEM}\n\n=== PROJECT REGISTER POLICY ===\n{prompt}"
    )
    run_gemini_stage(
        base=base,
        stage="spoken",
        units=units,
        shard_size=config.shard_size,
        poll_seconds=config.poll_seconds,
        system=spoken_system,
        client=gemini,
        transport=config.gemini_transport,
        response_schema=SPOKEN_ENGLISH_SCHEMA,
        assignment=lambda unit, _attempt: (
            f"{shared_by_unit[unit.unit_id]}\n\n=== FINAL ENGLISH TO CHECK ===\n"
            + json.dumps(
                _reader_with_overrides(base, unit, overrides),
                ensure_ascii=False,
                indent=2,
            ),
            {
                "system": spoken_system,
                "reader": _reader_with_overrides(base, unit, overrides),
                "shared": shared_by_unit[unit.unit_id],
                "mode": "advisory_spoken_english_review",
            },
            lambda data, unit=unit: validate_spoken_english(
                data,
                unit.expected_ayahs,
                reader_text(_reader_with_overrides(base, unit, overrides)),
            ),
        ),
    )

    persist_final_units(
        conn,
        base=base,
        units=units,
        run_id=config.run_id,
        overrides=overrides,
    )
    quality = run_production_quality_gate(
        conn,
        base=base,
        units=units,
        run_id=config.run_id,
        verses=verses,
        refrain_report=refrain_report,
    )
    now = utc_now()
    with conn:
        conn.execute(
            "UPDATE translation_runs SET status = ?, updated_at = ? WHERE run_id = ?",
            ("complete" if quality["passed"] else "qa_blocked", now, config.run_id),
        )
    if not quality["passed"]:
        (base / "PRODUCTION_COMPLETE.json").unlink(missing_ok=True)
        atomic_json(
            base / "QA_BLOCKED.json",
            {
                "run_id": config.run_id,
                "qa_report": "QA_REPORT.json",
                "error_count": quality["issue_counts"]["error"],
            },
        )
        raise ProductionError(
            f"Production persisted but QA is blocked by "
            f"{quality['issue_counts']['error']} error(s); see {base / 'QA_REPORT.md'}"
        )
    status = production_status(base, units)
    status["quality"] = {
        "passed": True,
        "spoken_findings": quality["review"]["spoken_findings"],
        "translator_flags": quality["review"]["translator_flags"],
    }
    (base / "QA_BLOCKED.json").unlink(missing_ok=True)
    atomic_json(base / "PRODUCTION_COMPLETE.json", status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--max-ayahs", type=int, default=DEFAULT_MAX_AYAHS)
    parser.add_argument("--max-arabic-chars", type=int, default=DEFAULT_MAX_ARABIC_CHARS)
    parser.add_argument("--context-ayahs", type=int, default=DEFAULT_CONTEXT_AYAHS)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument(
        "--seed-drafts-from",
        help="Reuse only validated draft artifacts from this compatible run ID",
    )
    parser.add_argument(
        "--gemini-transport",
        choices=sorted(GEMINI_TRANSPORTS),
        default="batch",
        help="Use Gemini file batches or checkpointed synchronous requests",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProductionConfig(
        run_id=args.run_id,
        max_ayahs=args.max_ayahs,
        max_arabic_chars=args.max_arabic_chars,
        context_ayahs=args.context_ayahs,
        shard_size=args.shard_size,
        poll_seconds=args.poll_seconds,
        seed_drafts_from=args.seed_drafts_from,
        gemini_transport=args.gemini_transport,
    )
    with connect(args.db) as conn:
        base, units, _verses = prepare_production(conn, config)
        if args.status:
            result = production_status(base, units)
        else:
            result = run_production(conn, config, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
