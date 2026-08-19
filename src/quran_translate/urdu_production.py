"""Resumable, budget-gated full-Quran Urdu production pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import DATA_DIR, DEFAULT_DB_PATH, DEFAULT_SOURCE_XML, PROJECT_ROOT
from .db import connect, init_db
from .production_clients import AnthropicBatchClient, ProviderError, anthropic_result_text
from .production_packets import (
    ProductionUnit,
    atomic_json,
    atomic_text,
    build_units,
    source_verses,
    write_packets,
)
from .refrains import repeated_ayah_groups
from .urdu_critic_benchmark import (
    APPROVAL_PATH,
    BENCHMARK_PATH,
    benchmark_system,
)
from .urdu_costs import (
    PRICING_PATH,
    estimate_request_ceiling,
    pricing,
    usage_cost,
)
from .urdu_quality import (
    CRITIC_SCHEMA,
    REVISION_SCHEMA,
    VERIFICATION_SCHEMA,
    deterministic_quality_gate,
    finding_records,
    validate_critic,
    validate_revision,
    validate_verification,
)
from .urdu_translation_bakeoff import (
    ALLOWED_FLAGS,
    CANDIDATES,
    CONTRACT_ATTEMPTS,
    ModelSpec,
    PROVIDER_CALLS,
    TRANSLATION_SCHEMA,
    extract_json,
    file_hash,
    load_environment,
    stable_hash,
    validate_translation,
)
from .validation import validate_source


PRODUCTION_ROOT = DATA_DIR / "work" / "urdu-production-v1"
POLICY_PATH = PROJECT_ROOT / "prompts" / "urdu-translation-v1.md"
DRAFT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "urdu-draft-production-v1.md"
SEMANTIC_LEDGER_MD_PATH = PROJECT_ROOT / "prompts" / "sense-ledger-v2.4.md"
SEMANTIC_LEDGER_JSON_PATH = DATA_DIR / "evidence" / "sense-ledger-v2.4.json"
URDU_LEDGER_PATH = PROJECT_ROOT / "prompts" / "urdu-production-ledger-v1.json"
CRITIC_PROMPT_PATH = PROJECT_ROOT / "prompts" / "urdu-critic-production-v1.md"
REVISER_PROMPT_PATH = PROJECT_ROOT / "prompts" / "urdu-reviser-production-v1.md"
VERIFIER_PROMPT_PATH = PROJECT_ROOT / "prompts" / "urdu-verifier-production-v1.md"
REFRAIN_PROMPT_PATH = PROJECT_ROOT / "prompts" / "urdu-refrain-resolver-v1.md"
MORPHOLOGY_PATH = DATA_DIR / "evidence" / "qac-morphology.txt"
DESIGN_PATH = PROJECT_ROOT / "URDU_PRODUCTION_V1.md"

CANONICAL_SOURCE_SHA256 = (
    "f78067cd98c51c03e450581e1e8713f4e7c352e0b62a4fe5c35811da28dd23bf"
)
CANONICAL_MORPHOLOGY_SHA256 = (
    "742bfac59941b2cb09736d5b7aae694af50792261fb8450cbf6afafcc340645f"
)
DRAFT_MODEL = next(
    model for model in CANDIDATES if model.candidate_id == "openrouter-muse-spark-12"
)
REVISION_MODEL = ModelSpec(
    "anthropic-opus-46-medium-reviser",
    "anthropic",
    "claude-opus-4-6",
    "medium",
    "Claude Opus 4.6 medium reviser",
)
MAX_OUTPUT_TOKENS = 24_000
REVISION_MAX_OUTPUT_TOKENS = 16_000
POLL_SECONDS = 30


class UrduProductionError(RuntimeError):
    """A production invariant or launch gate failed."""


class BudgetExceeded(UrduProductionError):
    """A provider request would cross the frozen hard cost ceiling."""


@dataclass(frozen=True)
class UrduProductionConfig:
    run_id: str
    hard_cost_ceiling_usd: float = 100.0
    max_ayahs: int = 24
    max_arabic_chars: int = 4_800
    context_ayahs: int = 3
    workers: int = 3


def run_dir(run_id: str) -> Path:
    return PRODUCTION_ROOT / run_id


def unit_dir(base: Path, unit: ProductionUnit) -> Path:
    return base / "units" / unit.unit_id


def artifact_path(base: Path, unit: ProductionUnit, stage: str) -> Path:
    return unit_dir(base, unit) / f"{stage}.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_inputs() -> list[Path]:
    return [
        DEFAULT_SOURCE_XML,
        MORPHOLOGY_PATH,
        POLICY_PATH,
        DRAFT_PROMPT_PATH,
        SEMANTIC_LEDGER_MD_PATH,
        SEMANTIC_LEDGER_JSON_PATH,
        URDU_LEDGER_PATH,
        CRITIC_PROMPT_PATH,
        REVISER_PROMPT_PATH,
        VERIFIER_PROMPT_PATH,
        REFRAIN_PROMPT_PATH,
        PRICING_PATH,
        DESIGN_PATH,
        APPROVAL_PATH,
    ]


def _critic_model() -> ModelSpec:
    if not APPROVAL_PATH.is_file():
        raise UrduProductionError(
            "No Urdu critic is approved. Run and approve the frozen critic benchmark first."
        )
    approval = _load_json(APPROVAL_PATH)
    if approval.get("version") != "urdu-critic-approval-v2":
        raise UrduProductionError("Unexpected Urdu critic approval version")
    if approval.get("benchmark_sha256") != file_hash(BENCHMARK_PATH):
        raise UrduProductionError("Urdu critic approval targets a different benchmark")
    if approval.get("system_sha256") != stable_hash(benchmark_system()):
        raise UrduProductionError("Urdu critic approval targets different instructions")
    result_path_value = approval.get("result_path")
    if not isinstance(result_path_value, str):
        raise UrduProductionError("Urdu critic approval lacks its result artifact")
    result_path = Path(result_path_value)
    if not result_path.is_file() or approval.get("result_sha256") != file_hash(result_path):
        raise UrduProductionError("Urdu critic approval result is missing or changed")
    result = _load_json(result_path)
    model = approval.get("model")
    if not isinstance(model, dict):
        raise UrduProductionError("Urdu critic approval lacks a model record")
    score = approval.get("score")
    if (
        not isinstance(score, dict)
        or not score.get("passed")
        or result.get("status") != "complete"
        or result.get("model") != model
        or result.get("score") != score
    ):
        raise UrduProductionError("Approved Urdu critic did not pass its benchmark")
    return ModelSpec(
        str(model["candidate_id"]),
        str(model["provider"]),
        str(model["model_id"]),
        str(model["reasoning"]),
        str(model["private_label"]),
    )


class BudgetLedger:
    def __init__(self, base: Path, ceiling: float) -> None:
        if ceiling <= 0 or ceiling > 100:
            raise UrduProductionError("Urdu production cost ceiling must be in (0, 100]")
        self.base = base
        self.ceiling = float(ceiling)
        self.lock = threading.Lock()
        self.reserved = 0.0

    def spent(self) -> float:
        total = 0.0
        for path in self.base.rglob("*.json"):
            try:
                value = _load_json(path).get("cost_usd")
            except (OSError, json.JSONDecodeError, AttributeError):
                continue
            if isinstance(value, (int, float)):
                total += float(value)
        return round(total, 8)

    def reserve(self, estimate: float) -> None:
        with self.lock:
            projected = self.spent() + self.reserved + estimate
            if projected > self.ceiling:
                raise BudgetExceeded(
                    f"Provider request would raise reserved spend to ${projected:.2f}, "
                    f"above the ${self.ceiling:.2f} ceiling"
                )
            self.reserved += estimate

    def release(self, estimate: float) -> None:
        with self.lock:
            self.reserved = max(0.0, self.reserved - estimate)

    def report(self) -> dict[str, float]:
        return {
            "ceiling_usd": round(self.ceiling, 2),
            "spent_usd": self.spent(),
            "reserved_usd": round(self.reserved, 8),
            "remaining_usd": round(self.ceiling - self.spent() - self.reserved, 8),
        }


def _shared_inputs(
    unit: ProductionUnit,
    *,
    verses: dict[tuple[int, int], str],
    packet: str,
    opening_bismillah: str | None,
) -> str:
    context = "\n".join(
        f"({ayah}) {verses[(unit.surah, ayah)]}"
        for ayah in range(unit.context_first, unit.context_last + 1)
    )
    target = "\n".join(
        f"({ayah}) {verses[(unit.surah, ayah)]}" for ayah in unit.expected_ayahs
    )
    opening = (
        "=== UNNUMBERED SURAH OPENING, CONTEXT ONLY ===\n"
        + opening_bismillah
        + "\nDo not return it as a numbered target ayah.\n\n"
        if opening_bismillah and unit.first_ayah == 1
        else ""
    )
    return (
        f"=== EVIDENCE PACKET ===\n{packet}\n\n"
        f"{opening}=== LOCAL ARABIC CONTEXT ===\n{context}\n\n"
        f"=== TARGET AYAHS {unit.surah}:{unit.first_ayah}-{unit.last_ayah} ===\n{target}"
    )


def _system_material(primary_path: Path) -> str:
    return (
        primary_path.read_text(encoding="utf-8").strip()
        + "\n\n=== TRANSLATION POLICY ===\n"
        + POLICY_PATH.read_text(encoding="utf-8").strip()
        + "\n\n=== SEMANTIC LEDGER ===\n"
        + SEMANTIC_LEDGER_MD_PATH.read_text(encoding="utf-8").strip()
        + "\n\n=== STRUCTURED SEMANTIC RECORDS ===\n"
        + SEMANTIC_LEDGER_JSON_PATH.read_text(encoding="utf-8").strip()
        + "\n\n=== URDU DECISION LEDGER ===\n"
        + URDU_LEDGER_PATH.read_text(encoding="utf-8").strip()
    )


def prepare_production(
    conn: sqlite3.Connection, config: UrduProductionConfig
) -> tuple[Path, list[ProductionUnit], dict[tuple[int, int], str], dict[int, str | None]]:
    if config.workers <= 0:
        raise UrduProductionError("workers must be positive")
    if config.hard_cost_ceiling_usd > 100:
        raise UrduProductionError("Hard cost ceiling may not exceed USD 100")
    missing = [str(path) for path in _required_inputs() if not path.is_file()]
    if missing:
        raise UrduProductionError(f"Missing Urdu production inputs: {missing}")
    if file_hash(DEFAULT_SOURCE_XML) != CANONICAL_SOURCE_SHA256:
        raise UrduProductionError("Pinned Tanzil source hash changed")
    if file_hash(MORPHOLOGY_PATH) != CANONICAL_MORPHOLOGY_SHA256:
        raise UrduProductionError("Pinned QAC morphology hash changed")
    init_db(conn)
    source_errors = [issue for issue in validate_source(conn) if issue.severity == "error"]
    if source_errors:
        raise UrduProductionError("Canonical Quran source database failed validation")
    imported = conn.execute("SELECT sha256 FROM source_files WHERE id = 1").fetchone()
    if not imported or imported["sha256"] != CANONICAL_SOURCE_SHA256:
        raise UrduProductionError("SQLite source does not match the pinned Tanzil XML")

    critic = _critic_model()
    pricing_record = pricing()
    if critic.model_id not in pricing_record["per_million_tokens"]:
        raise UrduProductionError(
            f"Pricing snapshot lacks approved critic {critic.model_id}"
        )
    units = build_units(
        conn,
        max_ayahs=config.max_ayahs,
        max_arabic_chars=config.max_arabic_chars,
        context_ayahs=config.context_ayahs,
    )
    if sum(len(unit.expected_ayahs) for unit in units) != 6236:
        raise UrduProductionError("Urdu units do not cover all 6,236 ayahs")
    verses = source_verses(conn)
    bismillah = {
        int(row["surah_number"]): row["bismillah"]
        for row in conn.execute(
            "SELECT surah_number, bismillah FROM source_ayahs WHERE ayah_number = 1"
        )
    }
    base = run_dir(config.run_id)
    base.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "quran-urdu-production-v1",
        "config": asdict(config),
        "models": {
            "draft": asdict(DRAFT_MODEL),
            "critic_verifier": asdict(critic),
            "selective_reviser": asdict(REVISION_MODEL),
        },
        "inputs": {
            str(path.relative_to(PROJECT_ROOT)): file_hash(path)
            for path in _required_inputs()
            if path.is_relative_to(PROJECT_ROOT)
        },
        "pricing": pricing_record,
        "unit_hash": stable_hash([unit.to_dict() for unit in units]),
        "runner_sha256": file_hash(Path(__file__)),
        "quality_sha256": file_hash(Path(__file__).with_name("urdu_quality.py")),
        "provider_sha256": file_hash(
            Path(__file__).with_name("urdu_translation_bakeoff.py")
        ),
    }
    manifest_path = base / "MANIFEST.json"
    if manifest_path.exists() and _load_json(manifest_path) != manifest:
        raise UrduProductionError(
            "Urdu production manifest changed; use a new run ID rather than mixing versions"
        )
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    units_payload = [unit.to_dict() for unit in units]
    units_path = base / "UNITS.json"
    if units_path.exists() and _load_json(units_path) != units_payload:
        raise UrduProductionError("Frozen Urdu unit boundaries changed")
    if not units_path.exists():
        atomic_json(units_path, units_payload)
    write_packets(
        units,
        output_dir=base / "evidence",
        morphology_path=MORPHOLOGY_PATH,
        source_path=DEFAULT_SOURCE_XML,
        verses=verses,
    )
    return base, units, verses, bismillah


def _load_stage(
    path: Path,
    *,
    input_hash: str,
    validator: Callable[[Any], Any],
) -> Any:
    if not path.exists():
        return None
    document = _load_json(path)
    if document.get("input_hash") != input_hash:
        raise UrduProductionError(f"Cached stage input changed: {path}")
    if document.get("status") != "complete":
        return None
    result = validator(document.get("result"))
    if result is None:
        raise UrduProductionError(f"Cached stage fails its contract: {path}")
    return result


def _draft_input(
    base: Path,
    unit: ProductionUnit,
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
) -> tuple[str, str]:
    system = _system_material(DRAFT_PROMPT_PATH)
    shared = _shared_inputs(
        unit,
        verses=verses,
        packet=(base / "evidence" / f"{unit.unit_id}.md").read_text(encoding="utf-8"),
        opening_bismillah=bismillah.get(unit.surah),
    )
    user = shared + "\n\nTranslate only the target ayahs and return the registered JSON object."
    return system, user


def _critic_input(
    base: Path,
    unit: ProductionUnit,
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
    draft: dict[str, Any],
) -> tuple[str, str]:
    system = _system_material(CRITIC_PROMPT_PATH)
    shared = _shared_inputs(
        unit,
        verses=verses,
        packet=(base / "evidence" / f"{unit.unit_id}.md").read_text(encoding="utf-8"),
        opening_bismillah=bismillah.get(unit.surah),
    )
    user = (
        shared
        + "\n\n=== MUSE SPARK URDU DRAFT ===\n"
        + json.dumps(draft, ensure_ascii=False, indent=2)
        + "\n\nAudit every target ayah and return the registered JSON object."
    )
    return system, user


def _sync_job(
    *,
    base: Path,
    unit: ProductionUnit,
    stage: str,
    model: ModelSpec,
    system: str,
    user: str,
    schema: dict[str, Any],
    validator: Callable[[Any], Any],
    budget: BudgetLedger,
) -> dict[str, str]:
    path = artifact_path(base, unit, stage)
    input_hash = stable_hash(
        {
            "stage": stage,
            "unit": unit.to_dict(),
            "model": asdict(model),
            "system": system,
            "user": user,
            "schema": schema,
            "contract_attempts": CONTRACT_ATTEMPTS,
        }
    )
    existing = _load_stage(path, input_hash=input_hash, validator=validator)
    if existing is not None:
        return {"unit_id": unit.unit_id, "status": "reused"}
    if path.exists() and int(_load_json(path).get("attempts", 0)) >= CONTRACT_ATTEMPTS:
        return {"unit_id": unit.unit_id, "status": "failed"}

    errors: list[str] = []
    consumed_attempts: set[int] = set()
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        failed_path = unit_dir(base, unit) / f"{stage}-attempt{attempt}-FAILED.json"
        if not failed_path.exists():
            continue
        failed = _load_json(failed_path)
        if failed.get("input_hash") != input_hash:
            raise UrduProductionError(f"Cached failed stage input changed: {failed_path}")
        consumed_attempts.add(attempt)
        errors.extend(str(item) for item in failed.get("errors", []))
    raw_text = ""
    raw_response: dict[str, Any] | None = None
    usage: dict[str, Any] = {}
    started = time.monotonic()
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        if attempt in consumed_attempts:
            continue
        raw_text = ""
        raw_response = None
        usage = {}
        assignment = user + (
            "\n\nPrevious response failed the strict contract: " + errors[-1]
            if errors
            else ""
        )
        reservation = estimate_request_ceiling(
            model.model_id, system, assignment, MAX_OUTPUT_TOKENS
        )
        budget.reserve(reservation)
        try:
            raw_text, usage, raw_response = PROVIDER_CALLS[model.provider](
                model, system, assignment, schema
            )
            result = validator(extract_json(raw_text))
            if result is None:
                raise UrduProductionError("response failed strict stage contract")
            cost = usage_cost(model.model_id, usage)
            atomic_json(
                path,
                {
                    "version": "urdu-production-stage-v1",
                    "stage": stage,
                    "unit_id": unit.unit_id,
                    "input_hash": input_hash,
                    "status": "complete",
                    "model": asdict(model),
                    "attempts": attempt,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "usage": usage,
                    "cost_usd": cost,
                    "result": result,
                    "errors_before_success": errors,
                    "raw_text": raw_text,
                    "raw_response": raw_response,
                },
            )
            return {"unit_id": unit.unit_id, "status": "complete"}
        except BudgetExceeded:
            raise
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}"[:3000])
            failed_payload: dict[str, Any] = {
                "version": "urdu-production-stage-failure-v1",
                "stage": stage,
                "unit_id": unit.unit_id,
                "input_hash": input_hash,
                "attempt": attempt,
                "errors": [errors[-1]],
                "usage": usage,
                "raw_text": raw_text,
                "raw_response": raw_response,
            }
            if usage:
                failed_payload["cost_usd"] = usage_cost(model.model_id, usage)
            atomic_json(
                unit_dir(base, unit) / f"{stage}-attempt{attempt}-FAILED.json",
                failed_payload,
            )
        finally:
            budget.release(reservation)
    atomic_json(
        path,
        {
            "version": "urdu-production-stage-v1",
            "stage": stage,
            "unit_id": unit.unit_id,
            "input_hash": input_hash,
            "status": "failed",
            "model": asdict(model),
            "attempts": CONTRACT_ATTEMPTS,
            "latency_seconds": round(time.monotonic() - started, 3),
            "usage": usage,
            "errors": errors,
            "raw_text": raw_text,
            "raw_response": raw_response,
        },
    )
    return {"unit_id": unit.unit_id, "status": "failed"}


def _run_parallel(
    jobs: list[tuple[ProductionUnit, Callable[[], dict[str, str]]]], workers: int
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call): unit for unit, call in jobs}
        for future in as_completed(futures):
            unit = futures[future]
            outcome = future.result()
            print(f"{outcome['status']}: {unit.unit_id}", flush=True)


def run_drafts(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
    *,
    budget: BudgetLedger,
    workers: int,
    limit: int | None = None,
) -> None:
    jobs: list[tuple[ProductionUnit, Callable[[], dict[str, str]]]] = []
    for unit in units:
        system, user = _draft_input(base, unit, verses, bismillah)
        validator = lambda value, unit=unit: validate_translation(
            value, unit.expected_ayahs
        )
        input_hash = stable_hash(
            {
                "stage": "draft",
                "unit": unit.to_dict(),
                "model": asdict(DRAFT_MODEL),
                "system": system,
                "user": user,
                "schema": TRANSLATION_SCHEMA,
                "contract_attempts": CONTRACT_ATTEMPTS,
            }
        )
        if _load_stage(
            artifact_path(base, unit, "draft"),
            input_hash=input_hash,
            validator=validator,
        ) is not None:
            continue
        jobs.append(
            (
                unit,
                lambda unit=unit, system=system, user=user, validator=validator: _sync_job(
                    base=base,
                    unit=unit,
                    stage="draft",
                    model=DRAFT_MODEL,
                    system=system,
                    user=user,
                    schema=TRANSLATION_SCHEMA,
                    validator=validator,
                    budget=budget,
                ),
            )
        )
    _run_parallel(jobs[:limit] if limit else jobs, workers)


def _artifact_result(base: Path, unit: ProductionUnit, stage: str) -> Any:
    path = artifact_path(base, unit, stage)
    if not path.is_file():
        raise UrduProductionError(f"Missing {stage} artifact: {path}")
    document = _load_json(path)
    if document.get("status") != "complete":
        raise UrduProductionError(f"{stage} is not complete: {path}")
    return document["result"]


def run_critics(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
    *,
    budget: BudgetLedger,
    workers: int,
    limit: int | None = None,
) -> None:
    model = _critic_model()
    jobs: list[tuple[ProductionUnit, Callable[[], dict[str, str]]]] = []
    for unit in units:
        draft = _artifact_result(base, unit, "draft")
        arabic = {ayah: verses[(unit.surah, ayah)] for ayah in unit.expected_ayahs}
        urdu = {int(row["ayah"]): str(row["urdu"]) for row in draft["ayahs"]}
        system, user = _critic_input(base, unit, verses, bismillah, draft)
        validator = lambda value, unit=unit, arabic=arabic, urdu=urdu: validate_critic(
            value,
            expected=unit.expected_ayahs,
            arabic_by_ayah=arabic,
            urdu_by_ayah=urdu,
        )
        jobs.append(
            (
                unit,
                lambda unit=unit, system=system, user=user, validator=validator: _sync_job(
                    base=base,
                    unit=unit,
                    stage="critic",
                    model=model,
                    system=system,
                    user=user,
                    schema=CRITIC_SCHEMA,
                    validator=validator,
                    budget=budget,
                ),
            )
        )
    _run_parallel(jobs[:limit] if limit else jobs, workers)


def _units_requiring_revision(base: Path, units: list[ProductionUnit]) -> list[ProductionUnit]:
    needed: list[ProductionUnit] = []
    for unit in units:
        critic = _artifact_result(base, unit, "critic")
        if any(
            finding["severity"] in {"blocking", "significant"}
            for row in critic["ayahs"]
            for finding in row["findings"]
        ):
            needed.append(unit)
    return needed


def _revision_assignment(
    base: Path,
    unit: ProductionUnit,
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
) -> tuple[str, str, list[str]]:
    system = _system_material(REVISER_PROMPT_PATH)
    draft = _artifact_result(base, unit, "draft")
    critic = _artifact_result(base, unit, "critic")
    records = finding_records(critic)
    shared = _shared_inputs(
        unit,
        verses=verses,
        packet=(base / "evidence" / f"{unit.unit_id}.md").read_text(encoding="utf-8"),
        opening_bismillah=bismillah.get(unit.surah),
    )
    user = (
        shared
        + "\n\n=== FROZEN MUSE BASE ===\n"
        + json.dumps(draft, ensure_ascii=False, indent=2)
        + "\n\n=== CRITIC FINDINGS ===\n"
        + json.dumps(records, ensure_ascii=False, indent=2)
        + "\n\nReturn every target ayah and one decision per finding."
    )
    return system, user, [str(record["finding_id"]) for record in records]


def _revision_request(system: str, user: str) -> dict[str, Any]:
    return {
        "model": REVISION_MODEL.model_id,
        "max_tokens": REVISION_MAX_OUTPUT_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": REVISION_MODEL.reasoning,
            "format": {"type": "json_schema", "schema": REVISION_SCHEMA},
        },
        "system": [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [{"role": "user", "content": user}],
    }


def run_revisions(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    bismillah: dict[int, str | None],
    *,
    budget: BudgetLedger,
    poll_seconds: int = POLL_SECONDS,
) -> None:
    needed = _units_requiring_revision(base, units)
    if not needed:
        return
    load_environment()
    client = AnthropicBatchClient(os.environ.get("ANTHROPIC_API_KEY", ""))
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        pending: list[tuple[ProductionUnit, str, str, list[str], str]] = []
        for unit in needed:
            system, user, finding_ids = _revision_assignment(
                base, unit, verses, bismillah
            )
            input_hash = stable_hash(
                {
                    "stage": "revision",
                    "unit": unit.to_dict(),
                    "model": asdict(REVISION_MODEL),
                    "system": system,
                    "user": user,
                    "schema": REVISION_SCHEMA,
                    "contract_attempts": CONTRACT_ATTEMPTS,
                }
            )
            validator = lambda value, unit=unit, finding_ids=finding_ids: validate_revision(
                value, expected=unit.expected_ayahs, finding_ids=finding_ids
            )
            if _load_stage(
                artifact_path(base, unit, "revision"),
                input_hash=input_hash,
                validator=validator,
            ) is None:
                pending.append((unit, system, user, finding_ids, input_hash))
        if not pending:
            return
        requests = [
            {
                "custom_id": unit.unit_id,
                "params": _revision_request(system, user),
            }
            for unit, system, user, _finding_ids, _input_hash in pending
        ]
        request_hash = stable_hash(requests)
        job_path = base / f"REVISION_BATCH_ATTEMPT_{attempt}.json"
        is_new_submission = not job_path.exists()
        reservation = (
            sum(
                estimate_request_ceiling(
                    REVISION_MODEL.model_id,
                    system,
                    user,
                    REVISION_MAX_OUTPUT_TOKENS,
                )
                for _unit, system, user, _finding_ids, _input_hash in pending
            )
            if is_new_submission
            else 0.0
        )
        if reservation:
            budget.reserve(reservation)
        try:
            if job_path.exists():
                job = _load_json(job_path)
                if job.get("request_hash") != request_hash:
                    raise UrduProductionError("Existing revision batch input changed")
                state = client.retrieve(str(job["batch_id"]))
            else:
                state = client.submit(requests)
                atomic_json(
                    job_path,
                    {
                        "version": "urdu-revision-batch-v1",
                        "batch_id": state.batch_id,
                        "state": state.state,
                        "attempt": attempt,
                        "request_hash": request_hash,
                        "unit_ids": [unit.unit_id for unit, *_rest in pending],
                    },
                )
            while not state.ended:
                job = _load_json(job_path)
                job["state"] = state.state
                atomic_json(job_path, job)
                print(f"revision batch {state.batch_id}: {state.state}", flush=True)
                time.sleep(poll_seconds)
                state = client.retrieve(state.batch_id)
            if not state.succeeded:
                raise ProviderError(f"Revision batch ended in {state.state}")
            rows = {str(row.get("custom_id")): row for row in client.results(state.batch_id)}
            for unit, _system, _user, finding_ids, input_hash in pending:
                row = rows.get(unit.unit_id)
                errors: list[str] = []
                usage: dict[str, Any] = {}
                if row is None:
                    errors.append("Anthropic batch omitted the unit")
                else:
                    try:
                        text, metadata = anthropic_result_text(row)
                        result = validate_revision(
                            extract_json(text),
                            expected=unit.expected_ayahs,
                            finding_ids=finding_ids,
                        )
                        if result is None:
                            raise UrduProductionError("revision failed strict contract")
                        usage = metadata.get("usage", {})
                        atomic_json(
                            artifact_path(base, unit, "revision"),
                            {
                                "version": "urdu-production-stage-v1",
                                "stage": "revision",
                                "unit_id": unit.unit_id,
                                "input_hash": input_hash,
                                "status": "complete",
                                "model": asdict(REVISION_MODEL),
                                "attempts": attempt,
                                "usage": usage,
                                "cost_usd": usage_cost(REVISION_MODEL.model_id, usage),
                                "result": result,
                                "raw": row,
                            },
                        )
                        continue
                    except Exception as exc:
                        errors.append(f"{type(exc).__name__}: {exc}"[:3000])
                atomic_json(
                    unit_dir(base, unit) / f"revision-attempt{attempt}-FAILED.json",
                    {
                        "version": "urdu-production-stage-failure-v1",
                        "stage": "revision",
                        "unit_id": unit.unit_id,
                        "input_hash": input_hash,
                        "attempt": attempt,
                        "errors": errors,
                        "usage": usage,
                        **(
                            {"cost_usd": usage_cost(REVISION_MODEL.model_id, usage)}
                            if usage
                            else {}
                        ),
                        "raw": row,
                    },
                )
        finally:
            if reservation:
                budget.release(reservation)
    failed = [
        unit.unit_id
        for unit in needed
        if not artifact_path(base, unit, "revision").is_file()
    ]
    if failed:
        raise UrduProductionError(
            "Selective Opus revision exhausted its contract for: " + ", ".join(failed)
        )


def _verification_input(
    base: Path,
    unit: ProductionUnit,
    verses: dict[tuple[int, int], str],
) -> tuple[str, str, list[int], dict[int, str], dict[int, str]]:
    system = _system_material(VERIFIER_PROMPT_PATH)
    draft = _artifact_result(base, unit, "draft")
    critic = _artifact_result(base, unit, "critic")
    revision = _artifact_result(base, unit, "revision")
    draft_by = {int(row["ayah"]): str(row["urdu"]) for row in draft["ayahs"]}
    revised_by = {int(row["ayah"]): str(row["urdu"]) for row in revision["ayahs"]}
    finding_ayahs = {int(record["ayah"]) for record in finding_records(critic)}
    changed = {ayah for ayah in unit.expected_ayahs if draft_by[ayah] != revised_by[ayah]}
    targets = sorted(finding_ayahs | changed)
    arabic = {ayah: verses[(unit.surah, ayah)] for ayah in targets}
    packet = (base / "evidence" / f"{unit.unit_id}.md").read_text(encoding="utf-8")
    user = (
        f"=== EVIDENCE PACKET ===\n{packet}\n\n"
        + "=== ARABIC TARGETS ===\n"
        + json.dumps(arabic, ensure_ascii=False, indent=2)
        + "\n\n=== FROZEN MUSE BASE ===\n"
        + json.dumps({ayah: draft_by[ayah] for ayah in targets}, ensure_ascii=False, indent=2)
        + "\n\n=== PROPOSED REVISION ===\n"
        + json.dumps({ayah: revised_by[ayah] for ayah in targets}, ensure_ascii=False, indent=2)
        + "\n\n=== CRITIC AND REVISION DECISIONS ===\n"
        + json.dumps(
            {"critic": finding_records(critic), "decisions": revision["decisions"]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nReturn one verification row per listed ayah in numerical order."
    )
    return system, user, targets, arabic, revised_by


def run_verifications(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    *,
    budget: BudgetLedger,
    workers: int,
    limit: int | None = None,
) -> None:
    model = _critic_model()
    jobs: list[tuple[ProductionUnit, Callable[[], dict[str, str]]]] = []
    for unit in _units_requiring_revision(base, units):
        system, user, targets, arabic, revised_by = _verification_input(
            base, unit, verses
        )
        validator = lambda value, targets=targets, arabic=arabic, revised_by=revised_by: validate_verification(
            value,
            expected=targets,
            arabic_by_ayah=arabic,
            proposed_by_ayah={ayah: revised_by[ayah] for ayah in targets},
        )
        jobs.append(
            (
                unit,
                lambda unit=unit, system=system, user=user, validator=validator: _sync_job(
                    base=base,
                    unit=unit,
                    stage="verification",
                    model=model,
                    system=system,
                    user=user,
                    schema=VERIFICATION_SCHEMA,
                    validator=validator,
                    budget=budget,
                ),
            )
        )
    _run_parallel(jobs[:limit] if limit else jobs, workers)


def _preliminary_rows(
    base: Path,
    units: list[ProductionUnit],
 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for unit in units:
        draft = _artifact_result(base, unit, "draft")
        critic = _artifact_result(base, unit, "critic")
        draft_by = {int(row["ayah"]): row for row in draft["ayahs"]}
        severe = [
            record
            for record in finding_records(critic)
            if record["severity"] in {"blocking", "significant"}
        ]
        if not severe:
            chosen = draft_by
        else:
            revision = _artifact_result(base, unit, "revision")
            verification = _artifact_result(base, unit, "verification")
            revised_by = {int(row["ayah"]): row for row in revision["ayahs"]}
            verified_by = {int(row["ayah"]): row for row in verification["ayahs"]}
            chosen = dict(draft_by)
            for ayah, row in revised_by.items():
                check = verified_by.get(ayah)
                if check and check["accept"]:
                    chosen[ayah] = row
                elif check:
                    unresolved.append(
                        {
                            "unit_id": unit.unit_id,
                            "ayah": ayah,
                            "reason": check["reason"],
                            "findings": check["findings"],
                            "fallback": "muse_base",
                        }
                    )
            decision_by = {
                str(row["finding_id"]): row for row in revision["decisions"]
            }
            for record in severe:
                decision = decision_by[str(record["finding_id"])]
                check = verified_by.get(int(record["ayah"]))
                if decision["decision"] == "escalated" or not check or not check["accept"]:
                    unresolved.append(
                        {
                            "unit_id": unit.unit_id,
                            "ayah": record["ayah"],
                            "finding": record,
                            "decision": decision,
                        }
                    )
        for ayah in unit.expected_ayahs:
            row = chosen[ayah]
            final_rows.append(
                {
                    "surah": unit.surah,
                    "ayah": ayah,
                    "urdu": str(row["urdu"]),
                    "review_flags": [
                        flag for flag in row.get("review_flags", []) if flag in ALLOWED_FLAGS
                    ],
                }
            )
    return final_rows, unresolved


REFRAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "urdu": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["group_id", "urdu", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}


def _validate_refrain_choices(
    document: Any, groups: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not isinstance(document, dict) or set(document) != {"groups"}:
        return None
    rows = document.get("groups")
    expected = [str(group["group_id"]) for group in groups]
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None
    if [row.get("group_id") for row in rows if isinstance(row, dict)] != expected:
        return None
    options = {
        str(group["group_id"]): set(group["options"]) for group in groups
    }
    clean: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"group_id", "urdu", "reason"}:
            return None
        group_id = str(row["group_id"])
        urdu = row.get("urdu")
        reason = row.get("reason")
        if urdu not in options[group_id] or not isinstance(reason, str) or not reason.strip():
            return None
        clean.append({"group_id": group_id, "urdu": str(urdu), "reason": reason.strip()})
    return {"groups": clean}


def run_refrain_resolution(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    *,
    budget: BudgetLedger,
    bundle_size: int = 20,
) -> dict[str, Any]:
    if bundle_size <= 0:
        raise UrduProductionError("Refrain bundle size must be positive")
    preliminary, _unresolved = _preliminary_rows(base, units)
    by_ref = {
        (int(row["surah"]), int(row["ayah"])): str(row["urdu"])
        for row in preliminary
    }
    divergent: list[dict[str, Any]] = []
    for group_id, group in repeated_ayah_groups(verses).items():
        refs = [tuple(ref) for ref in group["refs"]]
        options = sorted({by_ref[ref] for ref in refs})
        if len(options) > 1:
            divergent.append(
                {
                    "group_id": group_id,
                    "arabic": group["arabic"],
                    "refs": [f"{surah}:{ayah}" for surah, ayah in refs],
                    "options": options,
                }
            )
    model = _critic_model()
    system = _system_material(REFRAIN_PROMPT_PATH)
    resolved: dict[str, dict[str, str]] = {}
    for start in range(0, len(divergent), bundle_size):
        groups = divergent[start : start + bundle_size]
        bundle_number = start // bundle_size + 1
        path = base / "refrains" / f"bundle-{bundle_number:03d}.json"
        user = (
            "=== DIVERGENT IDENTICAL-ARABIC GROUPS ===\n"
            + json.dumps(groups, ensure_ascii=False, indent=2)
        )
        input_hash = stable_hash(
            {
                "stage": "refrain_resolution",
                "model": asdict(model),
                "system": system,
                "user": user,
                "schema": REFRAIN_SCHEMA,
                "contract_attempts": CONTRACT_ATTEMPTS,
            }
        )
        validator = lambda value, groups=groups: _validate_refrain_choices(value, groups)
        existing = _load_stage(path, input_hash=input_hash, validator=validator)
        if existing is None:
            errors: list[str] = []
            consumed_attempts: set[int] = set()
            for attempt in range(1, CONTRACT_ATTEMPTS + 1):
                failed_path = (
                    base
                    / "refrains"
                    / f"bundle-{bundle_number:03d}-attempt{attempt}-FAILED.json"
                )
                if not failed_path.exists():
                    continue
                failed = _load_json(failed_path)
                if failed.get("input_hash") != input_hash:
                    raise UrduProductionError(
                        f"Cached failed refrain input changed: {failed_path}"
                    )
                consumed_attempts.add(attempt)
                errors.extend(str(item) for item in failed.get("errors", []))
            raw_text = ""
            raw_response: dict[str, Any] | None = None
            usage: dict[str, Any] = {}
            for attempt in range(1, CONTRACT_ATTEMPTS + 1):
                if attempt in consumed_attempts:
                    continue
                raw_text = ""
                raw_response = None
                usage = {}
                assignment = user + (
                    "\n\nPrevious response failed the strict contract: " + errors[-1]
                    if errors
                    else ""
                )
                reservation = estimate_request_ceiling(
                    model.model_id, system, assignment, MAX_OUTPUT_TOKENS
                )
                budget.reserve(reservation)
                try:
                    raw_text, usage, raw_response = PROVIDER_CALLS[model.provider](
                        model, system, assignment, REFRAIN_SCHEMA
                    )
                    result = validator(extract_json(raw_text))
                    if result is None:
                        raise UrduProductionError(
                            "response failed strict refrain contract"
                        )
                    atomic_json(
                        path,
                        {
                            "version": "urdu-production-stage-v1",
                            "stage": "refrain_resolution",
                            "input_hash": input_hash,
                            "status": "complete",
                            "model": asdict(model),
                            "attempts": attempt,
                            "usage": usage,
                            "cost_usd": usage_cost(model.model_id, usage),
                            "result": result,
                            "errors_before_success": errors,
                            "raw_text": raw_text,
                            "raw_response": raw_response,
                        },
                    )
                    existing = result
                    break
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}"[:3000])
                    failed_payload: dict[str, Any] = {
                        "version": "urdu-production-stage-failure-v1",
                        "stage": "refrain_resolution",
                        "input_hash": input_hash,
                        "attempt": attempt,
                        "errors": [errors[-1]],
                        "usage": usage,
                        "raw_text": raw_text,
                        "raw_response": raw_response,
                    }
                    if usage:
                        failed_payload["cost_usd"] = usage_cost(model.model_id, usage)
                    atomic_json(
                        base
                        / "refrains"
                        / f"bundle-{bundle_number:03d}-attempt{attempt}-FAILED.json",
                        failed_payload,
                    )
                finally:
                    budget.release(reservation)
            if existing is None:
                raise UrduProductionError(
                    f"Refrain bundle {bundle_number} exhausted its contract: {errors}"
                )
        for row in existing["groups"]:
            resolved[str(row["group_id"])] = row
    marker = {
        "version": "urdu-refrain-resolution-v1",
        "divergent_groups": len(divergent),
        "resolved": resolved,
    }
    marker_path = base / "REFRAINS_COMPLETE.json"
    if marker_path.exists() and _load_json(marker_path) != marker:
        raise UrduProductionError("Frozen refrain resolution changed")
    atomic_json(marker_path, marker)
    return marker


def finalize(
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
) -> dict[str, Any]:
    final_rows, unresolved = _preliminary_rows(base, units)
    refrain_path = base / "REFRAINS_COMPLETE.json"
    if not refrain_path.is_file():
        raise UrduProductionError("Refrain resolution has not completed")
    refrain_marker = _load_json(refrain_path)
    resolved = refrain_marker.get("resolved")
    if not isinstance(resolved, dict):
        raise UrduProductionError("Refrain resolution marker is malformed")
    group_by_ref = {
        tuple(ref): group_id
        for group_id, group in repeated_ayah_groups(verses).items()
        for ref in group["refs"]
    }
    for row in final_rows:
        group_id = group_by_ref.get((int(row["surah"]), int(row["ayah"])))
        choice = resolved.get(group_id) if group_id else None
        if isinstance(choice, dict):
            row["urdu"] = str(choice["urdu"])

    translations = {
        (int(row["surah"]), int(row["ayah"])): str(row["urdu"])
        for row in final_rows
    }
    qa = deterministic_quality_gate(translations, verses)
    critic_findings: list[dict[str, Any]] = []
    for unit in units:
        critic = _artifact_result(base, unit, "critic")
        severe = any(
            record["severity"] in {"blocking", "significant"}
            for record in finding_records(critic)
        )
        decision_by: dict[str, dict[str, Any]] = {}
        verification_by: dict[int, dict[str, Any]] = {}
        if severe:
            revision = _artifact_result(base, unit, "revision")
            verification = _artifact_result(base, unit, "verification")
            decision_by = {
                str(row["finding_id"]): row for row in revision["decisions"]
            }
            verification_by = {
                int(row["ayah"]): row for row in verification["ayahs"]
            }
        for record in finding_records(critic):
            resolution = "advisory"
            decision = decision_by.get(str(record["finding_id"]))
            verification = verification_by.get(int(record["ayah"]))
            if record["severity"] in {"blocking", "significant"}:
                if (
                    decision
                    and decision["decision"] != "escalated"
                    and verification
                    and verification["accept"]
                ):
                    resolution = (
                        "critic_rejected_verified_base"
                        if decision["decision"] == "rejected"
                        else "repaired_and_verified"
                    )
                else:
                    resolution = "unresolved_muse_fallback"
            critic_findings.append(
                {
                    "unit_id": unit.unit_id,
                    **record,
                    "resolution": resolution,
                    "revision_decision": decision,
                    "verification": verification,
                }
            )
    translator_flags = [
        {
            "ref": f"{row['surah']}:{row['ayah']}",
            "flags": row["review_flags"],
        }
        for row in final_rows
        if row["review_flags"]
    ]
    output_dir = base / "output"
    atomic_json(output_dir / "quran-urdu.json", final_rows)
    atomic_text(
        output_dir / "quran-urdu.txt",
        "\n".join(
            f"{row['surah']}:{row['ayah']}\t{row['urdu']}" for row in final_rows
        )
        + "\n",
    )
    atomic_json(base / "QA_REPORT.json", {"quality": qa, "unresolved": unresolved})
    review_queue = {
        "version": "quran-urdu-review-queue-v1",
        "translator_flags": translator_flags,
        "critic_findings": critic_findings,
        "unresolved": unresolved,
        "counts": {
            "translator_flagged_ayahs": len(translator_flags),
            "critic_findings": len(critic_findings),
            "blocking_or_significant_findings": sum(
                item["severity"] in {"blocking", "significant"}
                for item in critic_findings
            ),
            "unresolved": len(unresolved),
        },
    }
    atomic_json(base / "REVIEW_QUEUE.json", review_queue)
    if unresolved or not qa["passed"]:
        marker = {
            "version": "quran-urdu-production-blocked-v1",
            "run_id": base.name,
            "unresolved_findings": len(unresolved),
            "deterministic_errors": qa["errors"],
            "qa_report": str(base / "QA_REPORT.json"),
            "review_queue": str(base / "REVIEW_QUEUE.json"),
        }
        atomic_json(base / "QA_BLOCKED.json", marker)
        return marker
    marker = {
        "version": "quran-urdu-production-complete-v1",
        "run_id": base.name,
        "ayahs": len(final_rows),
        "output_json": str(output_dir / "quran-urdu.json"),
        "output_text": str(output_dir / "quran-urdu.txt"),
        "qa_report": str(base / "QA_REPORT.json"),
        "review_queue": str(base / "REVIEW_QUEUE.json"),
    }
    atomic_json(base / "PRODUCTION_COMPLETE.json", marker)
    return marker


def production_status(base: Path, units: list[ProductionUnit]) -> dict[str, Any]:
    stages: dict[str, dict[str, int]] = {}
    for stage in ("draft", "critic", "revision", "verification"):
        if stage in {"revision", "verification"}:
            try:
                relevant = _units_requiring_revision(base, units)
            except UrduProductionError:
                relevant = []
        else:
            relevant = units
        complete = failed = 0
        for unit in relevant:
            path = artifact_path(base, unit, stage)
            if not path.exists():
                continue
            document = _load_json(path)
            if document.get("status") == "complete":
                complete += 1
            elif document.get("status") == "failed":
                failed += 1
        stages[stage] = {
            "total": len(relevant),
            "complete": complete,
            "failed": failed,
            "pending": len(relevant) - complete - failed,
        }
    manifest = _load_json(base / "MANIFEST.json")
    budget = BudgetLedger(base, float(manifest["config"]["hard_cost_ceiling_usd"]))
    return {
        "version": "quran-urdu-production-status-v1",
        "run_id": base.name,
        "stages": stages,
        "budget": budget.report(),
        "complete": (base / "PRODUCTION_COMPLETE.json").exists(),
        "qa_blocked": (base / "QA_BLOCKED.json").exists(),
        "refrains_complete": (base / "REFRAINS_COMPLETE.json").exists(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the gated Urdu Quran pipeline")
    parser.add_argument("command", choices=["prepare", "draft", "critic", "revise", "verify", "refrains", "finalize", "run", "status"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--hard-cost-ceiling-usd", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    return parser.parse_args()


def _validate_cli_args(command: str, limit: int | None) -> None:
    if limit is not None and limit <= 0:
        raise UrduProductionError("--limit must be positive")
    if command == "run" and limit is not None:
        raise UrduProductionError(
            "--limit is only valid for individual draft, critic, or verify stages"
        )


def main() -> None:
    args = parse_args()
    _validate_cli_args(args.command, args.limit)
    config = UrduProductionConfig(
        run_id=args.run_id,
        hard_cost_ceiling_usd=args.hard_cost_ceiling_usd,
        workers=args.workers,
    )
    with connect(args.db) as conn:
        base, units, verses, bismillah = prepare_production(conn, config)
    budget = BudgetLedger(base, config.hard_cost_ceiling_usd)
    if args.command == "prepare":
        result = production_status(base, units)
    elif args.command == "status":
        result = production_status(base, units)
    else:
        load_environment()
        if args.command in {"draft", "run"}:
            run_drafts(
                base,
                units,
                verses,
                bismillah,
                budget=budget,
                workers=config.workers,
                limit=args.limit,
            )
        if args.command in {"critic", "run"}:
            run_critics(
                base,
                units,
                verses,
                bismillah,
                budget=budget,
                workers=config.workers,
                limit=args.limit,
            )
        if args.command in {"revise", "run"}:
            run_revisions(
                base,
                units,
                verses,
                bismillah,
                budget=budget,
                poll_seconds=args.poll_seconds,
            )
        if args.command in {"verify", "run"}:
            run_verifications(
                base,
                units,
                verses,
                budget=budget,
                workers=config.workers,
                limit=args.limit,
            )
        if args.command in {"refrains", "run"}:
            run_refrain_resolution(base, units, verses, budget=budget)
        if args.command in {"finalize", "run"}:
            result = finalize(base, units, verses)
        else:
            result = production_status(base, units)
    atomic_json(base / "RUN.json", production_status(base, units))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
