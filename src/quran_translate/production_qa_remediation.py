"""Bounded, auditable remediation for terminal production QA blockers."""

from __future__ import annotations

from .production_clients import submit_batch_once

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from .config import DEFAULT_DB_PATH
from .critic_v2 import CRITIC_SYSTEM, FINDING_TYPES, SEVERITIES
from .db import connect, utc_now
from .production_clients import (
    AnthropicBatchClient,
    BatchState,
    GeminiSynchronousClient,
    ProviderError,
    anthropic_result_text,
)
from .production_packets import ProductionUnit, atomic_json, atomic_text
from .production_runner import (
    ANTHROPIC_EFFORT,
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_MODEL,
    GEMINI_MAX_TOKENS,
    GEMINI_MODEL,
    ProductionConfig,
    ProductionError,
    _anthropic_params,
    _gemini_text,
    cached_system_blocks,
    extract_json,
    file_hash,
    load_environment,
    prepare_production,
    production_status,
    prompt_material,
    run_dir,
    stable_hash,
)
from .refrains import repeated_ayah_groups
from .validation import PRODUCTION_V24_BANNED_TERMS, validate_run, validate_source


REMEDIATION_VERSION = "production-qa-remediation-v1"
REMEDIATION_DIR = "qa-remediation"
TARGETS_FILE = "TARGETS.json"
MANIFEST_FILE = "MANIFEST.json"
ADJUDICATION_FILE = "OPUS_ADJUDICATION.json"
VERIFICATION_FILE = "GEMINI_VERIFICATION.json"
CORRECTIONS_FILE = "CORRECTIONS.json"
REPORT_JSON = "QA_REMEDIATION_REPORT.json"
REPORT_MD = "QA_REMEDIATION_REPORT.md"
COMPLETE_FILE = "QA_REMEDIATION_COMPLETE.json"
BLOCKED_FILE = "QA_REMEDIATION_BLOCKED.json"
OPUS_ATTEMPTS = (1, 2)
GEMINI_ATTEMPTS = (1, 2)
VERIFICATION_BUNDLE_SIZE = 4


QA_REMEDIATION_SYSTEM = """You are the adjudicating reviser for a Quran translation
that has completed its full production run but stopped at a terminal QA gate.

The supplied QA findings are hypotheses, not commands. For every finding, decide
whether it identifies a real fidelity or registered-policy defect. Apply the smallest
correction that resolves an upheld defect. If a finding is false, reject it and keep
the existing English unless another upheld finding requires a change. Escalate only
when the supplied Arabic, context, evidence, and ledger do not support a responsible
decision.

Priority is fidelity, then preserved ambiguity, natural spoken English, and literary
force. Do not add imagery, agency, causality, motive, specificity, temporal sequence,
or moral judgment. Do not launder model-generated historical claims into the text or
reason. Morphology constrains form but does not decide contextual sense. Reasons may
cite only the supplied Arabic, morphology, context, concordance parallels, ledger, or
project policy.

Some targets are closure members of an identical-Arabic group. Every member of such
a group must receive exactly identical English. Choose one rendering that remains
faithful and grammatically usable in every supplied context. Return only the JSON
contract, with every target ref exactly once and every finding decision exactly once.
"""


ADJUDICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "english": {"type": "string"},
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
                "required": ["ref", "english", "decisions"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "ref": {"type": "string"},
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
                    "additionalProperties": False,
                },
            },
            "verdict": {"type": "string", "enum": ["pass", "revise"]},
        },
        "required": ["ref", "findings", "verdict"],
        "additionalProperties": False,
    },
}


def _ref_tuple(ref: str) -> tuple[int, int]:
    surah, ayah = ref.split(":", 1)
    return int(surah), int(ayah)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _load_run_config(base: Path) -> ProductionConfig:
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ProductionError("Production manifest lacks config")
    return ProductionConfig(**config)


def _unit_for_ref(units: list[ProductionUnit], ref: str) -> ProductionUnit:
    surah, ayah = _ref_tuple(ref)
    for unit in units:
        if unit.surah == surah and unit.first_ayah <= ayah <= unit.last_ayah:
            return unit
    raise ProductionError(f"No production unit contains {ref}")


def _morphology_excerpt(base: Path, unit: ProductionUnit, ref: str) -> str:
    packet = (base / "evidence" / f"{unit.unit_id}.md").read_text(encoding="utf-8")
    marker = f"## {ref} morphology\n"
    start = packet.find(marker)
    if start < 0:
        return ""
    end = packet.find("\n## ", start + len(marker))
    return packet[start : end if end >= 0 else len(packet)].strip()


def _finding_id(record: dict[str, Any]) -> str:
    return "qa-" + stable_hash(
        {
            "ref": record["ref"],
            "stage": record.get("stage"),
            "type": record["type"],
            "where": record.get("where"),
            "explanation": record["explanation"],
        }
    )[:16]


def _issue_finding(
    issue: dict[str, Any],
    fidelity_findings: list[dict[str, Any]],
    english: dict[str, str],
) -> dict[str, Any]:
    ref = str(issue["ref"])
    if issue["code"] == "run_validation":
        match = re.search(r"Banned/jargon term\(s\):\s*(.+)$", issue["message"])
        if not match:
            raise ProductionError(f"Cannot parse deterministic QA issue: {issue}")
        term = match.group(1).split(",", 1)[0].strip()
        source = english[ref]
        where_match = re.search(rf"\b{re.escape(term)}\b", source, re.I)
        if not where_match:
            raise ProductionError(f"Banned term {term!r} is absent from {ref}")
        record = {
            "ref": ref,
            "stage": "deterministic_validation",
            "type": "register_error",
            "severity": "blocking",
            "where": where_match.group(0),
            "arabic_ground": "",
            "explanation": issue["message"],
            "suggestion": "Use current spoken English without changing the claim.",
        }
    else:
        candidates = [
            finding
            for finding in fidelity_findings
            if finding.get("ref") == ref
            and f"{finding.get('type')}: {finding.get('explanation')}"
            == issue["message"]
        ]
        if len(candidates) != 1:
            raise ProductionError(
                f"Expected one source finding for {ref}, found {len(candidates)}"
            )
        record = {"ref": ref, **candidates[0]}
    record["finding_id"] = _finding_id(record)
    return record


def build_targets(
    *,
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    english: dict[str, str],
) -> list[dict[str, Any]]:
    qa = json.loads((base / "QA_REPORT.json").read_text(encoding="utf-8"))
    if qa.get("passed") is not False:
        raise ProductionError("Remediation requires a blocked production QA report")
    review = json.loads((base / "REVIEW_QUEUE.json").read_text(encoding="utf-8"))
    fidelity_findings = review.get("fidelity_findings")
    if not isinstance(fidelity_findings, list):
        raise ProductionError("Review queue lacks fidelity findings")

    blockers = [
        issue
        for issue in qa.get("issues", [])
        if issue.get("severity") == "error" and issue.get("ref")
    ]
    unsupported = [
        issue
        for issue in blockers
        if issue.get("code") not in {"run_validation", "unresolved_fidelity_finding"}
    ]
    if unsupported:
        raise ProductionError(f"Unsupported QA blockers: {unsupported}")

    findings_by_ref: dict[str, list[dict[str, Any]]] = {}
    for issue in blockers:
        finding = _issue_finding(issue, fidelity_findings, english)
        findings_by_ref.setdefault(str(issue["ref"]), []).append(finding)

    groups = repeated_ayah_groups(verses)
    group_by_ref: dict[str, tuple[str, dict[str, Any]]] = {
        f"{surah}:{ayah}": (group_id, group)
        for group_id, group in groups.items()
        for surah, ayah in group["refs"]
    }
    target_refs = set(findings_by_ref)
    for ref in list(target_refs):
        if ref in group_by_ref:
            target_refs.update(
                f"{surah}:{ayah}" for surah, ayah in group_by_ref[ref][1]["refs"]
            )

    refrain_report = json.loads((base / "REFRAINS.json").read_text(encoding="utf-8"))
    governed_groups = refrain_report.get("groups", {})
    targets: list[dict[str, Any]] = []
    for ref in sorted(target_refs, key=_ref_tuple):
        surah, ayah = _ref_tuple(ref)
        unit = _unit_for_ref(units, ref)
        records = findings_by_ref.get(ref, [])
        parallels: list[dict[str, str]] = []
        grounds = {
            str(finding.get("arabic_ground"))
            for finding in records
            if finding.get("arabic_ground")
        }
        for ground in sorted(grounds):
            matches = [
                (candidate_ayah, arabic)
                for (candidate_surah, candidate_ayah), arabic in verses.items()
                if candidate_surah == surah and ground in arabic and candidate_ayah != ayah
            ][:8]
            for candidate_ayah, arabic in matches:
                candidate_ref = f"{surah}:{candidate_ayah}"
                parallels.append(
                    {
                        "ref": candidate_ref,
                        "arabic": arabic,
                        "english": english[candidate_ref],
                    }
                )

        context = [
            {
                "ref": f"{surah}:{candidate}",
                "arabic": verses[(surah, candidate)],
                "english": english[f"{surah}:{candidate}"],
            }
            for candidate in range(max(1, ayah - 2), ayah + 3)
            if (surah, candidate) in verses
        ]
        group_record: dict[str, Any] | None = None
        if ref in group_by_ref:
            group_id, group = group_by_ref[ref]
            policy = governed_groups.get(group_id, {})
            group_record = {
                "group_id": group_id,
                "refs": [f"{s}:{a}" for s, a in group["refs"]],
                "required_identical": True,
                "existing_reason": policy.get("reason"),
                "existing_source": policy.get("source"),
            }
        targets.append(
            {
                "ref": ref,
                "unit_id": unit.unit_id,
                "arabic": verses[(surah, ayah)],
                "current_english": english[ref],
                "findings": records,
                "closure_only": not records,
                "identical_group": group_record,
                "local_context": context,
                "parallel_occurrences": parallels,
                "morphology": _morphology_excerpt(base, unit, ref),
            }
        )
    return targets


def _validate_decisions(
    document: Any,
    targets: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(document, dict) or set(document) != {"items"}:
        return None
    items = document.get("items")
    if not isinstance(items, list) or len(items) != len(targets):
        return None
    expected_refs = [target["ref"] for target in targets]
    if [item.get("ref") for item in items if isinstance(item, dict)] != expected_refs:
        return None

    expected_findings = {
        finding["finding_id"]: target["ref"]
        for target in targets
        for finding in target["findings"]
    }
    seen: set[str] = set()
    item_by_ref: dict[str, dict[str, Any]] = {}
    decision_by_id: dict[str, dict[str, str]] = {}
    for target, item in zip(targets, items, strict=True):
        if not isinstance(item, dict) or set(item) != {"ref", "english", "decisions"}:
            return None
        english = item.get("english")
        decisions = item.get("decisions")
        if not isinstance(english, str) or not english.strip() or not isinstance(decisions, list):
            return None
        target_id_order = [finding["finding_id"] for finding in target["findings"]]
        target_ids = set(target_id_order)
        for decision in decisions:
            if not isinstance(decision, dict) or set(decision) != {
                "finding_id",
                "decision",
                "reason",
            }:
                return None
            finding_id = decision.get("finding_id")
            status = decision.get("decision")
            reason = decision.get("reason")
            if (
                finding_id not in target_ids
                or finding_id in seen
                or status not in {"applied", "rejected", "escalated"}
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                return None
            seen.add(str(finding_id))
            decision_by_id[str(finding_id)] = {
                "finding_id": str(finding_id),
                "decision": str(status),
                "reason": reason.strip(),
            }
        if {decision["finding_id"] for decision in decisions} != target_ids:
            return None
        item_by_ref[target["ref"]] = {
            "ref": target["ref"],
            "english": english.strip(),
            "decisions": [decision_by_id[finding_id] for finding_id in target_id_order],
        }
    if seen != set(expected_findings):
        return None

    groups: dict[str, list[str]] = {}
    for target in targets:
        group = target.get("identical_group")
        if group:
            groups[str(group["group_id"])] = list(group["refs"])
    for refs in groups.values():
        if len({item_by_ref[ref]["english"] for ref in refs}) != 1:
            return None

    applied_groups = {
        str(target["identical_group"]["group_id"])
        for target in targets
        if target.get("identical_group")
        and any(
            decision_by_id[finding["finding_id"]]["decision"] == "applied"
            for finding in target["findings"]
        )
    }
    for target in targets:
        item = item_by_ref[target["ref"]]
        decisions = [decision_by_id[f["finding_id"]] for f in target["findings"]]
        group = target.get("identical_group")
        group_applied = bool(group and group["group_id"] in applied_groups)
        if any(d["decision"] == "applied" for d in decisions):
            if item["english"] == target["current_english"] and not group_applied:
                return None
        elif not group_applied and item["english"] != target["current_english"]:
            return None
    target_by_ref = {target["ref"]: target for target in targets}
    for group_id in applied_groups:
        refs = groups[group_id]
        if all(
            item_by_ref[ref]["english"] == target_by_ref[ref]["current_english"]
            for ref in refs
        ):
            return None
    return {"items": [item_by_ref[ref] for ref in expected_refs]}


def _wait_batch(
    client: AnthropicBatchClient,
    state: BatchState,
    poll_seconds: int,
    on_poll: Callable[[BatchState], None],
) -> BatchState:
    current = state
    while not current.ended:
        time.sleep(poll_seconds)
        current = client.retrieve(current.batch_id)
        on_poll(current)
    return current


def run_opus_adjudication(
    *,
    remediation: Path,
    targets: list[dict[str, Any]],
    system: list[dict[str, Any]],
    client: AnthropicBatchClient,
    poll_seconds: int,
) -> dict[str, Any]:
    artifact = remediation / ADJUDICATION_FILE
    user = (
        "=== FROZEN TERMINAL-QA TARGETS ===\n"
        + json.dumps(targets, ensure_ascii=False, indent=2)
        + "\n\nReturn the adjudication contract for every target in exactly this order."
    )
    input_hash = stable_hash(
        {
            "version": REMEDIATION_VERSION,
            "model": ANTHROPIC_MODEL,
            "system": system,
            "user": user,
            "schema": ADJUDICATION_SCHEMA,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "effort": ANTHROPIC_EFFORT,
        }
    )
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("input_hash") != input_hash:
            raise ProductionError("Cached QA adjudication input changed")
        result = _validate_decisions(payload.get("result"), targets)
        if result is None:
            raise ProductionError("Cached QA adjudication fails contract")
        return result

    jobs = remediation / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    for attempt in OPUS_ATTEMPTS:
        attempt_user = user
        if attempt > 1:
            attempt_user += (
                "\n\nCONTRACT RETRY: Return every ref and every finding decision exactly "
                "once. Keep identical-group English byte-identical. Return JSON only."
            )
        request = {
            "custom_id": "qa-remediation",
            "params": _anthropic_params(
                system=system,
                user=attempt_user,
                response_schema=ADJUDICATION_SCHEMA,
            ),
        }
        request_hash = stable_hash(request)
        job_path = jobs / f"opus-a{attempt}.json"
        if job_path.exists():
            job = json.loads(job_path.read_text(encoding="utf-8"))
            if job.get("request_hash") != request_hash:
                raise ProductionError(f"Opus remediation job changed: {job_path}")
            state = client.retrieve(str(job["batch_id"]))
        else:
            state = submit_batch_once(client, [request], job_path)
            atomic_json(
                job_path,
                {
                    "provider": "anthropic",
                    "attempt": attempt,
                    "request_hash": request_hash,
                    "batch_id": state.batch_id,
                    "state": state.state,
                    "submitted_at": utc_now(),
                },
            )

        def on_poll(current: BatchState) -> None:
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job.update({"state": current.state, "last_polled_at": utc_now()})
            atomic_json(job_path, job)

        state = _wait_batch(client, state, poll_seconds, on_poll)
        on_poll(state)
        if not state.succeeded:
            atomic_json(
                remediation / f"opus-attempt{attempt}-FAILED.json",
                {"attempt": attempt, "error": f"Batch ended in {state.state}"},
            )
            continue
        try:
            rows = client.results(state.batch_id)
            row = next(row for row in rows if row.get("custom_id") == "qa-remediation")
            raw, usage = anthropic_result_text(row)
            result = _validate_decisions(extract_json(raw), targets)
        except (ProviderError, StopIteration, json.JSONDecodeError, TypeError, ValueError) as exc:
            result = None
            raw = repr(exc)
            usage = {}
        if result is None:
            atomic_json(
                remediation / f"opus-attempt{attempt}-FAILED.json",
                {"attempt": attempt, "raw": raw, "usage": usage},
            )
            continue
        atomic_json(
            artifact,
            {
                "input_hash": input_hash,
                "model": ANTHROPIC_MODEL,
                "attempt": attempt,
                "batch_id": state.batch_id,
                "usage": usage,
                "result": result,
                "raw": raw,
            },
        )
        return result
    raise ProductionError("Opus QA adjudication failed its contract twice")


def _validate_verification(
    document: Any,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    expected_refs = [target["ref"] for target in targets]
    if not isinstance(document, list) or len(document) != len(targets):
        return None
    if [row.get("ref") for row in document if isinstance(row, dict)] != expected_refs:
        return None
    target_by_ref = {target["ref"]: target for target in targets}
    for row in document:
        if not isinstance(row, dict) or set(row) != {"ref", "findings", "verdict"}:
            return None
        findings = row.get("findings")
        verdict = row.get("verdict")
        if not isinstance(findings, list) or verdict not in {"pass", "revise"}:
            return None
        target = target_by_ref[row["ref"]]
        for finding in findings:
            if not isinstance(finding, dict):
                return None
            required = {"type", "severity", "where", "arabic_ground", "explanation"}
            if not required.issubset(finding) or set(finding) - (required | {"suggestion"}):
                return None
            if finding.get("type") not in FINDING_TYPES or finding.get("severity") not in SEVERITIES:
                return None
            for field in ("where", "arabic_ground", "explanation"):
                if not isinstance(finding.get(field), str) or not finding[field].strip():
                    return None
            if "suggestion" in finding and (
                not isinstance(finding["suggestion"], str) or not finding["suggestion"].strip()
            ):
                return None
            if finding["where"] == "<missing>" and finding["type"] != "omission":
                return None
            if finding["where"] != "<missing>" and _normalize(finding["where"]) not in _normalize(target["candidate_english"]):
                return None
            if _normalize(finding["arabic_ground"]) not in _normalize(target["arabic"]):
                return None
        major = any(finding["severity"] in {"blocking", "significant"} for finding in findings)
        if major and verdict != "revise":
            return None
        if verdict == "pass" and findings:
            return None
        if verdict == "revise" and not findings:
            return None
    return document


def _verification_user(targets: list[dict[str, Any]], attempt: int) -> str:
    compact = [
        {
            "ref": target["ref"],
            "arabic": target["arabic"],
            "candidate_english": target["candidate_english"],
            "local_context": target["local_context"],
            "morphology": target["morphology"],
        }
        for target in targets
    ]
    retry = (
        " This is a contract retry: keep findings concise and copy exact Arabic and "
        "English spans so every requested ref is returned."
        if attempt > 1
        else ""
    )
    return (
        "Freshly audit the candidate English below against its Arabic. Do not assume "
        "that a change is correct merely because it came from a remediation pass. "
        "Return only the registered ref-based JSON array in the supplied order."
        + retry
        + "\n\n=== REMEDIATED ENGLISH TO AUDIT ===\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    )


def run_gemini_verification(
    *,
    remediation: Path,
    targets: list[dict[str, Any]],
    system: str,
    client: GeminiSynchronousClient,
    delay_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    artifact = remediation / VERIFICATION_FILE
    bundles = [
        targets[index : index + VERIFICATION_BUNDLE_SIZE]
        for index in range(0, len(targets), VERIFICATION_BUNDLE_SIZE)
    ]
    input_hash = stable_hash(
        {
            "version": REMEDIATION_VERSION,
            "model": GEMINI_MODEL,
            "system": system,
            "targets": targets,
            "schema": VERIFICATION_SCHEMA,
            "max_tokens": GEMINI_MAX_TOKENS,
        }
    )
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("input_hash") != input_hash:
            raise ProductionError("Cached remediation verification input changed")
        result = _validate_verification(payload.get("result"), targets)
        if result is None:
            raise ProductionError("Cached remediation verification fails contract")
        return result

    jobs = remediation / "jobs"
    verification_dir = remediation / "verification"
    jobs.mkdir(parents=True, exist_ok=True)
    verification_dir.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, Any]] = []
    for bundle_index, bundle in enumerate(bundles, start=1):
        bundle_artifact = verification_dir / f"bundle-{bundle_index:03d}.json"
        bundle_hash = stable_hash(
            {"model": GEMINI_MODEL, "system": system, "targets": bundle, "schema": VERIFICATION_SCHEMA}
        )
        if bundle_artifact.exists():
            payload = json.loads(bundle_artifact.read_text(encoding="utf-8"))
            if payload.get("input_hash") != bundle_hash:
                raise ProductionError(f"Verification bundle changed: {bundle_artifact}")
            result = _validate_verification(payload.get("result"), bundle)
            if result is None:
                raise ProductionError(f"Verification bundle fails contract: {bundle_artifact}")
            combined.extend(result)
            continue

        result: list[dict[str, Any]] | None = None
        for attempt in GEMINI_ATTEMPTS:
            user = _verification_user(bundle, attempt)
            request_hash = stable_hash(
                {
                    "model": GEMINI_MODEL,
                    "system": system,
                    "user": user,
                    "response_schema": VERIFICATION_SCHEMA,
                    "max_output_tokens": GEMINI_MAX_TOKENS,
                    "temperature": 0,
                }
            )
            job_path = jobs / f"gemini-b{bundle_index:03d}-a{attempt}.json"
            if job_path.exists():
                job = json.loads(job_path.read_text(encoding="utf-8"))
                if job.get("request_hash") != request_hash:
                    raise ProductionError(f"Gemini remediation job changed: {job_path}")
                row = job.get("row")
                if not isinstance(row, dict):
                    raise ProductionError(f"Gemini remediation job lacks response: {job_path}")
            else:
                try:
                    row = client.generate(
                        model=GEMINI_MODEL,
                        system=system,
                        user=user,
                        response_schema=VERIFICATION_SCHEMA,
                        max_output_tokens=GEMINI_MAX_TOKENS,
                    )
                except ProviderError as exc:
                    atomic_json(
                        verification_dir / f"bundle-{bundle_index:03d}-attempt{attempt}-FAILED.json",
                        {"attempt": attempt, "error": str(exc)},
                    )
                    continue
                atomic_json(
                    job_path,
                    {
                        "provider": "google",
                        "transport": "sync",
                        "attempt": attempt,
                        "bundle": bundle_index,
                        "request_hash": request_hash,
                        "completed_at": utc_now(),
                        "row": row,
                    },
                )
            try:
                raw, usage = _gemini_text(row)
                result = _validate_verification(extract_json(raw), bundle)
            except (ProviderError, json.JSONDecodeError, TypeError, ValueError):
                result = None
                raw = json.dumps(row, ensure_ascii=False)
                usage = {}
            if result is None:
                atomic_json(
                    verification_dir / f"bundle-{bundle_index:03d}-attempt{attempt}-FAILED.json",
                    {"attempt": attempt, "raw": raw, "usage": usage},
                )
                continue
            atomic_json(
                bundle_artifact,
                {
                    "input_hash": bundle_hash,
                    "model": GEMINI_MODEL,
                    "attempt": attempt,
                    "usage": usage,
                    "result": result,
                    "raw": raw,
                },
            )
            if delay_seconds:
                time.sleep(delay_seconds)
            break
        if result is None:
            raise ProductionError(
                f"Gemini remediation verification failed twice for bundle {bundle_index}"
            )
        combined.extend(result)

    validated = _validate_verification(combined, targets)
    if validated is None:
        raise ProductionError("Combined remediation verification fails contract")
    atomic_json(
        artifact,
        {
            "input_hash": input_hash,
            "model": GEMINI_MODEL,
            "bundles": len(bundles),
            "result": validated,
            "bundle_hashes": {
                f"{index:03d}": file_hash(remediation / "verification" / f"bundle-{index:03d}.json")
                for index in range(1, len(bundles) + 1)
            },
        },
    )
    return validated


def _candidate_targets(
    targets: list[dict[str, Any]],
    adjudication: dict[str, Any],
) -> list[dict[str, Any]]:
    item_by_ref = {item["ref"]: item for item in adjudication["items"]}
    candidate_by_ref = {
        ref: item["english"] for ref, item in item_by_ref.items()
    }
    candidates = [
        {
            **target,
            "candidate_english": item_by_ref[target["ref"]]["english"],
            "decisions": item_by_ref[target["ref"]]["decisions"],
            "local_context": [
                {
                    **row,
                    "english": candidate_by_ref.get(row["ref"], row["english"]),
                }
                for row in target["local_context"]
            ],
            "parallel_occurrences": [
                {
                    **row,
                    "english": candidate_by_ref.get(row["ref"], row["english"]),
                }
                for row in target["parallel_occurrences"]
            ],
        }
        for target in targets
    ]
    return candidates


def _deterministic_candidate_issues(
    *,
    verses: dict[tuple[int, int], str],
    english: dict[str, str],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    expected = {f"{surah}:{ayah}" for surah, ayah in verses}
    if set(english) != expected:
        issues.append({"code": "coverage", "message": "Candidate reference set changed"})
    for ref, text in english.items():
        if not text.strip():
            issues.append({"code": "empty", "ref": ref, "message": "Empty candidate English"})
        for term in PRODUCTION_V24_BANNED_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", text, re.I):
                issues.append(
                    {"code": "banned_term", "ref": ref, "message": f"Banned term: {term}"}
                )
    for group in repeated_ayah_groups(verses).values():
        refs = [f"{surah}:{ayah}" for surah, ayah in group["refs"]]
        if len({english[ref] for ref in refs}) != 1:
            issues.append(
                {
                    "code": "refrain_divergence",
                    "ref": ", ".join(refs),
                    "message": "Identical Arabic has divergent candidate English",
                }
            )
    return issues


def _persist_candidates(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    candidates: list[dict[str, Any]],
    remediation_hash: str,
) -> None:
    now = utc_now()
    for target in candidates:
        ref = target["ref"]
        row = conn.execute(
            "SELECT translation, raw_translation_json FROM translations WHERE run_id = ? AND verse_key = ?",
            (run_id, ref),
        ).fetchone()
        if row is None:
            raise ProductionError(f"Cannot persist remediation; missing {ref}")
        raw = json.loads(row["raw_translation_json"])
        raw["qa_remediation"] = {
            "version": REMEDIATION_VERSION,
            "source_translation": target["current_english"],
            "translation": target["candidate_english"],
            "decisions": target["decisions"],
            "remediation_sha256": remediation_hash,
        }
        conn.execute(
            """
            UPDATE translations
            SET translation = ?, raw_translation_json = ?, updated_at = ?
            WHERE run_id = ? AND verse_key = ?
            """,
            (
                target["candidate_english"],
                json.dumps(raw, ensure_ascii=False, sort_keys=True),
                now,
                run_id,
                ref,
            ),
        )


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Production QA Remediation Report",
        "",
        f"- Verdict: **{'PASS' if report['passed'] else 'BLOCKED'}**",
        f"- Run: `{report['run_id']}`",
        f"- Original gate errors: {report['original_gate_errors']}",
        f"- Target refs after invariant closure: {report['target_refs']}",
        f"- Applied findings: {report['decisions']['applied']}",
        f"- Rejected findings: {report['decisions']['rejected']}",
        f"- Escalated findings: {report['decisions']['escalated']}",
        f"- Changed ayahs: {report['changed_ayahs']}",
        f"- Fresh major verification findings: {report['major_verification_findings']}",
        "",
        "## Issues",
        "",
    ]
    if report["issues"]:
        for issue in report["issues"]:
            ref = f" ({issue['ref']})" if issue.get("ref") else ""
            lines.append(f"- `{issue['code']}`{ref}: {issue['message']}")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Governance",
            "",
            "The original production artifacts and terminal QA report remain immutable. "
            "This report governs only the hashed remediation overlay and its fresh verification.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_remediation(
    conn: sqlite3.Connection,
    *,
    base: Path,
    units: list[ProductionUnit],
    verses: dict[tuple[int, int], str],
    targets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    verification: list[dict[str, Any]],
) -> dict[str, Any]:
    remediation = base / REMEDIATION_DIR
    qa = json.loads((base / "QA_REPORT.json").read_text(encoding="utf-8"))
    decisions = [decision for target in candidates for decision in target["decisions"]]
    major = [
        {"ref": row["ref"], **finding}
        for row in verification
        for finding in row["findings"]
        if finding["severity"] in {"blocking", "significant"}
    ]
    issues: list[dict[str, str]] = []
    for decision in decisions:
        if decision["decision"] == "escalated":
            issues.append(
                {
                    "code": "adjudication_escalated",
                    "ref": decision["finding_id"],
                    "message": decision["reason"],
                }
            )
    for finding in major:
        issues.append(
            {
                "code": "remediation_verification",
                "ref": finding["ref"],
                "message": f"{finding['type']}: {finding['explanation']}",
            }
        )

    english = {
        str(row["verse_key"]): str(row["translation"])
        for row in conn.execute(
            "SELECT verse_key, translation FROM translations WHERE run_id = ?",
            (base.name,),
        )
    }
    for target in candidates:
        english[target["ref"]] = target["candidate_english"]
    issues.extend(_deterministic_candidate_issues(verses=verses, english=english))

    counts = {status: sum(d["decision"] == status for d in decisions) for status in ("applied", "rejected", "escalated")}
    changed = [target for target in candidates if target["candidate_english"] != target["current_english"]]
    report = {
        "version": REMEDIATION_VERSION,
        "run_id": base.name,
        "passed": not issues,
        "original_qa_sha256": file_hash(base / "QA_REPORT.json"),
        "original_gate_errors": qa["issue_counts"]["error"],
        "target_refs": len(targets),
        "changed_ayahs": len(changed),
        "decisions": counts,
        "verification_findings": sum(len(row["findings"]) for row in verification),
        "major_verification_findings": len(major),
        "issues": issues,
    }
    atomic_json(remediation / REPORT_JSON, report)
    atomic_text(remediation / REPORT_MD, _markdown_report(report))
    if issues:
        atomic_json(remediation / BLOCKED_FILE, report)
        raise ProductionError(
            f"QA remediation blocked by {len(issues)} issue(s); see {remediation / REPORT_MD}"
        )

    corrections = {
        "version": REMEDIATION_VERSION,
        "run_id": base.name,
        "targets": [
            {
                "ref": target["ref"],
                "before": target["current_english"],
                "after": target["candidate_english"],
                "changed": target["candidate_english"] != target["current_english"],
                "decisions": target["decisions"],
            }
            for target in candidates
        ],
    }
    atomic_json(remediation / CORRECTIONS_FILE, corrections)
    remediation_hash = file_hash(remediation / CORRECTIONS_FILE)

    conn.execute("BEGIN")
    try:
        _persist_candidates(
            conn,
            run_id=base.name,
            candidates=candidates,
            remediation_hash=remediation_hash,
        )
        run_issues = validate_source(conn) + validate_run(conn, base.name)
        hard_run_issues = [
            issue
            for issue in run_issues
            if issue.severity == "error"
            or issue.message.startswith("Banned/jargon term")
        ]
        if hard_run_issues:
            raise ProductionError(
                "Post-remediation database validation failed: "
                + "; ".join(issue.message for issue in hard_run_issues[:10])
            )
        conn.execute(
            "UPDATE translation_runs SET status = 'complete', updated_at = ? WHERE run_id = ?",
            (utc_now(), base.name),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        blocked_report = {
            **report,
            "passed": False,
            "issues": [
                *report["issues"],
                {
                    "code": "post_persistence_validation",
                    "message": str(exc),
                },
            ],
        }
        atomic_json(remediation / REPORT_JSON, blocked_report)
        atomic_text(remediation / REPORT_MD, _markdown_report(blocked_report))
        atomic_json(remediation / BLOCKED_FILE, blocked_report)
        raise

    status = production_status(base, units)
    status.update(
        {
            "completion_mode": REMEDIATION_VERSION,
            "quality": {
                "passed": True,
                "original_gate_errors": report["original_gate_errors"],
                "remediation_report": f"{REMEDIATION_DIR}/{REPORT_JSON}",
                "changed_ayahs": report["changed_ayahs"],
            },
            "remediation_sha256": remediation_hash,
        }
    )
    atomic_json(remediation / COMPLETE_FILE, status)
    (remediation / BLOCKED_FILE).unlink(missing_ok=True)
    (base / "QA_BLOCKED.json").unlink(missing_ok=True)
    atomic_json(base / "PRODUCTION_COMPLETE.json", status)
    return status


def _freeze_inputs(base: Path, targets: list[dict[str, Any]]) -> Path:
    remediation = base / REMEDIATION_DIR
    remediation.mkdir(parents=True, exist_ok=True)
    targets_path = remediation / TARGETS_FILE
    target_record = {"version": REMEDIATION_VERSION, "targets": targets}
    if targets_path.exists():
        if json.loads(targets_path.read_text(encoding="utf-8")) != target_record:
            raise ProductionError("Frozen remediation targets changed")
    else:
        atomic_json(targets_path, target_record)
    manifest = {
        "version": REMEDIATION_VERSION,
        "production_manifest_sha256": file_hash(base / "manifest.json"),
        "qa_report_sha256": file_hash(base / "QA_REPORT.json"),
        "review_queue_sha256": file_hash(base / "REVIEW_QUEUE.json"),
        "refrains_sha256": file_hash(base / "REFRAINS.json"),
        "remediation_code_sha256": file_hash(Path(__file__)),
        "targets_sha256": file_hash(targets_path),
        "prompt_sha256": file_hash(Path(__file__).parents[2] / "prompts" / "production-v2.4.md"),
        "ledger_md_sha256": file_hash(Path(__file__).parents[2] / "prompts" / "sense-ledger-v2.4.md"),
        "ledger_json_sha256": file_hash(Path(__file__).parents[2] / "data" / "evidence" / "sense-ledger-v2.4.json"),
        "models": {"adjudicator": ANTHROPIC_MODEL, "verifier": GEMINI_MODEL},
        "schemas": {
            "adjudication": stable_hash(ADJUDICATION_SCHEMA),
            "verification": stable_hash(VERIFICATION_SCHEMA),
        },
        "attempts": {"opus": list(OPUS_ATTEMPTS), "gemini": list(GEMINI_ATTEMPTS)},
    }
    manifest_path = remediation / MANIFEST_FILE
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ProductionError("QA remediation manifest changed")
    else:
        atomic_json(manifest_path, manifest)
    return remediation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = run_dir(args.run_id)
    config = _load_run_config(base)
    with connect(args.db) as conn:
        prepared_base, units, verses = prepare_production(conn, config)
        english = {
            str(row["verse_key"]): str(row["translation"])
            for row in conn.execute(
                "SELECT verse_key, translation FROM translations WHERE run_id = ?",
                (args.run_id,),
            )
        }
        remediation = prepared_base / REMEDIATION_DIR
        targets_path = remediation / TARGETS_FILE
        if targets_path.exists():
            targets = json.loads(targets_path.read_text(encoding="utf-8"))["targets"]
        else:
            targets = build_targets(
                base=prepared_base,
                units=units,
                verses=verses,
                english=english,
            )
        remediation = _freeze_inputs(prepared_base, targets)

        complete_path = remediation / COMPLETE_FILE
        if complete_path.exists():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if not (prepared_base / "PRODUCTION_COMPLETE.json").exists():
                atomic_json(prepared_base / "PRODUCTION_COMPLETE.json", complete)
            print(json.dumps(complete, ensure_ascii=False, indent=2))
            return

        load_environment()
        prompt, ledger_md, ledger_json = prompt_material()
        opus_system = cached_system_blocks(
            QA_REMEDIATION_SYSTEM,
            prompt,
            ledger_md,
            ledger_json,
        )
        opus = AnthropicBatchClient(os.environ.get("ANTHROPIC_API_KEY", ""))
        adjudication = run_opus_adjudication(
            remediation=remediation,
            targets=targets,
            system=opus_system,
            client=opus,
            poll_seconds=config.poll_seconds,
        )
        candidates = _candidate_targets(targets, adjudication)

        critic_system = (
            f"{CRITIC_SYSTEM}\n\n"
            "For this remediation audit, use the supplied string `ref` rather than an "
            "integer ayah field. Return the ref-based registered schema exactly.\n\n"
            f"=== PROJECT TRANSLATION POLICY ===\n{prompt}\n\n"
            f"=== MODEL-FACING SENSE LEDGER ===\n{ledger_md}\n\n"
            f"=== STRUCTURED SENSE RECORDS ===\n{ledger_json}"
        )
        gemini = GeminiSynchronousClient(os.environ.get("GOOGLE_API_KEY", ""))
        verification = run_gemini_verification(
            remediation=remediation,
            targets=candidates,
            system=critic_system,
            client=gemini,
        )
        status = finalize_remediation(
            conn,
            base=prepared_base,
            units=units,
            verses=verses,
            targets=targets,
            candidates=candidates,
            verification=verification,
        )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
