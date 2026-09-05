"""Mechanically enumerate and resolve identical-ayah refrain consistency."""

from __future__ import annotations

from .production_clients import submit_batch_once

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import utc_now
from .production_clients import (
    AnthropicBatchClient,
    BatchState,
    ProviderError,
    anthropic_result_text,
)
from .production_packets import atomic_json
from .refrains import repeated_ayah_groups


REFRAIN_SYSTEM = """You are resolving consistency for Quran ayahs whose complete
Arabic text is byte-for-byte identical after encoding and whitespace normalization.
Choose one English rendering that faithfully works in every supplied occurrence.

Prefer an existing candidate if it preserves the Arabic in every context. Do not
merge semantic material from neighboring ayahs, add imagery, explain the line, or
vary the English by occurrence. Fidelity outranks elegance. The result will be
applied mechanically to every listed reference.

Return only this JSON object:
{"group_id": "sha256", "english": "one invariant rendering",
 "reason": "brief source-grounded reason"}
"""

REFRAIN_CONTRACT_ATTEMPTS = 2
REFRAIN_MAX_TOKENS = 12_000
REFRAIN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "group_id": {"type": "string"},
        "english": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["group_id", "english", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RefrainWork:
    group_id: str
    arabic: str
    refs: tuple[tuple[int, int], ...]
    renderings: tuple[str, ...]
    user: str


def _stable_hash(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _validate_resolution(document: Any, group_id: str) -> dict[str, str] | None:
    if not isinstance(document, dict) or set(document) != {"group_id", "english", "reason"}:
        return None
    if document.get("group_id") != group_id:
        return None
    english = document.get("english")
    reason = document.get("reason")
    if not isinstance(english, str) or not english.strip():
        return None
    if not isinstance(reason, str) or not reason.strip():
        return None
    return {"group_id": group_id, "english": english.strip(), "reason": reason.strip()}


def _context(
    refs: list[tuple[int, int]],
    verses: dict[tuple[int, int], str],
    translations: dict[tuple[int, int], str],
) -> str:
    lines: list[str] = []
    for surah, ayah in refs:
        lines.append(f"## Occurrence {surah}:{ayah}")
        if (surah, ayah - 1) in verses:
            lines.append(f"Previous Arabic: {verses[(surah, ayah - 1)]}")
        lines.append(f"Repeated Arabic: {verses[(surah, ayah)]}")
        lines.append(f"Current English: {translations[(surah, ayah)]}")
        if (surah, ayah + 1) in verses:
            lines.append(f"Next Arabic: {verses[(surah, ayah + 1)]}")
        lines.append("")
    return "\n".join(lines)


def build_refrain_work(
    *,
    verses: dict[tuple[int, int], str],
    translations: dict[tuple[int, int], str],
    canonical: dict[str, str],
) -> tuple[list[RefrainWork], dict[str, dict[str, Any]]]:
    work: list[RefrainWork] = []
    settled: dict[str, dict[str, Any]] = {}
    for group_id, group in sorted(repeated_ayah_groups(verses).items()):
        refs = [tuple(ref) for ref in group["refs"]]
        renderings = sorted({translations[ref] for ref in refs})
        if group_id in canonical:
            settled[group_id] = {
                "group_id": group_id,
                "english": canonical[group_id],
                "reason": "Frozen project refrain policy",
                "source": "policy",
                "refs": [list(ref) for ref in refs],
            }
            continue
        if len(renderings) == 1:
            settled[group_id] = {
                "group_id": group_id,
                "english": renderings[0],
                "reason": "All independently generated occurrences already agree",
                "source": "existing_invariant",
                "refs": [list(ref) for ref in refs],
            }
            continue
        user = (
            f"GROUP_ID: {group_id}\n"
            f"IDENTICAL ARABIC: {group['arabic']}\n\n"
            f"CURRENT CANDIDATE RENDERINGS:\n"
            + "\n".join(f"- {rendering}" for rendering in renderings)
            + "\n\nALL OCCURRENCES WITH LOCAL CONTEXT:\n"
            + _context(refs, verses, translations)
        )
        work.append(
            RefrainWork(
                group_id=group_id,
                arabic=str(group["arabic"]),
                refs=tuple(refs),
                renderings=tuple(renderings),
                user=user,
            )
        )
    return work, settled


def _wait(
    client: AnthropicBatchClient,
    state: BatchState,
    poll_seconds: int,
) -> BatchState:
    current = state
    while not current.ended:
        time.sleep(poll_seconds)
        current = client.retrieve(current.batch_id)
    return current


def resolve_refrains(
    *,
    base: Path,
    verses: dict[tuple[int, int], str],
    translations: dict[tuple[int, int], str],
    policy_path: Path,
    shared_system: list[dict[str, Any]],
    client: AnthropicBatchClient,
    poll_seconds: int,
) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    canonical = policy.get("canonical", policy)
    if not isinstance(canonical, dict):
        raise ValueError("Refrain policy canonical map is malformed")
    work, settled = build_refrain_work(
        verses=verses, translations=translations, canonical=canonical
    )
    work_by_id = {item.group_id: item for item in work}
    input_payload = [
        {
            "group_id": item.group_id,
            "arabic": item.arabic,
            "refs": [list(ref) for ref in item.refs],
            "renderings": list(item.renderings),
            "user_hash": _stable_hash(item.user),
        }
        for item in work
    ]
    input_hash = _stable_hash(input_payload)
    input_path = base / "refrains" / "INPUT.json"
    if input_path.exists():
        current = json.loads(input_path.read_text(encoding="utf-8"))
        if current != {"input_hash": input_hash, "groups": input_payload}:
            raise RuntimeError("Refrain input changed after resolution began")
    else:
        atomic_json(input_path, {"input_hash": input_hash, "groups": input_payload})

    for item in work:
        artifact = base / "refrains" / f"{item.group_id}.json"
        if not artifact.exists():
            continue
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("input_hash") != _stable_hash(asdict_refrain(item)):
            raise RuntimeError(f"Cached refrain input changed: {artifact}")
        if _validate_resolution(payload.get("result"), item.group_id) is None:
            raise RuntimeError(f"Cached refrain resolution is invalid: {artifact}")

    for attempt in range(1, REFRAIN_CONTRACT_ATTEMPTS + 1):
        attempt_work = (
            list(work)
            if attempt == 1
            else [
                item
                for item in work
                if (base / "refrains" / f"{item.group_id}-attempt{attempt - 1}-FAILED.json").exists()
                and not (base / "refrains" / f"{item.group_id}.json").exists()
            ]
        )
        if not attempt_work:
            break
        requests = [
            {
                "custom_id": f"refr-{item.group_id[:32]}",
                "params": {
                    "model": "claude-opus-4-6",
                    "max_tokens": REFRAIN_MAX_TOKENS,
                    "thinking": {"type": "adaptive"},
                    "output_config": {
                        "effort": "high",
                        "format": {
                            "type": "json_schema",
                            "schema": REFRAIN_JSON_SCHEMA,
                        },
                    },
                    "system": [
                        {"type": "text", "text": REFRAIN_SYSTEM},
                        *shared_system,
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": item.user
                            + (
                                "\n\nRETRY: The prior response failed the JSON contract. "
                                "Return only the exact requested object."
                                if attempt > 1
                                else ""
                            ),
                        }
                    ],
                },
            }
            for item in attempt_work
        ]
        request_hash = _stable_hash(requests)
        job_path = base / "jobs" / f"refrains-a{attempt}.json"
        if job_path.exists():
            job = json.loads(job_path.read_text(encoding="utf-8"))
            if job.get("request_hash") != request_hash:
                raise RuntimeError("Refrain provider request changed")
            state = client.retrieve(str(job["batch_id"]))
        else:
            state = submit_batch_once(client, requests, job_path)
            atomic_json(
                job_path,
                {
                    "provider": "anthropic",
                    "stage": "refrains",
                    "attempt": attempt,
                    "request_hash": request_hash,
                    "batch_id": state.batch_id,
                    "state": state.state,
                    "submitted_at": utc_now(),
                },
            )
        state = _wait(client, state, poll_seconds)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        job.update({"state": state.state, "last_polled_at": utc_now()})
        atomic_json(job_path, job)
        rows = {str(row.get("custom_id")): row for row in client.results(state.batch_id)}
        for item in attempt_work:
            artifact = base / "refrains" / f"{item.group_id}.json"
            if artifact.exists():
                continue
            row = rows.get(f"refr-{item.group_id[:32]}")
            if row is None:
                atomic_json(
                    base / "refrains" / f"{item.group_id}-attempt{attempt}-FAILED.json",
                    {
                        "attempt": attempt,
                        "error": "Provider result lacked this custom_id",
                    },
                )
                continue
            try:
                raw, usage = anthropic_result_text(row)
                resolution = _validate_resolution(_extract_json(raw), item.group_id)
            except (ProviderError, json.JSONDecodeError, TypeError, ValueError):
                resolution = None
                raw = json.dumps(row, ensure_ascii=False)
                usage = {}
            if resolution is None:
                atomic_json(
                    base / "refrains" / f"{item.group_id}-attempt{attempt}-FAILED.json",
                    {"attempt": attempt, "raw": raw, "usage": usage},
                )
                continue
            atomic_json(
                artifact,
                {
                    "input_hash": _stable_hash(asdict_refrain(item)),
                    "model": "claude-opus-4-6",
                    "attempt": attempt,
                    "usage": usage,
                    "result": resolution,
                    "raw": raw,
                },
            )

    pending = [
        item.group_id
        for item in work
        if not (base / "refrains" / f"{item.group_id}.json").exists()
    ]
    if pending:
        raise RuntimeError(
            "Refrain contract failed twice for: " + ", ".join(pending)
        )

    for group_id, item in work_by_id.items():
        artifact = base / "refrains" / f"{group_id}.json"
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        resolution = _validate_resolution(payload.get("result"), group_id)
        if resolution is None:
            raise RuntimeError(f"Cached refrain resolution is invalid: {artifact}")
        settled[group_id] = {
            **resolution,
            "source": "opus_resolution",
            "refs": [list(ref) for ref in item.refs],
        }

    overrides: dict[str, str] = {}
    for record in settled.values():
        for surah, ayah in record["refs"]:
            overrides[f"{surah}:{ayah}"] = record["english"]
    report = {
        "version": "production-refrains-v1",
        "groups_total": len(settled),
        "groups_model_resolved": len(work),
        "overrides": overrides,
        "groups": settled,
    }
    atomic_json(base / "REFRAINS.json", report)
    return report


def asdict_refrain(item: RefrainWork) -> dict[str, Any]:
    return {
        "group_id": item.group_id,
        "arabic": item.arabic,
        "refs": item.refs,
        "renderings": item.renderings,
        "user": item.user,
    }
