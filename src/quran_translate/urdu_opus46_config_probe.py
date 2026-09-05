"""Bounded production-configuration probe for Urdu translation.

This is intentionally separate from the frozen model bakeoff. It compares a
medium-effort Opus 4.6 direct draft with a medium-effort Opus 4.6 revision of
the frozen Muse Spark 1.2 draft for Quran 2:177-187.
"""

from __future__ import annotations

from .production_clients import submit_batch_once

import argparse
import html
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from . import urdu_translation_bakeoff as base
from .config import OUTPUT_DIR
from .production_clients import AnthropicBatchClient, ProviderError, anthropic_result_text
from .production_packets import atomic_json, atomic_text


PROBE_ID = "quran-urdu-opus46-medium-taqwa-probe-20260818-v1"
DEFAULT_ROOT = OUTPUT_DIR / "urdu" / "bakeoffs" / PROBE_ID
SOURCE_ROOT = (
    OUTPUT_DIR
    / "urdu"
    / "bakeoffs"
    / "quran-urdu-translation-models-20260818-v2"
)
PASSAGE_ID = "fasting"
MODEL_ID = "claude-opus-4-6"
MAX_OUTPUT_TOKENS = 16_000
EFFORT = "medium"
POLL_SECONDS = 20
TREATMENTS = {
    "opus46-medium-direct": "Direct medium-effort Opus 4.6",
    "muse-opus46-medium-revision": "Muse Spark 1.2 draft revised by medium-effort Opus 4.6",
}

# Anthropic Batch API prices, USD per million tokens. The system prompt uses
# a one-hour cache entry, hence the 5.00 cache-write rate after the 50% batch
# discount.
BATCH_RATES = {
    "input_tokens": 2.50,
    "cache_creation_input_tokens": 5.00,
    "cache_read_input_tokens": 0.25,
    "output_tokens": 12.50,
}


class UrduProbeError(RuntimeError):
    """The bounded probe cannot proceed without violating its frozen inputs."""


def _source_input_path() -> Path:
    return SOURCE_ROOT / "inputs" / f"{PASSAGE_ID}.json"


def _source_muse_path() -> Path:
    return (
        SOURCE_ROOT
        / "private"
        / "translations"
        / "openrouter-muse-spark-12"
        / f"{PASSAGE_ID}.json"
    )


def _load_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(_source_input_path().read_text(encoding="utf-8"))
    muse_document = json.loads(_source_muse_path().read_text(encoding="utf-8"))
    passage = next(item for item in base.PASSAGES if item.passage_id == PASSAGE_ID)
    muse_result = base.validate_translation(
        muse_document.get("result"), passage.expected_ayahs
    )
    if muse_document.get("status") != "complete" or muse_result is None:
        raise UrduProbeError("Frozen Muse fasting draft is missing or invalid")
    return payload, muse_result


def revision_user(payload: dict[str, Any], muse_result: dict[str, Any]) -> str:
    muse_json = json.dumps(muse_result, ensure_ascii=False, indent=2)
    return (
        base._translation_user(payload)
        + "\n\n=== FROZEN MUSE SPARK BASE DRAFT ===\n"
        + muse_json
        + "\n\n=== CONSTRAINED REVISION TASK ===\n"
        "Treat the base draft as a semantic skeleton, not as authority over the Arabic. "
        "Preserve every supported sense, live ambiguity, participant, legal scope, image, "
        "and refrain. Improve only wording that is stiff, breathless, unclear aloud, or "
        "contrary to the supplied ledger. Repair a genuine fidelity defect when the Arabic "
        "or supplied evidence requires it. Do not add brackets, parenthetical glosses, "
        "tafsir, imagery, intensity, honorifics, or explanatory material. Keep repeated "
        "taqwa expressions contextually principled and internally consistent. Return one "
        "complete Urdu row for every target ayah under the registered JSON contract."
    )


def treatment_requests(
    payload: dict[str, Any], muse_result: dict[str, Any]
) -> list[dict[str, Any]]:
    system = [
        {
            "type": "text",
            "text": base._translation_system(),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    users = {
        "opus46-medium-direct": base._translation_user(payload),
        "muse-opus46-medium-revision": revision_user(payload, muse_result),
    }
    return [
        {
            "custom_id": treatment_id,
            "params": {
                "model": MODEL_ID,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "thinking": {"type": "adaptive"},
                "output_config": {
                    "effort": EFFORT,
                    "format": {
                        "type": "json_schema",
                        "schema": base.TRANSLATION_SCHEMA,
                    },
                },
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        }
        for treatment_id, user in users.items()
    ]


def estimate_batch_cost(usage: dict[str, Any]) -> float:
    return sum(
        float(usage.get(field, 0) or 0) * rate / 1_000_000
        for field, rate in BATCH_RATES.items()
    )


def _manifest() -> dict[str, Any]:
    payload, muse_result = _load_sources()
    requests = treatment_requests(payload, muse_result)
    return {
        "version": "urdu-opus46-config-probe-v1",
        "probe_id": PROBE_ID,
        "scope": "Quran 2:177-187 only; separate from the frozen model bakeoff",
        "source_bakeoff": str(SOURCE_ROOT),
        "source_input_sha256": base.file_hash(_source_input_path()),
        "source_muse_sha256": base.file_hash(_source_muse_path()),
        "model": MODEL_ID,
        "effort": EFFORT,
        "max_output_tokens_per_treatment": MAX_OUTPUT_TOKENS,
        "logical_attempts_per_treatment": 1,
        "transport": "Anthropic Message Batches",
        "treatments": list(TREATMENTS),
        "request_hash": base.stable_hash(requests),
        "code_sha256": base.file_hash(Path(__file__)),
    }


def prepare(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "private").mkdir(parents=True, exist_ok=True)
    (root / "blind").mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    path = root / "MANIFEST.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise UrduProbeError("Probe manifest or implementation changed")
    if not path.exists():
        atomic_json(path, manifest)
    return status(root, verify_manifest=False)


def _assert_manifest(root: Path) -> None:
    path = root / "MANIFEST.json"
    if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != _manifest():
        raise UrduProbeError("Probe manifest or frozen inputs changed")


def _artifact_path(root: Path, treatment_id: str) -> Path:
    return root / "private" / f"{treatment_id}.json"


def _save_result(root: Path, row: dict[str, Any]) -> None:
    treatment_id = str(row.get("custom_id") or "")
    if treatment_id not in TREATMENTS:
        raise UrduProbeError(f"Unexpected Anthropic result ID: {treatment_id}")
    path = _artifact_path(root, treatment_id)
    if path.exists():
        return
    passage = next(item for item in base.PASSAGES if item.passage_id == PASSAGE_ID)
    try:
        text, metadata = anthropic_result_text(row)
        parsed = base.extract_json(text)
        result = base.validate_translation(parsed, passage.expected_ayahs)
        if result is None:
            raise UrduProbeError("Response failed the exact Urdu translation contract")
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
        document = {
            "version": "urdu-opus46-config-probe-result-v1",
            "status": "complete",
            "treatment_id": treatment_id,
            "model": MODEL_ID,
            "effort": EFFORT,
            "usage": usage,
            "batch_cost_usd": round(estimate_batch_cost(usage), 6),
            "result": result,
            "raw_row": row,
        }
    except Exception as exc:
        document = {
            "version": "urdu-opus46-config-probe-result-v1",
            "status": "failed",
            "treatment_id": treatment_id,
            "model": MODEL_ID,
            "effort": EFFORT,
            "error": f"{type(exc).__name__}: {exc}"[:3000],
            "raw_row": row,
        }
    atomic_json(path, document)


def run(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    prepare(root)
    _assert_manifest(root)
    existing = [_artifact_path(root, item) for item in TREATMENTS]
    if all(path.exists() for path in existing):
        if all(
            json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
            for path in existing
        ):
            package(root)
        state = status(root)
        atomic_json(root / "RUN.json", state)
        return state

    base.load_environment()
    client = AnthropicBatchClient(os.environ.get("ANTHROPIC_API_KEY", ""))
    payload, muse_result = _load_sources()
    requests = treatment_requests(payload, muse_result)
    job_path = root / "BATCH_JOB.json"
    request_hash = base.stable_hash(requests)
    if job_path.exists():
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job.get("request_hash") != request_hash:
            raise UrduProbeError("Existing batch job does not match the frozen requests")
        state = client.retrieve(str(job["batch_id"]))
    else:
        state = submit_batch_once(client, requests, job_path)
        atomic_json(
            job_path,
            {
                "provider": "anthropic",
                "batch_id": state.batch_id,
                "state": state.state,
                "request_hash": request_hash,
                "custom_ids": list(TREATMENTS),
            },
        )

    while not state.ended:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        job["state"] = state.state
        atomic_json(job_path, job)
        print(f"batch {state.batch_id}: {state.state}", flush=True)
        time.sleep(POLL_SECONDS)
        state = client.retrieve(state.batch_id)

    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["state"] = state.state
    atomic_json(job_path, job)
    if not state.succeeded:
        raise ProviderError(f"Anthropic probe batch ended in {state.state}")
    rows = client.results(state.batch_id)
    seen = {str(row.get("custom_id") or "") for row in rows}
    if seen != set(TREATMENTS):
        raise UrduProbeError(f"Batch result IDs differ from treatments: {sorted(seen)}")
    for row in rows:
        _save_result(root, row)

    if all(
        json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
        for path in existing
    ):
        package(root)
    state_out = status(root)
    atomic_json(root / "RUN.json", state_out)
    return state_out


def _blind_key(root: Path) -> dict[str, str]:
    path = root / "PRIVATE_BLIND_KEY.json"
    identities = ["muse-baseline", *TREATMENTS]
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        mapping = document.get("mapping")
        if not isinstance(mapping, dict) or set(mapping) != set(identities):
            raise UrduProbeError("Probe blind key has the wrong roster")
        return {str(key): str(value) for key, value in mapping.items()}
    shuffled = list(identities)
    secrets.SystemRandom().shuffle(shuffled)
    mapping = {
        identity: f"Candidate {chr(ord('A') + index)}"
        for index, identity in enumerate(shuffled)
    }
    atomic_json(
        path,
        {"version": "urdu-opus46-config-probe-key-v1", "mapping": mapping},
    )
    path.chmod(0o600)
    return mapping


def _review_html(workbook: dict[str, Any]) -> str:
    sections: list[str] = []
    for candidate in workbook["candidates"]:
        rows = "".join(
            "<article><b>2:{ayah}</b><p lang='ur' dir='rtl'>{urdu}</p></article>".format(
                ayah=int(row["ayah"]), urdu=html.escape(str(row["urdu"]))
            )
            for row in candidate["ayahs"]
        )
        sections.append(
            f"<section><h2>{html.escape(candidate['code'])}</h2>{rows}</section>"
        )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Urdu taqwa configuration probe</title>
<style>
body{margin:0;background:#f5f7f4;color:#172019;font:16px/1.55 system-ui,sans-serif}header{position:sticky;top:0;background:#172019;color:white;padding:16px 5vw;display:flex;justify-content:space-between;align-items:center}main{max-width:1100px;margin:auto;padding:28px 5vw}section{border-top:3px solid #24704a;margin:34px 0;padding-top:12px}article{display:grid;grid-template-columns:70px 1fr;gap:18px;border-top:1px solid #cfd8d1;padding:12px 0}p{font:24px/1.9 "Noto Nastaliq Urdu","Noto Naskh Arabic",serif;margin:0;text-align:right}button{background:#f5c85c;border:0;border-radius:6px;padding:10px 14px;font-weight:700}@media(max-width:600px){article{grid-template-columns:1fr}p{font-size:21px}}
</style></head><body><header><strong>Quran 2:177-187 · blinded configuration probe</strong><button id="copy">Copy all text</button></header><main>
<p>Compare fidelity, taqwa consistency, legal clarity, spoken cadence, and unsupported additions. Model identities remain sealed.</p>
""" + "".join(sections) + """</main><script>
document.getElementById('copy').onclick=async()=>{const t=document.querySelector('main').innerText;try{await navigator.clipboard.writeText(t)}catch(e){const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();document.execCommand('copy');a.remove()}document.getElementById('copy').textContent='Copied'};
</script></body></html>"""


def package(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    _assert_manifest(root)
    payload, muse_result = _load_sources()
    results = {"muse-baseline": muse_result}
    for treatment_id in TREATMENTS:
        path = _artifact_path(root, treatment_id)
        if not path.is_file():
            raise UrduProbeError(f"Missing treatment result: {treatment_id}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("status") != "complete":
            raise UrduProbeError(f"Treatment did not complete: {treatment_id}")
        results[treatment_id] = document["result"]
    mapping = _blind_key(root)
    candidates = [
        {"code": mapping[identity], "ayahs": result["ayahs"]}
        for identity, result in results.items()
    ]
    candidates.sort(key=lambda item: item["code"])
    workbook = {
        "version": "urdu-opus46-config-probe-workbook-v1",
        "probe_id": PROBE_ID,
        "ref": "2:177-187",
        "arabic": payload["target"],
        "candidates": candidates,
    }
    blind = root / "blind"
    atomic_json(blind / "BLIND_MANIFEST.json", workbook)
    page = _review_html(workbook)
    private_terms = [MODEL_ID, "Muse", "Opus", *TREATMENTS]
    if any(term in page for term in private_terms):
        raise UrduProbeError("Private treatment identity leaked into review page")
    atomic_text(blind / "review.html", page)
    return workbook


def status(root: Path = DEFAULT_ROOT, *, verify_manifest: bool = True) -> dict[str, Any]:
    if verify_manifest:
        _assert_manifest(root)
    complete = 0
    failed = 0
    costs: dict[str, float] = {}
    for treatment_id in TREATMENTS:
        path = _artifact_path(root, treatment_id)
        if not path.exists():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("status") == "complete":
            complete += 1
            costs[treatment_id] = float(document.get("batch_cost_usd", 0) or 0)
        else:
            failed += 1
    if failed:
        state = "blocked"
    elif complete == len(TREATMENTS) and (root / "blind" / "review.html").is_file():
        state = "ready_for_blind_review"
    elif (root / "BATCH_JOB.json").is_file():
        state = "running_or_awaiting_results"
    else:
        state = "prepared"
    return {
        "version": "urdu-opus46-config-probe-status-v1",
        "probe_id": PROBE_ID,
        "status": state,
        "complete": complete,
        "failed": failed,
        "pending": len(TREATMENTS) - complete - failed,
        "costs_usd": costs,
        "total_cost_usd": round(sum(costs.values()), 6),
        "review": str(root / "blind" / "review.html"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "package"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root)
    elif args.command == "run":
        result = run(args.root)
    elif args.command == "package":
        result = package(args.root)
    else:
        result = status(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
