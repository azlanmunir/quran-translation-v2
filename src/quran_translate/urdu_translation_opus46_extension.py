"""Blinded Opus 4.6 extension to the completed Urdu translation bakeoff."""

from __future__ import annotations

import argparse
import json
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import urdu_translation_bakeoff as base
from .config import OUTPUT_DIR
from .metadata import surah_info
from .production_packets import atomic_json, atomic_text


EXTENSION_ID = "quran-urdu-translation-opus46-finalists-20260818-v1"
ORIGINAL_MANIFEST_SHA256 = (
    "f47d017240e37d12195d72745ddf73f478c04bacb5c97ec368f1136c44d8a378"
)
RECOVERY_MANIFEST_SHA256 = (
    "6c69d9f80d6b395afcfc54a296ddffa9f2a76bd35c9acc4fc7705b928fde90ba"
)
OPERATIONAL_FORFEIT_SHA256 = (
    "1b3fd3f2d5cdabdc99ff0fea318bce2688a6914b89b91304e12bcee2a76f645c"
)
OPUS_46_RECOVERY_MAX_OUTPUT_TOKENS = 32_000
FORFEITED_PASSAGE_IDS = {"fasting"}
DEFAULT_ROOT = OUTPUT_DIR / "urdu" / "bakeoffs" / EXTENSION_ID
SOURCE_ROOT = (
    OUTPUT_DIR
    / "urdu"
    / "bakeoffs"
    / "quran-urdu-translation-models-20260818-v2"
)

OPUS_46 = base.ModelSpec(
    "anthropic-opus-4-6",
    "anthropic",
    "claude-opus-4-6",
    "high",
    "Claude Opus 4.6",
)
FINALISTS = (
    OPUS_46,
    next(item for item in base.CANDIDATES if item.candidate_id == "anthropic-opus-4-8"),
    next(item for item in base.CANDIDATES if item.candidate_id == "anthropic-fable-5"),
    next(item for item in base.CANDIDATES if item.candidate_id == "openrouter-muse-spark-12"),
)
SEEDED_FINALISTS = tuple(item for item in FINALISTS if item != OPUS_46)


def _file_hash(path: Path) -> str:
    return base.file_hash(path)


def _result_path(root: Path, candidate: base.ModelSpec, passage: base.PassageSpec) -> Path:
    return base._private_result_path(root, candidate, passage)


def _translation_input_hash(
    candidate: base.ModelSpec,
    payload: dict[str, Any],
    *,
    max_output_tokens: int,
) -> str:
    return base.stable_hash(
        {
            "candidate": asdict(candidate),
            "policy": base.file_hash(base.POLICY_PATH),
            "ledger_md": base.file_hash(base.LEDGER_MD_PATH),
            "ledger_json": base.file_hash(base.LEDGER_JSON_PATH),
            "payload": payload,
            "schema": base.TRANSLATION_SCHEMA,
            "max_output_tokens": max_output_tokens,
            "contract_attempts": base.CONTRACT_ATTEMPTS,
        }
    )


def _load_opus46_complete(
    path: Path,
    payload: dict[str, Any],
    expected_ayahs: list[int],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    accepted_hashes = {
        _translation_input_hash(
            OPUS_46,
            payload,
            max_output_tokens=max_output_tokens,
        )
        for max_output_tokens in (
            base.MAX_OUTPUT_TOKENS,
            OPUS_46_RECOVERY_MAX_OUTPUT_TOKENS,
        )
    }
    if document.get("input_hash") not in accepted_hashes:
        raise base.UrduBakeoffError(f"Cached Opus 4.6 input mismatch: {path}")
    if document.get("status") != "complete":
        return None
    result = base.validate_translation(document.get("result"), expected_ayahs)
    if result is None:
        raise base.UrduBakeoffError(f"Cached Opus 4.6 output fails contract: {path}")
    return document


def _blind_key(root: Path) -> dict[str, str]:
    path = root / "PRIVATE_BLIND_KEY.json"
    candidate_ids = [item.candidate_id for item in FINALISTS]
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        mapping = document.get("mapping")
        if not isinstance(mapping, dict) or set(mapping) != set(candidate_ids):
            raise base.UrduBakeoffError("Existing finalist blind key has the wrong roster")
        if len(set(mapping.values())) != len(candidate_ids):
            raise base.UrduBakeoffError("Existing finalist blind key has duplicate codes")
        path.chmod(0o600)
        return {str(key): str(value) for key, value in mapping.items()}

    shuffled = list(candidate_ids)
    secrets.SystemRandom().shuffle(shuffled)
    mapping = {
        candidate_id: f"Candidate {chr(ord('A') + index)}"
        for index, candidate_id in enumerate(shuffled)
    }
    atomic_json(
        path,
        {
            "version": "urdu-translation-finalist-blind-key-v1",
            "bakeoff_id": EXTENSION_ID,
            "mapping": mapping,
        },
    )
    path.chmod(0o600)
    return mapping


def _eligible_passages(root: Path) -> tuple[base.PassageSpec, ...]:
    marker = root / "OPERATIONAL_FORFEIT.json"
    if not marker.is_file() or _file_hash(marker) != OPERATIONAL_FORFEIT_SHA256:
        raise base.UrduBakeoffError("Frozen Opus 4.6 operational-forfeit marker is missing or changed")
    return tuple(
        passage for passage in base.PASSAGES if passage.passage_id not in FORFEITED_PASSAGE_IDS
    )


def _extension_manifest(root: Path) -> dict[str, Any]:
    inputs = {
        passage.passage_id: _file_hash(root / "inputs" / f"{passage.passage_id}.json")
        for passage in base.PASSAGES
    }
    seeded = {
        f"{candidate.candidate_id}/{passage.passage_id}": _file_hash(
            _result_path(root, candidate, passage)
        )
        for candidate in SEEDED_FINALISTS
        for passage in base.PASSAGES
    }
    return {
        "version": "urdu-translation-opus46-extension-v1",
        "bakeoff_id": EXTENSION_ID,
        "source_bakeoff": str(SOURCE_ROOT),
        "source_manifest_sha256": _file_hash(SOURCE_ROOT / "MANIFEST.json"),
        "candidates": [asdict(item) for item in FINALISTS],
        "auditors": [asdict(item) for item in base.AUDITORS],
        "passages": [asdict(item) for item in base.PASSAGES],
        "inputs": inputs,
        "seeded_results": seeded,
        "generation": {
            "max_output_tokens": base.MAX_OUTPUT_TOKENS,
            "contract_attempts": base.CONTRACT_ATTEMPTS,
            "opus_4_6_recovery_max_output_tokens": OPUS_46_RECOVERY_MAX_OUTPUT_TOKENS,
            "recovery_scope": "Only 16k outputs that exhausted both strict-contract attempts",
        },
        "amendment": {
            "version": "v1.2",
            "reason": "Bounded recovery completed two passages; fasting exceeded 30 minutes",
            "previous_manifest_sha256": RECOVERY_MANIFEST_SHA256,
            "operational_forfeit_sha256": OPERATIONAL_FORFEIT_SHA256,
            "forfeited_passage_ids": sorted(FORFEITED_PASSAGE_IDS),
        },
        "code": {
            "base_harness": _file_hash(Path(base.__file__)),
            "extension_harness": _file_hash(Path(__file__)),
        },
    }


def prepare(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    if not (SOURCE_ROOT / "MANIFEST.json").is_file():
        raise base.UrduBakeoffError(f"Source bakeoff is missing: {SOURCE_ROOT}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(parents=True, exist_ok=True)

    for passage in base.PASSAGES:
        source_input = SOURCE_ROOT / "inputs" / f"{passage.passage_id}.json"
        destination_input = root / "inputs" / source_input.name
        source_bytes = source_input.read_bytes()
        if destination_input.exists() and destination_input.read_bytes() != source_bytes:
            raise base.UrduBakeoffError(f"Finalist input differs from source: {destination_input}")
        if not destination_input.exists():
            atomic_text(destination_input, source_bytes.decode("utf-8"))

        payload = json.loads(destination_input.read_text(encoding="utf-8"))
        for candidate in SEEDED_FINALISTS:
            input_hash = base._translation_input_hash(candidate, payload)
            source_result = _result_path(SOURCE_ROOT, candidate, passage)
            document = base._load_complete_translation(
                source_result, input_hash, passage.expected_ayahs
            )
            if document is None:
                raise base.UrduBakeoffError(
                    f"Source finalist is incomplete: {candidate.candidate_id}/{passage.passage_id}"
                )
            destination_result = _result_path(root, candidate, passage)
            if destination_result.exists():
                if destination_result.read_bytes() != source_result.read_bytes():
                    raise base.UrduBakeoffError(
                        f"Seeded finalist differs from source: {destination_result}"
                    )
            else:
                atomic_text(destination_result, source_result.read_text(encoding="utf-8"))

    manifest = _extension_manifest(root)
    manifest_path = root / "MANIFEST.json"
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != manifest and _file_hash(manifest_path) == ORIGINAL_MANIFEST_SHA256:
            archived = root / "MANIFEST.v1.json"
            if not archived.is_file() or _file_hash(archived) != ORIGINAL_MANIFEST_SHA256:
                raise base.UrduBakeoffError("Original finalist manifest was not preserved")
            atomic_json(
                root / "AMENDMENT_v1.1.json",
                {
                    "version": "urdu-translation-opus46-extension-amendment-v1.1",
                    "reason": (
                        "Three dense Opus 4.6 passages exhausted 16k output tokens on "
                        "adaptive thinking before returning contract-complete JSON."
                    ),
                    "change": (
                        "Retry only those exhausted passages once under the existing two-attempt "
                        "contract with a 32k output ceiling; preserve every 16k failure artifact."
                    ),
                    "previous_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
                },
            )
            atomic_json(manifest_path, manifest)
        elif current != manifest and _file_hash(manifest_path) == RECOVERY_MANIFEST_SHA256:
            archived = root / "MANIFEST.v1.1.json"
            marker = root / "OPERATIONAL_FORFEIT.json"
            if not archived.is_file() or _file_hash(archived) != RECOVERY_MANIFEST_SHA256:
                raise base.UrduBakeoffError("Recovery manifest was not preserved")
            if not marker.is_file() or _file_hash(marker) != OPERATIONAL_FORFEIT_SHA256:
                raise base.UrduBakeoffError("Operational-forfeit marker is missing or changed")
            atomic_json(
                root / "AMENDMENT_v1.2.json",
                {
                    "version": "urdu-translation-opus46-extension-amendment-v1.2",
                    "reason": (
                        "The fasting passage produced no contract-complete response before the "
                        "bounded 30-minute recovery ceiling."
                    ),
                    "change": (
                        "Record fasting as an operational forfeit; blind and audit the ten "
                        "passages completed by all four finalists."
                    ),
                    "previous_manifest_sha256": RECOVERY_MANIFEST_SHA256,
                    "operational_forfeit_sha256": OPERATIONAL_FORFEIT_SHA256,
                },
            )
            atomic_json(manifest_path, manifest)
        elif current != manifest:
            raise base.UrduBakeoffError(
                "Finalist manifest changed; refusing a mixed-version resume"
            )
    else:
        atomic_json(manifest_path, manifest)
    _blind_key(root)
    state = status(root, verify_manifest=False)
    atomic_json(root / "RUN.json", state)
    return state


def _assert_manifest(root: Path) -> None:
    path = root / "MANIFEST.json"
    if not path.is_file():
        raise base.UrduBakeoffError("Finalist extension is not prepared")
    if json.loads(path.read_text(encoding="utf-8")) != _extension_manifest(root):
        raise base.UrduBakeoffError("Finalist manifest or frozen inputs changed")


def package(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    _assert_manifest(root)
    mapping = _blind_key(root)
    passages_out: list[dict[str, Any]] = []
    for passage in _eligible_passages(root):
        payload = json.loads(
            (root / "inputs" / f"{passage.passage_id}.json").read_text(encoding="utf-8")
        )
        candidates_out: list[dict[str, Any]] = []
        for candidate in FINALISTS:
            path = _result_path(root, candidate, passage)
            if candidate == OPUS_46:
                document = _load_opus46_complete(path, payload, passage.expected_ayahs)
            else:
                input_hash = base._translation_input_hash(candidate, payload)
                document = base._load_complete_translation(
                    path,
                    input_hash,
                    passage.expected_ayahs,
                )
            if document is None:
                raise base.UrduBakeoffError(
                    f"Cannot blind incomplete finalist: {candidate.candidate_id}/{passage.passage_id}"
                )
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
    workbook = {
        "version": "urdu-translation-finalist-workbook-v1",
        "bakeoff_id": EXTENSION_ID,
        "candidate_codes": sorted(mapping.values()),
        "operational_forfeits": [
            {
                "model_blind_code": mapping[OPUS_46.candidate_id],
                "passage_id": "fasting",
                "reason": "No contract-complete output within the bounded production envelope.",
            }
        ],
        "passages": passages_out,
    }
    blind = root / "blind"
    atomic_json(blind / "BLIND_MANIFEST.json", workbook)
    html = base._workbook_html(workbook)
    old_key = "const KEY='urdu-bakeoff-v1-scores';"
    if old_key not in html:
        raise base.UrduBakeoffError("Could not isolate finalist workbook local storage")
    html = html.replace(
        old_key,
        "const KEY='urdu-opus46-finalists-v1-scores';",
        1,
    )
    atomic_text(blind / "review.html", html)

    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in blind.rglob("*") if path.is_file()
    )
    private_terms = [
        value
        for candidate in FINALISTS
        for value in (candidate.candidate_id, candidate.model_id, candidate.private_label)
    ]
    leaked = [term for term in private_terms if term and term in serialized]
    if leaked:
        raise base.UrduBakeoffError(f"Private identity leaked into finalist workbook: {leaked}")
    return workbook


def run_generation(root: Path = DEFAULT_ROOT, *, workers: int = 4) -> dict[str, Any]:
    prepare(root)
    _assert_manifest(root)
    base.load_environment()
    standard_jobs: list[tuple[base.PassageSpec, dict[str, Any]]] = []
    recovery_jobs: list[tuple[base.PassageSpec, dict[str, Any]]] = []
    for passage in base.PASSAGES:
        if passage.passage_id in FORFEITED_PASSAGE_IDS:
            continue
        payload = json.loads(
            (root / "inputs" / f"{passage.passage_id}.json").read_text(encoding="utf-8")
        )
        path = _result_path(root, OPUS_46, passage)
        if _load_opus46_complete(path, payload, passage.expected_ayahs) is not None:
            continue
        if not path.exists():
            standard_jobs.append((passage, payload))
            continue
        failed = json.loads(path.read_text(encoding="utf-8"))
        failed_hash = failed.get("input_hash")
        standard_hash = _translation_input_hash(
            OPUS_46,
            payload,
            max_output_tokens=base.MAX_OUTPUT_TOKENS,
        )
        recovery_hash = _translation_input_hash(
            OPUS_46,
            payload,
            max_output_tokens=OPUS_46_RECOVERY_MAX_OUTPUT_TOKENS,
        )
        if failed_hash == recovery_hash and int(failed.get("attempts", 0)) >= base.CONTRACT_ATTEMPTS:
            continue
        if failed_hash != standard_hash or int(failed.get("attempts", 0)) < base.CONTRACT_ATTEMPTS:
            raise base.UrduBakeoffError(f"Unexpected Opus 4.6 failure state: {path}")
        archive = (
            root
            / "private"
            / "recovery"
            / "opus-4-6-16k-failures"
            / f"{passage.passage_id}.json"
        )
        if archive.exists() and archive.read_bytes() != path.read_bytes():
            raise base.UrduBakeoffError(f"Preserved 16k failure differs: {archive}")
        if not archive.exists():
            atomic_text(archive, path.read_text(encoding="utf-8"))
        path.unlink()
        recovery_jobs.append((passage, payload))

    def execute_jobs(
        jobs: list[tuple[base.PassageSpec, dict[str, Any]]],
        *,
        max_output_tokens: int,
    ) -> None:
        if not jobs:
            return
        previous_max = base.MAX_OUTPUT_TOKENS
        base.MAX_OUTPUT_TOKENS = max_output_tokens
        try:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {
                    executor.submit(
                        base._generate_translation_job, root, OPUS_46, passage, payload
                    ): passage
                    for passage, payload in jobs
                }
                for future in as_completed(futures):
                    passage = futures[future]
                    outcome = future.result()
                    print(
                        f"translation {outcome['status']}: opus-4-6 {passage.passage_id} "
                        f"({max_output_tokens // 1000}k)",
                        flush=True,
                    )
        finally:
            base.MAX_OUTPUT_TOKENS = previous_max

    execute_jobs(standard_jobs, max_output_tokens=base.MAX_OUTPUT_TOKENS)
    execute_jobs(
        recovery_jobs,
        max_output_tokens=OPUS_46_RECOVERY_MAX_OUTPUT_TOKENS,
    )
    state = status(root)
    if (
        state["generation"]["complete"] + state["generation"]["forfeited"]
        == state["generation"]["total"]
        and not state["generation"]["failed"]
    ):
        package(root)
        state = status(root)
    atomic_json(root / "RUN.json", state)
    return state


def run_audits(root: Path = DEFAULT_ROOT, *, workers: int = 2) -> dict[str, Any]:
    _assert_manifest(root)
    if not (root / "blind" / "BLIND_MANIFEST.json").is_file():
        package(root)
    base.load_environment()
    workbook = json.loads(
        (root / "blind" / "BLIND_MANIFEST.json").read_text(encoding="utf-8")
    )
    passage_by_id = {item["passage_id"]: item for item in workbook["passages"]}
    jobs: list[tuple[base.ModelSpec, base.PassageSpec, dict[str, Any], dict[str, Any]]] = []
    eligible_passages = _eligible_passages(root)
    for passage in eligible_passages:
        payload = json.loads(
            (root / "inputs" / f"{passage.passage_id}.json").read_text(encoding="utf-8")
        )
        entry = passage_by_id[passage.passage_id]
        for auditor in base.AUDITORS:
            path = base._audit_result_path(root, auditor, passage)
            input_hash = base._audit_input_hash(auditor, payload, entry)
            if base._load_complete_audit(
                path,
                input_hash=input_hash,
                payload=payload,
                passage_entry=entry,
            ) is None:
                if path.exists() and int(json.loads(path.read_text(encoding="utf-8")).get("attempts", 0)) >= base.CONTRACT_ATTEMPTS:
                    continue
                jobs.append((auditor, passage, payload, entry))
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    base._generate_audit_job, root, auditor, passage, payload, entry
                ): (auditor, passage)
                for auditor, passage, payload, entry in jobs
            }
            for future in as_completed(futures):
                auditor, passage = futures[future]
                outcome = future.result()
                print(
                    f"audit {outcome['status']}: {auditor.candidate_id} {passage.passage_id}",
                    flush=True,
                )
    state = status(root)
    if state["audits"]["complete"] == state["audits"]["total"] and not state["audits"]["failed"]:
        previous_id = base.BAKEOFF_ID
        try:
            base.BAKEOFF_ID = EXTENSION_ID
            previous_passages = base.PASSAGES
            base.PASSAGES = eligible_passages
            try:
                base.build_audit_summary(root)
            finally:
                base.PASSAGES = previous_passages
        finally:
            base.BAKEOFF_ID = previous_id
        state = status(root)
    atomic_json(root / "RUN.json", state)
    return state


def status(root: Path = DEFAULT_ROOT, *, verify_manifest: bool = True) -> dict[str, Any]:
    if verify_manifest:
        _assert_manifest(root)
    generation = {
        "complete": 0,
        "failed": 0,
        "forfeited": 0,
        "pending": 0,
        "total": len(FINALISTS) * len(base.PASSAGES),
    }
    for passage in base.PASSAGES:
        for candidate in FINALISTS:
            path = _result_path(root, candidate, passage)
            if not path.exists():
                if candidate == OPUS_46 and passage.passage_id in FORFEITED_PASSAGE_IDS:
                    generation["forfeited"] += 1
                    continue
                generation["pending"] += 1
                continue
            value = json.loads(path.read_text(encoding="utf-8")).get("status")
            generation["complete" if value == "complete" else "failed"] += 1
    eligible_passages = _eligible_passages(root)
    audits = {
        "complete": 0,
        "failed": 0,
        "pending": 0,
        "total": len(base.AUDITORS) * len(eligible_passages),
    }
    for passage in eligible_passages:
        for auditor in base.AUDITORS:
            path = base._audit_result_path(root, auditor, passage)
            if not path.exists():
                audits["pending"] += 1
                continue
            value = json.loads(path.read_text(encoding="utf-8")).get("status")
            audits["complete" if value == "complete" else "failed"] += 1
    if generation["failed"] or audits["failed"]:
        state = "blocked"
    elif audits["complete"] == audits["total"] and (root / "blind" / "AUDIT_SUMMARY.json").is_file():
        state = "ready_for_blind_review"
    elif generation["complete"] + generation["forfeited"] == generation["total"]:
        state = "generation_complete"
    else:
        state = "prepared"
    return {
        "version": "urdu-translation-opus46-extension-status-v1",
        "bakeoff_id": EXTENSION_ID,
        "status": state,
        "generation": generation,
        "audits": audits,
        "workbook": str(root / "blind" / "review.html"),
        "audit_summary": str(root / "blind" / "AUDIT_SUMMARY.json"),
        "usage": base._usage_totals(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "audit", "status"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root)
    elif args.command == "run":
        result = run_generation(args.root, workers=args.workers)
    elif args.command == "audit":
        result = run_audits(args.root, workers=min(args.workers, 2))
    else:
        result = status(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
