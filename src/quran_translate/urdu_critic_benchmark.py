"""Frozen, low-cost regression benchmark for selecting the Urdu critic."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import DATA_DIR, DEFAULT_SOURCE_XML, OUTPUT_DIR, PROJECT_ROOT
from .production_packets import atomic_json
from .refrains import load_quran_xml
from .urdu_quality import FINDING_SCHEMA, SEVERITY_RANK, validate_finding
from .urdu_costs import PRICING_PATH, usage_cost
from .urdu_provider import is_terminal_provider_failure
from .urdu_translation_bakeoff import (
    AUDITORS,
    CONTRACT_ATTEMPTS,
    ModelSpec,
    PROVIDER_CALLS,
    extract_json,
    file_hash,
    load_environment,
    stable_hash,
)


BENCHMARK_PATH = DATA_DIR / "evidence" / "urdu-critic-regression-v2.json"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "urdu-critic-production-v1.md"
POLICY_PATH = PROJECT_ROOT / "prompts" / "urdu-translation-v1.md"
SEMANTIC_LEDGER_MD_PATH = PROJECT_ROOT / "prompts" / "sense-ledger-v2.4.md"
SEMANTIC_LEDGER_JSON_PATH = DATA_DIR / "evidence" / "sense-ledger-v2.4.json"
URDU_LEDGER_PATH = PROJECT_ROOT / "prompts" / "urdu-production-ledger-v1.json"
APPROVAL_PATH = DATA_DIR / "evidence" / "urdu-critic-approved-v2.json"
APPROVED_RESULT_PATH = DATA_DIR / "evidence" / "urdu-critic-approved-result-v2.json"
DEFAULT_ROOT = OUTPUT_DIR / "urdu" / "critic-benchmarks" / "urdu-critic-regression-v2"
# PROVIDER_CALLS uses the frozen bakeoff transport, whose registered ceiling is
# 24k. Keep the benchmark manifest honest about the request actually sent.
MAX_OUTPUT_TOKENS = 24_000


class UrduCriticBenchmarkError(RuntimeError):
    """The critic benchmark cannot proceed without breaking its frozen contract."""


def benchmark_system() -> str:
    return (
        PROMPT_PATH.read_text(encoding="utf-8").strip()
        + "\n\n=== TRANSLATION POLICY ===\n"
        + POLICY_PATH.read_text(encoding="utf-8").strip()
        + "\n\n=== SEMANTIC LEDGER ===\n"
        + SEMANTIC_LEDGER_MD_PATH.read_text(encoding="utf-8").strip()
        + "\n\n=== STRUCTURED SEMANTIC RECORDS ===\n"
        + SEMANTIC_LEDGER_JSON_PATH.read_text(encoding="utf-8").strip()
        + "\n\n=== URDU DECISION LEDGER ===\n"
        + URDU_LEDGER_PATH.read_text(encoding="utf-8").strip()
    )


def benchmark_schema(case_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string", "enum": case_ids},
                        "findings": {"type": "array", "items": FINDING_SCHEMA},
                        "verdict": {"type": "string", "enum": ["pass", "revise"]},
                    },
                    "required": ["case_id", "findings", "verdict"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["cases"],
        "additionalProperties": False,
    }


def load_benchmark() -> dict[str, Any]:
    document = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    cases = document.get("cases")
    thresholds = document.get("thresholds")
    if document.get("version") != "urdu-critic-regression-v2":
        raise UrduCriticBenchmarkError("Unexpected critic benchmark version")
    if not isinstance(cases, list) or not cases or not isinstance(thresholds, dict):
        raise UrduCriticBenchmarkError("Critic benchmark is malformed")
    ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(ids) != len(cases) or len(set(ids)) != len(ids):
        raise UrduCriticBenchmarkError("Critic benchmark case IDs are invalid")
    return document


def build_assignment(document: dict[str, Any]) -> tuple[str, dict[str, dict[str, str]]]:
    verses = load_quran_xml(DEFAULT_SOURCE_XML)
    source_by_case: dict[str, dict[str, str]] = {}
    public_cases: list[dict[str, str]] = []
    for case in document["cases"]:
        surah, ayah = (int(value) for value in str(case["ref"]).split(":"))
        arabic = verses[(surah, ayah)]
        case_id = str(case["case_id"])
        source_by_case[case_id] = {"arabic": arabic, "urdu": str(case["urdu"])}
        public_cases.append(
            {
                "case_id": case_id,
                "ref": str(case["ref"]),
                "arabic": arabic,
                "urdu": str(case["urdu"]),
            }
        )
    user = (
        "Audit every independent test case. Some are clean controls. Do not assume "
        "that every item contains a defect. Return cases in the supplied order.\n\n"
        + json.dumps(public_cases, ensure_ascii=False, indent=2)
    )
    return user, source_by_case


def validate_response(
    response: Any,
    *,
    benchmark: dict[str, Any],
    source_by_case: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    expected = [str(case["case_id"]) for case in benchmark["cases"]]
    if not isinstance(response, dict) or set(response) != {"cases"}:
        return None
    rows = response.get("cases")
    if not isinstance(rows, list) or len(rows) != len(expected):
        return None
    if [row.get("case_id") for row in rows if isinstance(row, dict)] != expected:
        return None
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"case_id", "findings", "verdict"}:
            return None
        case_id = str(row["case_id"])
        findings = row.get("findings")
        verdict = row.get("verdict")
        if not isinstance(findings, list) or verdict not in {"pass", "revise"}:
            return None
        clean_findings: list[dict[str, str]] = []
        for finding in findings:
            clean = validate_finding(
                finding,
                arabic=source_by_case[case_id]["arabic"],
                urdu=source_by_case[case_id]["urdu"],
            )
            if clean is None:
                return None
            clean_findings.append(clean)
        severe = any(
            item["severity"] in {"blocking", "significant"}
            for item in clean_findings
        )
        if verdict == "revise" and not severe:
            return None
        if verdict == "pass" and severe:
            return None
        clean_rows.append(
            {"case_id": case_id, "findings": clean_findings, "verdict": verdict}
        )
    return {"cases": clean_rows}


def score_response(benchmark: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    expected_by_id = {str(case["case_id"]): case for case in benchmark["cases"]}
    result_by_id = {str(row["case_id"]): row for row in response["cases"]}
    defect_cases = [case for case in benchmark["cases"] if case["expected"] == "defect"]
    clean_cases = [case for case in benchmark["cases"] if case["expected"] == "clean"]
    detected: list[str] = []
    missed: list[str] = []
    false_positives: list[str] = []
    per_case: list[dict[str, Any]] = []
    for case_id, case in expected_by_id.items():
        row = result_by_id[case_id]
        minimum = SEVERITY_RANK[str(case["minimum_severity"])]
        severe_findings = [
            finding
            for finding in row["findings"]
            if SEVERITY_RANK[finding["severity"]] >= minimum
        ]
        if case["expected"] == "defect":
            matched = any(
                finding["type"] in set(case["accepted_types"])
                for finding in severe_findings
            )
            (detected if matched else missed).append(case_id)
        else:
            matched = not severe_findings
            if not matched:
                false_positives.append(case_id)
        per_case.append(
            {
                "case_id": case_id,
                "expected": case["expected"],
                "passed": matched,
                "reported_types": [finding["type"] for finding in severe_findings],
            }
        )
    recall = len(detected) / len(defect_cases)
    false_positive_rate = len(false_positives) / len(clean_cases)
    required = set(benchmark["thresholds"]["required_case_ids"])
    passed = (
        recall >= float(benchmark["thresholds"]["minimum_defect_recall"])
        and false_positive_rate
        <= float(benchmark["thresholds"]["maximum_clean_false_positive_rate"])
        and required.issubset(detected)
    )
    return {
        "version": "urdu-critic-regression-score-v2",
        "passed": passed,
        "defect_cases": len(defect_cases),
        "detected": detected,
        "missed": missed,
        "defect_recall": round(recall, 6),
        "clean_cases": len(clean_cases),
        "false_positives": false_positives,
        "clean_false_positive_rate": round(false_positive_rate, 6),
        "required_cases_passed": sorted(required.intersection(detected)),
        "per_case": per_case,
    }


def _manifest() -> dict[str, Any]:
    benchmark = load_benchmark()
    return {
        "version": "urdu-critic-benchmark-manifest-v2",
        "benchmark_sha256": file_hash(BENCHMARK_PATH),
        "system_inputs": {
            str(path.relative_to(PROJECT_ROOT)): file_hash(path)
            for path in (
                PROMPT_PATH,
                POLICY_PATH,
                SEMANTIC_LEDGER_MD_PATH,
                SEMANTIC_LEDGER_JSON_PATH,
                URDU_LEDGER_PATH,
            )
        },
        "system_sha256": stable_hash(benchmark_system()),
        "pricing_sha256": file_hash(PRICING_PATH),
        "source_sha256": file_hash(DEFAULT_SOURCE_XML),
        "models": [asdict(model) for model in AUDITORS],
        "schema": benchmark_schema([str(case["case_id"]) for case in benchmark["cases"]]),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "contract_attempts": CONTRACT_ATTEMPTS,
        "runner_sha256": file_hash(Path(__file__)),
    }


def prepare(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    path = root / "MANIFEST.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise UrduCriticBenchmarkError("Critic benchmark manifest changed")
    if not path.exists():
        atomic_json(path, manifest)
    return status(root)


def _result_path(root: Path, model: ModelSpec) -> Path:
    return root / f"{model.candidate_id}.json"


def _model_cost(root: Path, model: ModelSpec) -> float:
    total = 0.0
    paths = [
        _result_path(root, model),
        *root.glob(f"{model.candidate_id}-attempt*-FAILED.json"),
    ]
    for path in paths:
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8")).get("cost_usd")
        if isinstance(value, (int, float)):
            total += float(value)
    return round(total, 8)


def _input_hash(model: ModelSpec, system: str, user: str, schema: dict[str, Any]) -> str:
    return stable_hash(
        {
            "model": asdict(model),
            "system": system,
            "user": user,
            "schema": schema,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "contract_attempts": CONTRACT_ATTEMPTS,
        }
    )


def run_model(model: ModelSpec, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    prepare(root)
    benchmark = load_benchmark()
    system = benchmark_system()
    user, sources = build_assignment(benchmark)
    schema = benchmark_schema([str(case["case_id"]) for case in benchmark["cases"]])
    input_hash = _input_hash(model, system, user, schema)
    path = _result_path(root, model)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("input_hash") != input_hash:
            raise UrduCriticBenchmarkError(f"Cached benchmark input changed: {path}")
        if existing.get("status") == "complete":
            return existing
        if existing.get("terminal_provider_failure"):
            return existing
        if int(existing.get("attempts", 0)) >= CONTRACT_ATTEMPTS:
            return existing

    errors: list[str] = []
    consumed_attempts: set[int] = set()
    terminal_provider_failure = False
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        failed_path = root / f"{model.candidate_id}-attempt{attempt}-FAILED.json"
        if not failed_path.exists():
            continue
        failed = json.loads(failed_path.read_text(encoding="utf-8"))
        if failed.get("input_hash") != input_hash:
            raise UrduCriticBenchmarkError(
                f"Cached failed benchmark input changed: {failed_path}"
            )
        consumed_attempts.add(attempt)
        errors.extend(str(item) for item in failed.get("errors", []))
        terminal_provider_failure = terminal_provider_failure or bool(
            failed.get("terminal_provider_failure")
        )
    usage: dict[str, Any] = {}
    raw_response: dict[str, Any] | None = None
    raw_text = ""
    started = time.monotonic()
    last_attempt = max(consumed_attempts, default=0)
    for attempt in range(1, CONTRACT_ATTEMPTS + 1):
        if terminal_provider_failure:
            break
        if attempt in consumed_attempts:
            continue
        raw_text = ""
        raw_response = None
        usage = {}
        last_attempt = attempt
        try:
            retry = (
                "\n\nPrevious output failed the registered contract: " + errors[-1]
                if errors
                else ""
            )
            raw_text, usage, raw_response = PROVIDER_CALLS[model.provider](
                model, system, user + retry, schema
            )
            validated = validate_response(
                extract_json(raw_text),
                benchmark=benchmark,
                source_by_case=sources,
            )
            if validated is None:
                raise UrduCriticBenchmarkError(
                    "response failed exact benchmark contract"
                )
            document = {
                "version": "urdu-critic-benchmark-result-v2",
                "input_hash": input_hash,
                "status": "complete",
                "model": asdict(model),
                "attempts": attempt,
                "latency_seconds": round(time.monotonic() - started, 3),
                "usage": usage,
                "cost_usd": usage_cost(model.model_id, usage),
                "result": validated,
                "score": score_response(benchmark, validated),
                "errors_before_success": errors,
                "raw_text": raw_text,
                "raw_response": raw_response,
                "terminal_provider_failure": is_terminal_provider_failure(exc),
            }
            atomic_json(path, document)
            return document
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}"[:3000])
            failed_payload: dict[str, Any] = {
                "version": "urdu-critic-benchmark-failure-v2",
                "input_hash": input_hash,
                "model": asdict(model),
                "attempt": attempt,
                "errors": [errors[-1]],
                "usage": usage,
                "raw_text": raw_text,
                "raw_response": raw_response,
            }
            if usage:
                failed_payload["cost_usd"] = usage_cost(model.model_id, usage)
            atomic_json(
                root / f"{model.candidate_id}-attempt{attempt}-FAILED.json",
                failed_payload,
            )
            if is_terminal_provider_failure(exc):
                terminal_provider_failure = True
                break
    document = {
        "version": "urdu-critic-benchmark-result-v2",
        "input_hash": input_hash,
        "status": "failed",
        "model": asdict(model),
        "attempts": last_attempt,
        "latency_seconds": round(time.monotonic() - started, 3),
        "usage": usage,
        "terminal_provider_failure": terminal_provider_failure,
        "errors": errors,
        "raw_text": raw_text,
        "raw_response": raw_response,
    }
    atomic_json(path, document)
    return document


def rescore_existing(
    model: ModelSpec,
    source_path: Path,
    root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    prepare(root)
    if not source_path.is_file():
        raise UrduCriticBenchmarkError(f"Source result does not exist: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete" or source.get("model") != asdict(model):
        raise UrduCriticBenchmarkError("Source result is incomplete or from another model")
    benchmark = load_benchmark()
    system = benchmark_system()
    user, sources = build_assignment(benchmark)
    schema = benchmark_schema([str(case["case_id"]) for case in benchmark["cases"]])
    input_hash = _input_hash(model, system, user, schema)
    if source.get("input_hash") != input_hash:
        raise UrduCriticBenchmarkError(
            "Source result was generated from a different provider assignment"
        )
    validated = validate_response(
        source.get("result"),
        benchmark=benchmark,
        source_by_case=sources,
    )
    if validated is None:
        raise UrduCriticBenchmarkError("Source result fails the current strict contract")
    document = {
        "version": "urdu-critic-benchmark-result-v2",
        "input_hash": input_hash,
        "status": "complete",
        "model": asdict(model),
        "attempts": source.get("attempts"),
        "latency_seconds": source.get("latency_seconds"),
        "usage": source.get("usage", {}),
        "cost_usd": 0.0,
        "source_cost_usd": source.get("cost_usd"),
        "result": validated,
        "score": score_response(benchmark, validated),
        "raw_text": source.get("raw_text"),
        "raw_response": source.get("raw_response"),
        "provenance": {
            "kind": "deterministic_rescore",
            "source_path": str(source_path.resolve()),
            "source_sha256": file_hash(source_path),
        },
    }
    path = _result_path(root, model)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != document:
        raise UrduCriticBenchmarkError(f"Existing rescored result changed: {path}")
    atomic_json(path, document)
    return document


def status(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model in AUDITORS:
        path = _result_path(root, model)
        if not path.exists():
            rows.append(
                {
                    "model": model.candidate_id,
                    "status": "pending",
                    "cost_usd": _model_cost(root, model),
                }
            )
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model.candidate_id,
                "status": document.get("status"),
                "score": document.get("score"),
                "usage": document.get("usage", {}),
                "cost_usd": _model_cost(root, model),
            }
        )
    return {
        "version": "urdu-critic-benchmark-status-v2",
        "benchmark": str(BENCHMARK_PATH),
        "results": rows,
        "total_cost_usd": round(sum(row["cost_usd"] for row in rows), 8),
        "approved": json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
        if APPROVAL_PATH.exists()
        else None,
    }


def approve(candidate_id: str, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    model = next((item for item in AUDITORS if item.candidate_id == candidate_id), None)
    if model is None:
        raise UrduCriticBenchmarkError(f"Unknown critic candidate: {candidate_id}")
    path = _result_path(root, model)
    if not path.exists():
        raise UrduCriticBenchmarkError("Critic candidate has not been benchmarked")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or not result.get("score", {}).get("passed"):
        raise UrduCriticBenchmarkError("Critic candidate did not pass the frozen benchmark")
    pricing = json.loads(PRICING_PATH.read_text(encoding="utf-8"))
    if model.model_id not in pricing.get("per_million_tokens", {}):
        raise UrduCriticBenchmarkError(
            f"Pricing snapshot lacks critic candidate {model.model_id}"
        )
    approved_result = {
        "version": "urdu-critic-approved-result-v2",
        "input_hash": result["input_hash"],
        "status": result["status"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "cost_usd": result.get("cost_usd"),
        "source_cost_usd": result.get("source_cost_usd"),
        "result": result["result"],
        "score": result["score"],
        "provenance": result.get("provenance"),
    }
    if APPROVED_RESULT_PATH.exists() and json.loads(
        APPROVED_RESULT_PATH.read_text(encoding="utf-8")
    ) != approved_result:
        raise UrduCriticBenchmarkError("A different critic result is already approved")
    atomic_json(APPROVED_RESULT_PATH, approved_result)
    approval = {
        "version": "urdu-critic-approval-v2",
        "model": asdict(model),
        "benchmark_sha256": file_hash(BENCHMARK_PATH),
        "system_sha256": stable_hash(benchmark_system()),
        "result_path": str(APPROVED_RESULT_PATH.relative_to(PROJECT_ROOT)),
        "result_sha256": file_hash(APPROVED_RESULT_PATH),
        "score": result["score"],
    }
    if APPROVAL_PATH.exists() and json.loads(
        APPROVAL_PATH.read_text(encoding="utf-8")
    ) != approval:
        raise UrduCriticBenchmarkError("A different critic is already approved")
    atomic_json(APPROVAL_PATH, approval)
    return approval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Urdu production critic")
    parser.add_argument(
        "command", choices=["prepare", "run", "rescore", "status", "approve"]
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidate", choices=[item.candidate_id for item in AUDITORS])
    parser.add_argument("--source", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = prepare(args.root)
    elif args.command == "status":
        result = status(args.root)
    elif args.command == "run":
        if not args.candidate:
            raise SystemExit("--candidate is required for run")
        load_environment()
        model = next(item for item in AUDITORS if item.candidate_id == args.candidate)
        run_model(model, args.root)
        result = status(args.root)
    elif args.command == "rescore":
        if not args.candidate or not args.source:
            raise SystemExit("--candidate and --source are required for rescore")
        model = next(item for item in AUDITORS if item.candidate_id == args.candidate)
        rescore_existing(model, args.source, args.root)
        result = status(args.root)
    else:
        if not args.candidate:
            raise SystemExit("--candidate is required for approve")
        result = approve(args.candidate, args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
