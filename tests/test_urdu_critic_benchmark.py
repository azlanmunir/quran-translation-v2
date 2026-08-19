from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from quran_translate.production_packets import atomic_json
from quran_translate.urdu_critic_benchmark import (
    _input_hash,
    benchmark_schema,
    benchmark_system,
    build_assignment,
    load_benchmark,
    prepare,
    rescore_existing,
    run_model,
    score_response,
    validate_response,
)
from quran_translate.urdu_translation_bakeoff import AUDITORS


class UrduCriticBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = load_benchmark()
        _user, cls.sources = build_assignment(cls.benchmark)

    def perfect_response(self) -> dict:
        rows = []
        for case in self.benchmark["cases"]:
            if case["expected"] == "clean":
                rows.append(
                    {"case_id": case["case_id"], "findings": [], "verdict": "pass"}
                )
                continue
            source = self.sources[case["case_id"]]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "findings": [
                        {
                            "type": case["accepted_types"][0],
                            "severity": case["minimum_severity"],
                            "where": source["urdu"],
                            "arabic_ground": source["arabic"],
                            "explanation": case["rationale"],
                            "suggestion": "Repair the identified defect.",
                        }
                    ],
                    "verdict": "revise",
                }
            )
        return {"cases": rows}

    def test_perfect_response_validates_and_passes(self) -> None:
        response = self.perfect_response()
        clean = validate_response(
            response,
            benchmark=self.benchmark,
            source_by_case=self.sources,
        )
        self.assertIsNotNone(clean)
        score = score_response(self.benchmark, clean)
        self.assertTrue(score["passed"])
        self.assertEqual(1.0, score["defect_recall"])
        self.assertEqual(0.0, score["clean_false_positive_rate"])

    def test_missing_required_case_fails_even_if_average_is_high(self) -> None:
        response = self.perfect_response()
        required = self.benchmark["thresholds"]["required_case_ids"][0]
        row = next(item for item in response["cases"] if item["case_id"] == required)
        row["findings"] = []
        row["verdict"] = "pass"
        clean = validate_response(
            response,
            benchmark=self.benchmark,
            source_by_case=self.sources,
        )
        score = score_response(self.benchmark, clean)
        self.assertFalse(score["passed"])
        self.assertIn(required, score["missed"])

    def test_prepare_is_frozen_and_spends_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = prepare(root)
            manifest = (root / "MANIFEST.json").read_bytes()
            second = prepare(root)
            self.assertEqual(first["version"], second["version"])
            self.assertEqual(manifest, (root / "MANIFEST.json").read_bytes())
            self.assertFalse(any(root.glob("auditor-*.json")))

    def test_benchmark_uses_production_policy_and_ledgers(self) -> None:
        system = benchmark_system()
        self.assertIn("=== TRANSLATION POLICY ===", system)
        self.assertIn("=== SEMANTIC LEDGER ===", system)
        self.assertIn("=== URDU DECISION LEDGER ===", system)
        self.assertIn("UPL-RAFATH", system)

    def test_groundless_benchmark_finding_is_rejected(self) -> None:
        response = self.perfect_response()
        response["cases"][0]["findings"][0]["where"] = "غیر موجود عبارت"
        self.assertIsNone(
            validate_response(
                response,
                benchmark=self.benchmark,
                source_by_case=self.sources,
            )
        )

    def test_rescore_reuses_identical_provider_assignment_without_cost(self) -> None:
        model = AUDITORS[0]
        user, _sources = build_assignment(self.benchmark)
        schema = benchmark_schema(
            [str(case["case_id"]) for case in self.benchmark["cases"]]
        )
        response = self.perfect_response()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "pilot.json"
            atomic_json(
                source_path,
                {
                    "status": "complete",
                    "model": asdict(model),
                    "input_hash": _input_hash(
                        model, benchmark_system(), user, schema
                    ),
                    "attempts": 1,
                    "latency_seconds": 1.0,
                    "usage": {"prompt_token_count": 1},
                    "cost_usd": 0.01,
                    "result": response,
                    "raw_text": "{}",
                    "raw_response": {},
                },
            )
            output_root = root / "v2"
            result = rescore_existing(model, source_path, output_root)
            self.assertTrue(result["score"]["passed"])
            self.assertEqual(0.0, result["cost_usd"])
            self.assertEqual(0.01, result["source_cost_usd"])
            self.assertEqual("deterministic_rescore", result["provenance"]["kind"])

    def test_terminal_provider_failure_is_not_retried(self) -> None:
        calls = 0

        def no_credits(*_args):
            nonlocal calls
            calls += 1
            raise RuntimeError("credit_balance_exhausted: no credits remaining")

        model = AUDITORS[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.dict(
                "quran_translate.urdu_critic_benchmark.PROVIDER_CALLS",
                {model.provider: no_credits},
            ):
                first = run_model(model, root)
                second = run_model(model, root)
        self.assertEqual("failed", first["status"])
        self.assertTrue(first["terminal_provider_failure"])
        self.assertEqual(first, second)
        self.assertEqual(1, calls)


if __name__ == "__main__":
    unittest.main()
