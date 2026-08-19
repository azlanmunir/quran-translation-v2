from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quran_translate.urdu_critic_benchmark import (
    benchmark_system,
    build_assignment,
    load_benchmark,
    prepare,
    score_response,
    validate_response,
)


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


if __name__ == "__main__":
    unittest.main()
