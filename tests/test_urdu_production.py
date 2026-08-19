from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quran_translate.production_packets import ProductionUnit, atomic_json
from quran_translate.urdu_production import (
    BudgetExceeded,
    BudgetLedger,
    DRAFT_MODEL,
    REFRAIN_SCHEMA,
    REVISION_MODEL,
    UrduProductionError,
    _critic_model,
    _sync_job,
    _validate_cli_args,
    _validate_refrain_choices,
    usage_cost,
)
from quran_translate.urdu_critic_benchmark import BENCHMARK_PATH, benchmark_system
from quran_translate.urdu_translation_bakeoff import (
    TRANSLATION_SCHEMA,
    file_hash,
    stable_hash,
    validate_translation,
)


class UrduProductionTests(unittest.TestCase):
    def test_frozen_price_math_handles_provider_usage_shapes(self) -> None:
        self.assertEqual(0.123, usage_cost(DRAFT_MODEL.model_id, {"cost": 0.123}))
        terra = usage_cost(
            "gpt-5.6-terra",
            {
                "input_tokens": 1000,
                "input_tokens_details": {"cached_tokens": 400},
                "output_tokens": 500,
            },
        )
        self.assertAlmostEqual(0.00728, terra)
        gemini = usage_cost(
            "gemini-3.7-flash",
            {
                "prompt_token_count": 1000,
                "cached_content_token_count": 400,
                "candidates_token_count": 500,
                "thoughts_token_count": 200,
            },
        )
        self.assertAlmostEqual(0.003105, gemini)
        opus = usage_cost(
            REVISION_MODEL.model_id,
            {
                "input_tokens": 1000,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 300,
                "output_tokens": 500,
            },
        )
        self.assertAlmostEqual(0.009825, opus)

    def test_budget_ledger_counts_persisted_spend_and_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            atomic_json(base / "units" / "u1" / "draft.json", {"cost_usd": 0.7})
            ledger = BudgetLedger(base, 1.0)
            ledger.reserve(0.2)
            self.assertAlmostEqual(0.1, ledger.report()["remaining_usd"])
            with self.assertRaises(BudgetExceeded):
                ledger.reserve(0.11)
            ledger.release(0.2)
            self.assertAlmostEqual(0.3, ledger.report()["remaining_usd"])

    def test_sync_stage_is_checkpointed_and_reused(self) -> None:
        calls = 0

        def fake_provider(_model, _system, _user, _schema):
            nonlocal calls
            calls += 1
            result = {
                "ayahs": [
                    {"ayah": 1, "urdu": "اللہ ایک ہے۔", "review_flags": []}
                ]
            }
            return json.dumps(result, ensure_ascii=False), {"cost": 0.01}, {"ok": True}

        unit = ProductionUnit("s001_001_001", 1, 1, 1, 1, 1, 1)
        validator = lambda value: validate_translation(value, [1])
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            budget = BudgetLedger(base, 1.0)
            with patch.dict(
                "quran_translate.urdu_production.PROVIDER_CALLS",
                {"openrouter": fake_provider},
            ):
                first = _sync_job(
                    base=base,
                    unit=unit,
                    stage="draft",
                    model=DRAFT_MODEL,
                    system="system",
                    user="user",
                    schema=TRANSLATION_SCHEMA,
                    validator=validator,
                    budget=budget,
                )
                second = _sync_job(
                    base=base,
                    unit=unit,
                    stage="draft",
                    model=DRAFT_MODEL,
                    system="system",
                    user="user",
                    schema=TRANSLATION_SCHEMA,
                    validator=validator,
                    budget=budget,
                )
            self.assertEqual("complete", first["status"])
            self.assertEqual("reused", second["status"])
            self.assertEqual(1, calls)
            self.assertAlmostEqual(0.01, budget.spent())

    def test_contract_invalid_attempt_is_persisted_and_charged(self) -> None:
        calls = 0

        def fake_provider(_model, _system, _user, _schema):
            nonlocal calls
            calls += 1
            if calls == 1:
                return '{"ayahs": []}', {"cost": 0.01}, {"attempt": 1}
            result = {
                "ayahs": [
                    {"ayah": 1, "urdu": "اللہ ایک ہے۔", "review_flags": []}
                ]
            }
            return (
                json.dumps(result, ensure_ascii=False),
                {"cost": 0.02},
                {"attempt": 2},
            )

        unit = ProductionUnit("s001_001_001", 1, 1, 1, 1, 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            budget = BudgetLedger(base, 1.0)
            with patch.dict(
                "quran_translate.urdu_production.PROVIDER_CALLS",
                {"openrouter": fake_provider},
            ):
                result = _sync_job(
                    base=base,
                    unit=unit,
                    stage="draft",
                    model=DRAFT_MODEL,
                    system="system",
                    user="user",
                    schema=TRANSLATION_SCHEMA,
                    validator=lambda value: validate_translation(value, [1]),
                    budget=budget,
                )
            self.assertEqual("complete", result["status"])
            self.assertEqual(2, calls)
            self.assertTrue(
                (base / "units" / unit.unit_id / "draft-attempt1-FAILED.json").is_file()
            )
            self.assertAlmostEqual(0.03, budget.spent())

    def test_refrain_choice_must_be_one_of_supplied_options(self) -> None:
        groups = [
            {
                "group_id": "g1",
                "options": ["ایک ہی ترجمہ", "دوسرا ترجمہ"],
            }
        ]
        valid = {
            "groups": [
                {"group_id": "g1", "urdu": "ایک ہی ترجمہ", "reason": "زیادہ درست"}
            ]
        }
        self.assertIsNotNone(_validate_refrain_choices(valid, groups))
        invalid = json.loads(json.dumps(valid, ensure_ascii=False))
        invalid["groups"][0]["urdu"] = "نیا تیسرا ترجمہ"
        self.assertIsNone(_validate_refrain_choices(invalid, groups))
        self.assertEqual("object", REFRAIN_SCHEMA["type"])

    def test_all_stage_run_rejects_diagnostic_limit(self) -> None:
        with self.assertRaisesRegex(Exception, "only valid for individual"):
            _validate_cli_args("run", 1)
        with self.assertRaisesRegex(Exception, "must be positive"):
            _validate_cli_args("draft", 0)
        _validate_cli_args("draft", 1)
        _validate_cli_args("run", None)

    def test_critic_approval_is_tied_to_untampered_result(self) -> None:
        model = {
            "candidate_id": "auditor-gpt-56-terra",
            "provider": "openai",
            "model_id": "gpt-5.6-terra",
            "reasoning": "high",
            "private_label": "GPT-5.6 Terra auditor",
        }
        score = {"passed": True, "defect_recall": 1.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "result.json"
            atomic_json(
                result_path,
                {"status": "complete", "model": model, "score": score},
            )
            approval_path = root / "approval.json"
            atomic_json(
                approval_path,
                {
                    "version": "urdu-critic-approval-v1",
                    "model": model,
                    "benchmark_sha256": file_hash(BENCHMARK_PATH),
                    "system_sha256": stable_hash(benchmark_system()),
                    "result_path": str(result_path.resolve()),
                    "result_sha256": file_hash(result_path),
                    "score": score,
                },
            )
            with patch(
                "quran_translate.urdu_production.APPROVAL_PATH", approval_path
            ):
                self.assertEqual("gpt-5.6-terra", _critic_model().model_id)
                atomic_json(
                    result_path,
                    {"status": "complete", "model": model, "score": {"passed": False}},
                )
                with self.assertRaisesRegex(UrduProductionError, "missing or changed"):
                    _critic_model()


if __name__ == "__main__":
    unittest.main()
