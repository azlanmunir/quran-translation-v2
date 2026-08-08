from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quran_translate.config import DEFAULT_SOURCE_XML
from quran_translate.db import init_db
from quran_translate.production_clients import BatchState
from quran_translate.production_packets import ProductionUnit, build_units, source_verses
from quran_translate.production_refrains import resolve_refrains
from quran_translate.production_runner import (
    ProductionConfig,
    ProductionError,
    run_anthropic_stage,
    run_production,
    validate_critic_response,
    validate_reader,
    validate_revision,
)
from quran_translate.source_import import import_tanzil_xml
from quran_translate.refrains import repeated_ayah_groups


class FakeAnthropicBatchClient:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.retrieve_calls = 0
        self.requests: list[dict] = []

    def submit(self, requests: list[dict]) -> BatchState:
        self.submit_calls += 1
        self.requests = requests
        return BatchState("fake-batch-1", "ended", {})

    def retrieve(self, batch_id: str) -> BatchState:
        self.retrieve_calls += 1
        return BatchState(batch_id, "ended", {})

    def results(self, _batch_id: str) -> list[dict]:
        rows = []
        for request in self.requests:
            if request["custom_id"].startswith("refr-"):
                user = request["params"]["messages"][0]["content"]
                group_id = re.search(r"GROUP_ID: ([0-9a-f]+)", user).group(1)
                candidates = user.split("CURRENT CANDIDATE RENDERINGS:\n", 1)[1]
                english = candidates.splitlines()[0].removeprefix("- ")
                text = json.dumps(
                    {
                        "group_id": group_id,
                        "english": english,
                        "reason": "Fake test resolution",
                    }
                )
                rows.append(
                    {
                        "custom_id": request["custom_id"],
                        "result": {
                            "type": "succeeded",
                            "message": {
                                "model": "claude-opus-4-6",
                                "stop_reason": "end_turn",
                                "usage": {"input_tokens": 10, "output_tokens": 5},
                                "content": [{"type": "text", "text": text}],
                            },
                        },
                    }
                )
                continue
            match = re.search(r"s(\d{3})_(\d{3})_(\d{3})$", request["custom_id"])
            if not match:
                raise AssertionError(request["custom_id"])
            surah, first, last = (int(value) for value in match.groups())
            text = json.dumps(
                [
                    {"ayah": ayah, "english": f"English {surah}:{ayah}"}
                    for ayah in range(first, last + 1)
                ]
            )
            rows.append(
                {
                    "custom_id": request["custom_id"],
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "model": "claude-opus-4-6",
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 10, "output_tokens": 5},
                            "content": [{"type": "text", "text": text}],
                        },
                    },
                }
            )
        return rows


class FakeGeminiBatchClient:
    def __init__(self) -> None:
        self.jobs: dict[str, list[dict]] = {}

    def submit_file(self, *, model: str, input_path: Path, display_name: str) -> BatchState:
        del model
        job_id = f"fake-{display_name}"
        self.jobs[job_id] = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return BatchState(job_id, "JOB_STATE_SUCCEEDED", {})

    def retrieve(self, batch_id: str) -> BatchState:
        return BatchState(batch_id, "JOB_STATE_SUCCEEDED", {})

    def file_results(self, batch_id: str) -> list[dict]:
        rows = []
        for item in self.jobs[batch_id]:
            match = re.fullmatch(r"s(\d{3})_(\d{3})_(\d{3})", item["key"])
            if not match:
                raise AssertionError(item["key"])
            _surah, first, last = (int(value) for value in match.groups())
            result = [
                {"ayah": ayah, "findings": [], "verdict": "pass"}
                for ayah in range(first, last + 1)
            ]
            rows.append(
                {
                    "key": item["key"],
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": json.dumps(result)}]
                                }
                            }
                        ],
                        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
                    },
                }
            )
        return rows


class FlakyRefrainClient(FakeAnthropicBatchClient):
    def results(self, batch_id: str) -> list[dict]:
        if self.submit_calls == 1:
            return [
                {
                    "custom_id": request["custom_id"],
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "model": "claude-opus-4-6",
                            "usage": {},
                            "content": [{"type": "text", "text": "{}"}],
                        },
                    },
                }
                for request in self.requests
            ]
        return super().results(batch_id)


class BlockingFinalGeminiClient(FakeGeminiBatchClient):
    def file_results(self, batch_id: str) -> list[dict]:
        rows = super().file_results(batch_id)
        if "refrain_verification" not in batch_id or not rows:
            return rows
        request = self.jobs[batch_id][0]["request"]
        user = request["contents"][0]["parts"][0]["text"]
        target = re.search(
            r"=== TARGET AYAHS TO TRANSLATE:.*?===\n\((\d+)\) ([^\n]+)",
            user,
            re.DOTALL,
        )
        reader_match = re.search(
            r"=== ENGLISH TO AUDIT ===\n(\[.*?\])\n\nThis audit",
            user,
            re.DOTALL,
        )
        if not target or not reader_match:
            raise AssertionError("Could not parse fake refrain-verification request")
        ayah = int(target.group(1))
        arabic = target.group(2)
        result = json.loads(
            rows[0]["response"]["candidates"][0]["content"]["parts"][0]["text"]
        )
        result[0] = {
            "ayah": ayah,
            "findings": [
                {
                    "type": "omission",
                    "severity": "significant",
                    "where": "<missing>",
                    "arabic_ground": arabic,
                    "explanation": "A deliberate QA-gate test finding.",
                }
            ],
            "verdict": "revise",
        }
        rows[0]["response"]["candidates"][0]["content"]["parts"][0]["text"] = (
            json.dumps(result)
        )
        return rows


class ProductionRunnerTests(unittest.TestCase):
    def make_source_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.row_factory = sqlite3.Row
        init_db(conn)
        import_tanzil_xml(conn, DEFAULT_SOURCE_XML)
        return conn

    def test_units_cover_full_source_once_with_frozen_boundaries(self) -> None:
        conn = self.make_source_db()
        units = build_units(
            conn,
            max_ayahs=32,
            max_arabic_chars=6500,
            context_ayahs=3,
        )
        refs = [
            (unit.surah, ayah)
            for unit in units
            for ayah in unit.expected_ayahs
        ]
        self.assertEqual(6236, len(refs))
        self.assertEqual(6236, len(set(refs)))
        self.assertEqual((1, 1), refs[0])
        self.assertEqual((114, 6), refs[-1])
        self.assertEqual(254, len(units))
        self.assertTrue(all(len(unit.expected_ayahs) <= 32 for unit in units))
        for unit in units:
            chars = sum(
                len(row["arabic_uthmani_min"])
                for row in conn.execute(
                    """
                    SELECT arabic_uthmani_min FROM source_ayahs
                    WHERE surah_number = ? AND ayah_number BETWEEN ? AND ?
                    """,
                    (unit.surah, unit.first_ayah, unit.last_ayah),
                )
            )
            self.assertLessEqual(chars, 6500)

    def test_reader_and_revision_contracts_are_strict(self) -> None:
        self.assertIsNotNone(
            validate_reader(
                [{"ayah": 1, "english": "A line.", "review_flags": ["legal"]}],
                [1],
            )
        )
        self.assertIsNone(
            validate_reader([{"ayah": 1, "english": "A line.", "extra": True}], [1])
        )
        self.assertIsNotNone(
            validate_revision(
                {
                    "ayahs": [{"ayah": 1, "english": "A repair."}],
                    "decisions": [
                        {
                            "finding_id": "f-1-0",
                            "decision": "applied",
                            "reason": "Repairs the omission.",
                        }
                    ],
                },
                [1],
                ["f-1-0"],
            )
        )
        fabricated_ground = [
            {
                "ayah": 1,
                "findings": [
                    {
                        "type": "omission",
                        "severity": "significant",
                        "where": "<missing>",
                        "arabic_ground": "ليس في المصدر",
                        "explanation": "A claimed omission.",
                    }
                ],
                "verdict": "revise",
            }
        ]
        self.assertIsNone(
            validate_critic_response(
                fabricated_ground,
                [1],
                [{"ayah": 1, "english": "A line."}],
                {1: "النص العربي"},
            )
        )
        self.assertIsNone(
            validate_revision(
                {
                    "ayahs": [{"ayah": 1, "english": "A repair."}],
                    "decisions": [],
                },
                [1],
                ["f-1-0"],
            )
        )

    def test_anthropic_resume_reuses_frozen_shard_after_partial_ingest(self) -> None:
        units = [
            ProductionUnit("s001_001_001", 1, 1, 1, 1, 1, 1),
            ProductionUnit("s001_002_002", 2, 1, 2, 2, 2, 2),
        ]
        client = FakeAnthropicBatchClient()

        def assignment(unit: ProductionUnit, _attempt: int):
            expected = unit.expected_ayahs
            return (
                f"Translate {unit.unit_id}",
                {"unit": unit.to_dict()},
                lambda data, expected=expected: validate_reader(data, expected),
            )

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            run_anthropic_stage(
                base=base,
                stage="draft",
                units=units,
                shard_size=10,
                poll_seconds=0,
                system=[{"type": "text", "text": "system"}],
                assignment=assignment,
                client=client,
            )
            second = base / "units" / units[1].unit_id / "draft.json"
            second.unlink()

            run_anthropic_stage(
                base=base,
                stage="draft",
                units=units,
                shard_size=10,
                poll_seconds=0,
                system=[{"type": "text", "text": "system"}],
                assignment=assignment,
                client=client,
            )
            self.assertTrue(second.exists())
            self.assertEqual(1, client.submit_calls)
            self.assertGreaterEqual(client.retrieve_calls, 1)

    def test_refrain_contract_failure_retries_only_failed_groups(self) -> None:
        client = FlakyRefrainClient()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            policy = base / "policy.json"
            policy.write_text('{"canonical": {}}', encoding="utf-8")
            report = resolve_refrains(
                base=base,
                verses={(1, 1): "same", (1, 2): "same"},
                translations={(1, 1): "First", (1, 2): "Second"},
                policy_path=policy,
                shared_system=[],
                client=client,
                poll_seconds=0,
            )
            self.assertEqual(2, client.submit_calls)
            self.assertEqual(1, report["groups_model_resolved"])
            self.assertEqual(1, len(set(report["overrides"].values())))
            resumed = resolve_refrains(
                base=base,
                verses={(1, 1): "same", (1, 2): "same"},
                translations={(1, 1): "First", (1, 2): "Second"},
                policy_path=policy,
                shared_system=[],
                client=client,
                poll_seconds=0,
            )
            self.assertEqual(report["overrides"], resumed["overrides"])
            self.assertEqual(2, client.submit_calls)

    def test_full_dry_run_never_constructs_provider_clients(self) -> None:
        conn = self.make_source_db()
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch(
                    "quran_translate.production_runner.PRODUCTION_ROOT", Path(temp)
                ),
                patch(
                    "quran_translate.production_runner.AnthropicBatchClient",
                    side_effect=AssertionError("Anthropic client constructed in dry run"),
                ),
                patch(
                    "quran_translate.production_runner.GeminiBatchClient",
                    side_effect=AssertionError("Gemini client constructed in dry run"),
                ),
            ):
                report = run_production(
                    conn,
                    ProductionConfig(run_id="dry_test"),
                    dry_run=True,
                )
        self.assertEqual(6236, report["ayah_count"])
        self.assertEqual(254, report["unit_count"])
        self.assertEqual(0, report["api_requests_submitted"])

    def test_full_fake_provider_run_persists_all_6236_ayahs(self) -> None:
        conn = self.make_source_db()
        anthropic = FakeAnthropicBatchClient()
        gemini = FakeGeminiBatchClient()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "fake_full"
            with (
                patch(
                    "quran_translate.production_runner.PRODUCTION_ROOT", Path(temp)
                ),
                patch("quran_translate.production_runner.load_environment"),
                patch(
                    "quran_translate.production_runner.AnthropicBatchClient",
                    return_value=anthropic,
                ),
                patch(
                    "quran_translate.production_runner.GeminiBatchClient",
                    return_value=gemini,
                ),
            ):
                result = run_production(
                    conn,
                    ProductionConfig(
                        run_id="fake_full", shard_size=48, poll_seconds=0
                    ),
                )
            qa = json.loads((base / "QA_REPORT.json").read_text(encoding="utf-8"))
            complete_marker = (base / "PRODUCTION_COMPLETE.json").exists()
        translated = conn.execute(
            "SELECT COUNT(*) FROM translations WHERE run_id = 'fake_full'"
        ).fetchone()[0]
        self.assertEqual(6236, translated)
        self.assertEqual(6236, result["final_ayahs_ready"])
        self.assertEqual(254, result["stage_counts"]["draft"])
        self.assertEqual(254, result["stage_counts"]["critic"])
        self.assertEqual(0, result["stage_counts"]["revision"])
        self.assertEqual(254, result["stage_counts"]["spoken"])
        self.assertTrue(qa["passed"])
        self.assertEqual(96, qa["refrains"]["groups"])
        self.assertTrue(complete_marker)
        status = conn.execute(
            "SELECT status FROM translation_runs WHERE run_id = 'fake_full'"
        ).fetchone()[0]
        self.assertEqual("complete", status)
        verses = source_verses(conn)
        for group in repeated_ayah_groups(verses).values():
            rendered = {
                conn.execute(
                    "SELECT translation FROM translations WHERE run_id = 'fake_full' "
                    "AND verse_key = ?",
                    (f"{surah}:{ayah}",),
                ).fetchone()[0]
                for surah, ayah in group["refs"]
            }
            self.assertEqual(1, len(rendered))

    def test_major_final_fidelity_finding_blocks_completion(self) -> None:
        conn = self.make_source_db()
        anthropic = FakeAnthropicBatchClient()
        gemini = BlockingFinalGeminiClient()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "blocked_full"
            with (
                patch(
                    "quran_translate.production_runner.PRODUCTION_ROOT", Path(temp)
                ),
                patch("quran_translate.production_runner.load_environment"),
                patch(
                    "quran_translate.production_runner.AnthropicBatchClient",
                    return_value=anthropic,
                ),
                patch(
                    "quran_translate.production_runner.GeminiBatchClient",
                    return_value=gemini,
                ),
            ):
                with self.assertRaises(ProductionError):
                    run_production(
                        conn,
                        ProductionConfig(
                            run_id="blocked_full", shard_size=254, poll_seconds=0
                        ),
                    )
            self.assertTrue((base / "QA_BLOCKED.json").exists())
            self.assertFalse((base / "PRODUCTION_COMPLETE.json").exists())
            qa = json.loads((base / "QA_REPORT.json").read_text(encoding="utf-8"))
            self.assertFalse(qa["passed"])
            self.assertGreater(qa["review"]["major_fidelity_findings"], 0)
        status = conn.execute(
            "SELECT status FROM translation_runs WHERE run_id = 'blocked_full'"
        ).fetchone()[0]
        self.assertEqual("qa_blocked", status)


if __name__ == "__main__":
    unittest.main()
