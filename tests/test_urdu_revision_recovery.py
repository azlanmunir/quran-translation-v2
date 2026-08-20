from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quran_translate.production_clients import BatchState
from quran_translate.production_packets import ProductionUnit, atomic_json
from quran_translate.urdu_production import BudgetLedger
from quran_translate.urdu_revision_recovery import (
    COMPLETE_NAME,
    freeze_targets,
    run_recovery,
)
from quran_translate.urdu_translation_bakeoff import file_hash


class FakeAnthropicBatchClient:
    def __init__(self) -> None:
        self.submissions = 0

    def submit(self, requests):
        self.submissions += 1
        self.requests = requests
        return BatchState("batch-1", "ended", {})

    def retrieve(self, batch_id):
        return BatchState(batch_id, "ended", {})

    def results(self, _batch_id):
        return [
            {
                "custom_id": request["custom_id"],
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "ayahs": [
                                            {
                                                "ayah": 1,
                                                "urdu": "اللہ ایک ہے۔",
                                                "review_flags": [],
                                            }
                                        ],
                                        "decisions": [
                                            {
                                                "finding_id": "f-1",
                                                "decision": "applied",
                                                "reason": "درستگی",
                                            }
                                        ],
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        ],
                        "usage": {"input_tokens": 100, "output_tokens": 20},
                    },
                },
            }
            for request in self.requests
        ]


class UrduRevisionRecoveryTests(unittest.TestCase):
    def _fixture(self, base: Path) -> tuple[ProductionUnit, dict[str, object]]:
        unit = ProductionUnit("s001_001_001", 1, 1, 1, 1, 1, 1)
        unit_root = base / "units" / unit.unit_id
        atomic_json(
            unit_root / "revision-attempt1-FAILED.json",
            {
                "version": "urdu-production-stage-failure-v1",
                "stage": "revision",
                "unit_id": unit.unit_id,
                "input_hash": "input-1",
                "attempt": 1,
                "errors": [
                    "ProviderError: Your credit balance is too low to access the Anthropic API"
                ],
                "usage": {},
                "raw": {"result": {"type": "errored"}},
            },
        )
        return unit, {
            "system": "system",
            "user": "user",
            "finding_ids": ["f-1"],
            "input_hash": "input-1",
        }

    def test_recovery_is_bounded_preserves_failure_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            unit, material = self._fixture(base)
            failure = base / "units" / unit.unit_id / "revision-attempt1-FAILED.json"
            before = failure.read_bytes()
            client = FakeAnthropicBatchClient()
            with patch(
                "quran_translate.urdu_revision_recovery._units_requiring_revision",
                return_value=[unit],
            ), patch(
                "quran_translate.urdu_revision_recovery._revision_material",
                return_value=(
                    material["system"],
                    material["user"],
                    material["finding_ids"],
                    material["input_hash"],
                ),
            ):
                marker = freeze_targets(base, [unit], {}, {})
                first = run_recovery(
                    base,
                    [unit],
                    {},
                    {},
                    marker,
                    budget=BudgetLedger(base, 100),
                    client=client,
                    poll_seconds=0,
                )
                second = run_recovery(
                    base,
                    [unit],
                    {},
                    {},
                    marker,
                    budget=BudgetLedger(base, 100),
                    client=client,
                    poll_seconds=0,
                )

            self.assertEqual(before, failure.read_bytes())
            self.assertEqual(1, client.submissions)
            self.assertEqual(1, first["complete"])
            self.assertTrue(second["recovery_complete"])
            self.assertTrue((base / COMPLETE_NAME).is_file())
            revision = json.loads(
                (base / "units" / unit.unit_id / "revision.json").read_text()
            )
            self.assertEqual("input-1", revision["input_hash"])
            self.assertEqual(2, revision["attempts"])

    def test_freeze_refuses_non_billing_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            unit, material = self._fixture(base)
            failure = base / "units" / unit.unit_id / "revision-attempt1-FAILED.json"
            document = json.loads(failure.read_text())
            document["errors"] = ["revision failed strict contract"]
            atomic_json(failure, document)
            with patch(
                "quran_translate.urdu_revision_recovery._units_requiring_revision",
                return_value=[unit],
            ), patch(
                "quran_translate.urdu_revision_recovery._revision_material",
                return_value=(
                    material["system"],
                    material["user"],
                    material["finding_ids"],
                    material["input_hash"],
                ),
            ):
                with self.assertRaisesRegex(Exception, "non-billing"):
                    freeze_targets(base, [unit], {}, {})

    def test_target_marker_detects_changed_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            unit, material = self._fixture(base)
            with patch(
                "quran_translate.urdu_revision_recovery._units_requiring_revision",
                return_value=[unit],
            ), patch(
                "quran_translate.urdu_revision_recovery._revision_material",
                return_value=(
                    material["system"],
                    material["user"],
                    material["finding_ids"],
                    material["input_hash"],
                ),
            ):
                freeze_targets(base, [unit], {}, {})
                failure = (
                    base / "units" / unit.unit_id / "revision-attempt1-FAILED.json"
                )
                original_hash = file_hash(failure)
                failure.write_text(failure.read_text() + "\n", encoding="utf-8")
                self.assertNotEqual(original_hash, file_hash(failure))
                with self.assertRaisesRegex(Exception, "evidence changed"):
                    freeze_targets(base, [unit], {}, {})


if __name__ == "__main__":
    unittest.main()
