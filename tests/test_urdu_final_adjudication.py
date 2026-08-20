from __future__ import annotations

import json
import unittest
from copy import deepcopy

from quran_translate.urdu_final_adjudication import (
    DECISIONS_PATH,
    _apply_decisions,
    validate_decisions,
)
from quran_translate.urdu_production import UrduProductionError
from quran_translate.urdu_translation_bakeoff import stable_hash


class UrduFinalAdjudicationTests(unittest.TestCase):
    def test_registered_decisions_cover_the_terminal_block(self) -> None:
        payload = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
        decisions = validate_decisions(payload, payload["run_id"])
        self.assertEqual(
            {item["ref"] for item in decisions},
            {"17:7", "45:11", "53:20", "66:5"},
        )
        self.assertEqual(
            sum(item["expected_unresolved_records"] for item in decisions), 7
        )

    def test_application_is_exact_guarded_and_clears_only_matching_records(self) -> None:
        rows = [
            {"surah": 1, "ayah": 1, "urdu": "پہلے", "review_flags": []},
            {"surah": 1, "ayah": 2, "urdu": "باقی", "review_flags": []},
        ]
        unresolved = [
            {"unit_id": "u1", "ayah": 1, "kind": "a"},
            {"unit_id": "u1", "ayah": 1, "kind": "b"},
            {"unit_id": "u2", "ayah": 2, "kind": "c"},
        ]
        payload = {
            "version": "quran-urdu-final-adjudications-v1",
            "run_id": "test-run",
            "expected_unresolved_sha256": stable_hash(unresolved),
            "decisions": [
                {
                    "ref": "1:1",
                    "unit_id": "u1",
                    "ayah": 1,
                    "before": "پہلے",
                    "after": "بعد",
                    "category": "test",
                    "rationale": "Exact guarded test decision.",
                    "evidence": ["TEST:1:1"],
                    "expected_unresolved_records": 2,
                }
            ],
        }
        updated, remaining = _apply_decisions(
            deepcopy(rows), deepcopy(unresolved), payload
        )
        self.assertEqual(updated[0]["urdu"], "بعد")
        self.assertEqual(remaining, [unresolved[2]])

        changed = deepcopy(payload)
        changed["decisions"][0]["before"] = "غلط"
        with self.assertRaisesRegex(UrduProductionError, "before-text guard"):
            _apply_decisions(deepcopy(rows), deepcopy(unresolved), changed)

    def test_unresolved_set_hash_is_frozen(self) -> None:
        payload = {
            "version": "quran-urdu-final-adjudications-v1",
            "run_id": "test-run",
            "expected_unresolved_sha256": "0" * 64,
            "decisions": [
                {
                    "ref": "1:1",
                    "unit_id": "u1",
                    "ayah": 1,
                    "before": "پہلے",
                    "after": "بعد",
                    "category": "test",
                    "rationale": "Exact guarded test decision.",
                    "evidence": ["TEST:1:1"],
                    "expected_unresolved_records": 1,
                }
            ],
        }
        with self.assertRaisesRegex(UrduProductionError, "unresolved Urdu QA set"):
            _apply_decisions(
                [{"surah": 1, "ayah": 1, "urdu": "پہلے"}],
                [{"unit_id": "u1", "ayah": 1}],
                payload,
            )


if __name__ == "__main__":
    unittest.main()
