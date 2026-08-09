from __future__ import annotations

import unittest

from quran_translate.production_qa_final_adjudication import build_final_targets


class ProductionQAFinalAdjudicationTests(unittest.TestCase):
    def test_verifier_findings_become_ordered_targets_with_group_closure(self) -> None:
        group = {
            "group_id": "same",
            "refs": ["1:1", "1:2"],
            "required_identical": True,
        }
        candidates = [
            {
                "ref": ref,
                "unit_id": "s001_001_002",
                "arabic": "نفس النص",
                "current_english": "Original.",
                "candidate_english": "Candidate.",
                "decisions": [],
                "identical_group": group,
                "local_context": [],
                "parallel_occurrences": [],
                "morphology": "",
            }
            for ref in ("1:1", "1:2")
        ]
        verification = [
            {
                "ref": "1:1",
                "findings": [
                    {
                        "type": "sense_error",
                        "severity": "significant",
                        "where": "Candidate",
                        "arabic_ground": "نفس",
                        "explanation": "The candidate sense is wrong.",
                    }
                ],
                "verdict": "revise",
            },
            {"ref": "1:2", "findings": [], "verdict": "pass"},
        ]

        targets = build_final_targets(candidates, verification)

        self.assertEqual(["1:1", "1:2"], [target["ref"] for target in targets])
        self.assertEqual("Candidate.", targets[0]["current_english"])
        self.assertFalse(targets[0]["closure_only"])
        self.assertTrue(targets[1]["closure_only"])
        self.assertTrue(targets[0]["findings"][0]["finding_id"].startswith("qa-"))


if __name__ == "__main__":
    unittest.main()
