from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quran_translate.production_packets import ProductionUnit, atomic_json
from quran_translate.production_qa_remediation import (
    _deterministic_candidate_issues,
    _validate_decisions,
    _validate_verification,
    build_targets,
)
from quran_translate.refrains import repeated_ayah_groups


class ProductionQARemediationTests(unittest.TestCase):
    def test_target_builder_closes_identical_group_and_contract_enforces_it(self) -> None:
        unit = ProductionUnit("s001_001_002", 1, 1, 1, 2, 1, 2)
        verses = {(1, 1): "نفس النص", (1, 2): "نفس النص"}
        english = {"1:1": "The old line.", "1:2": "The old line."}
        groups = repeated_ayah_groups(verses)
        group_id = next(iter(groups))
        finding = {
            "stage": "final_verification",
            "ref": "1:1",
            "type": "sense_error",
            "severity": "significant",
            "where": "old",
            "arabic_ground": "نفس",
            "explanation": "The old wording has the wrong contextual sense.",
            "suggestion": "The new line.",
        }
        issue = {
            "code": "unresolved_fidelity_finding",
            "severity": "error",
            "ref": "1:1",
            "message": "sense_error: The old wording has the wrong contextual sense.",
        }

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            atomic_json(
                base / "QA_REPORT.json",
                {"passed": False, "issues": [issue], "issue_counts": {"error": 1}},
            )
            atomic_json(
                base / "REVIEW_QUEUE.json",
                {"fidelity_findings": [finding]},
            )
            atomic_json(
                base / "REFRAINS.json",
                {
                    "groups": {
                        group_id: {
                            "reason": "Frozen test refrain.",
                            "source": "policy",
                        }
                    },
                    "overrides": {"1:1": "The old line.", "1:2": "The old line."},
                },
            )
            packet = (
                "# Evidence\n\n## 1:1 morphology\n- نَفْس: N root=نفس\n\n"
                "## 1:2 morphology\n- نَفْس: N root=نفس\n"
            )
            path = base / "evidence" / f"{unit.unit_id}.md"
            path.parent.mkdir(parents=True)
            path.write_text(packet, encoding="utf-8")

            targets = build_targets(
                base=base,
                units=[unit],
                verses=verses,
                english=english,
            )

        self.assertEqual(["1:1", "1:2"], [target["ref"] for target in targets])
        self.assertFalse(targets[0]["closure_only"])
        self.assertTrue(targets[1]["closure_only"])
        finding_id = targets[0]["findings"][0]["finding_id"]
        valid = {
            "items": [
                {
                    "ref": "1:1",
                    "english": "The new line.",
                    "decisions": [
                        {
                            "finding_id": finding_id,
                            "decision": "applied",
                            "reason": "The Arabic and shared context support the correction.",
                        }
                    ],
                },
                {"ref": "1:2", "english": "The new line.", "decisions": []},
            ]
        }
        self.assertIsNotNone(_validate_decisions(valid, targets))
        unchanged = json.loads(json.dumps(valid))
        for item in unchanged["items"]:
            item["english"] = "The old line."
        self.assertIsNone(_validate_decisions(unchanged, targets))
        divergent = json.loads(json.dumps(valid))
        divergent["items"][1]["english"] = "A divergent line."
        self.assertIsNone(_validate_decisions(divergent, targets))

    def test_verification_and_deterministic_gates_reject_bad_outputs(self) -> None:
        targets = [
            {
                "ref": "1:1",
                "arabic": "النص",
                "candidate_english": "The text.",
            }
        ]
        valid = [{"ref": "1:1", "findings": [], "verdict": "pass"}]
        self.assertEqual(valid, _validate_verification(valid, targets))
        invalid = [
            {
                "ref": "1:1",
                "findings": [
                    {
                        "type": "sense_error",
                        "severity": "significant",
                        "where": "not in candidate",
                        "arabic_ground": "النص",
                        "explanation": "Invalid quote should fail the contract.",
                    }
                ],
                "verdict": "revise",
            }
        ]
        self.assertIsNone(_validate_verification(invalid, targets))

        issues = _deterministic_candidate_issues(
            verses={(1, 1): "same", (1, 2): "same"},
            english={"1:1": "Thus one.", "1:2": "Different."},
        )
        self.assertEqual(
            {"banned_term", "refrain_divergence"},
            {issue["code"] for issue in issues},
        )


if __name__ == "__main__":
    unittest.main()
