from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quran_translate.production_final_verification_recovery import (
    recover_final_verification_units,
)
from quran_translate.production_packets import ProductionUnit, atomic_json


class SequenceGeminiClient:
    def __init__(self, responses: list[list[dict]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        result = self.responses.pop(0)
        return {
            "response": {
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(result)}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                },
            }
        }


def _finding(where: str) -> dict:
    return {
        "type": "sense_error",
        "severity": "minor",
        "where": where,
        "arabic_ground": "النص",
        "explanation": "Use the next test wording.",
    }


class ProductionFinalVerificationRecoveryTests(unittest.TestCase):
    def test_recovery_targets_final_reader_and_preserves_prior_records(self) -> None:
        unit = ProductionUnit("s001_001_001", 1, 1, 1, 1, 1, 1)
        invalid = [
            {
                "ayah": 1,
                "findings": [
                    {
                        **_finding("The repaired line."),
                        "arabic_ground": "wrong quote",
                    }
                ],
                "verdict": "revise",
            }
        ]
        valid = [{"ayah": 1, "findings": [], "verdict": "pass"}]
        client = SequenceGeminiClient([invalid, valid])

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            unit_path = base / "units" / unit.unit_id
            atomic_json(
                unit_path / "draft.json",
                {"result": [{"ayah": 1, "english": "The opening."}]},
            )
            atomic_json(
                unit_path / "critic.json",
                {
                    "result": [
                        {
                            "ayah": 1,
                            "findings": [_finding("The opening.")],
                            "verdict": "revise",
                        }
                    ]
                },
            )
            atomic_json(
                unit_path / "revision.json",
                {
                    "result": {
                        "ayahs": [{"ayah": 1, "english": "The revised line."}],
                        "decisions": [
                            {
                                "finding_id": "f-1-0",
                                "decision": "applied",
                                "reason": "First test revision.",
                            }
                        ],
                    }
                },
            )
            atomic_json(
                unit_path / "verification.json",
                {
                    "result": [
                        {
                            "ayah": 1,
                            "findings": [_finding("The revised line.")],
                            "verdict": "revise",
                        }
                    ]
                },
            )
            atomic_json(
                unit_path / "repair.json",
                {
                    "result": {
                        "ayahs": [{"ayah": 1, "english": "The repaired line."}],
                        "decisions": [
                            {
                                "finding_id": "f-1-0",
                                "decision": "applied",
                                "reason": "Bounded test repair.",
                            }
                        ],
                    }
                },
            )
            prior_failure = unit_path / "final_verification-attempt2-FAILED.json"
            atomic_json(prior_failure, {"attempt": 2, "raw": "preserve me"})

            result = recover_final_verification_units(
                base=base,
                units=[unit],
                verses={(1, 1): "النص"},
                shared_by_unit={unit.unit_id: "Shared evidence."},
                critic_system="Critic system.",
                client=client,  # type: ignore[arg-type]
                delay_seconds=0,
            )

            self.assertEqual(2, len(client.calls))
            self.assertEqual(1, result["recovered_count"])
            self.assertTrue(prior_failure.is_file())
            self.assertTrue(
                (
                    base
                    / "jobs"
                    / "final-verification-sync-recovery-a3-s001_001_001.json"
                ).is_file()
            )
            artifact = json.loads(
                (unit_path / "final_verification.json").read_text()
            )
            self.assertEqual(4, artifact["attempt"])
            self.assertEqual("sync-recovery", artifact["transport"])
            self.assertIn("The repaired line.", client.calls[0]["user"])
            self.assertIn("final fidelity check", client.calls[0]["user"])

            no_call_client = SequenceGeminiClient([])
            resumed = recover_final_verification_units(
                base=base,
                units=[unit],
                verses={(1, 1): "النص"},
                shared_by_unit={unit.unit_id: "Shared evidence."},
                critic_system="Critic system.",
                client=no_call_client,  # type: ignore[arg-type]
                delay_seconds=0,
            )
            self.assertEqual([], no_call_client.calls)
            self.assertEqual(1, resumed["recovered_count"])


if __name__ == "__main__":
    unittest.main()
