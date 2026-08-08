from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quran_translate.production_packets import ProductionUnit, atomic_json
from quran_translate.production_recovery import recover_critic_units


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


class ProductionRecoveryTests(unittest.TestCase):
    def test_recovery_uses_fresh_jobs_and_preserves_failed_attempt(self) -> None:
        unit = ProductionUnit("s001_001_001", 1, 1, 1, 1, 1, 1)
        invalid = [
            {
                "ayah": 1,
                "findings": [
                    {
                        "type": "sense_error",
                        "severity": "minor",
                        "where": "The opening.",
                        "arabic_ground": "wrong quote",
                        "explanation": "A test finding.",
                    }
                ],
                "verdict": "revise",
            }
        ]
        valid = [{"ayah": 1, "findings": [], "verdict": "pass"}]
        client = SequenceGeminiClient([invalid, valid])

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            atomic_json(
                base / "units" / unit.unit_id / "draft.json",
                {"result": [{"ayah": 1, "english": "The opening."}]},
            )
            result = recover_critic_units(
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
            self.assertTrue(
                (base / "jobs" / "critic-sync-recovery-a3-s001_001_001.json").is_file()
            )
            self.assertTrue(
                (
                    base
                    / "units"
                    / unit.unit_id
                    / "critic-recovery-attempt3-FAILED.json"
                ).is_file()
            )
            artifact = json.loads(
                (base / "units" / unit.unit_id / "critic.json").read_text()
            )
            self.assertEqual(4, artifact["attempt"])
            self.assertEqual("sync-recovery", artifact["transport"])
            self.assertIn("النص", client.calls[0]["user"])
            self.assertIn("exact contiguous substring", client.calls[0]["user"])

            no_call_client = SequenceGeminiClient([])
            resumed = recover_critic_units(
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
