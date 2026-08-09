from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quran_translate.production_packets import ProductionUnit, atomic_json
from quran_translate.production_spoken_recovery import recover_spoken_units


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


class ProductionSpokenRecoveryTests(unittest.TestCase):
    def test_compact_recovery_preserves_failure_and_resumes(self) -> None:
        unit = ProductionUnit("s001_001_002", 1, 1, 1, 2, 1, 2)
        reader = [
            {"ayah": 1, "english": "The opening."},
            {"ayah": 2, "english": "The road."},
        ]
        invalid = [
            {"ayah": 1, "findings": [], "verdict": "pass"},
        ]
        valid = [
            {"ayah": 1, "findings": [], "verdict": "pass"},
            {"ayah": 2, "findings": [], "verdict": "pass"},
        ]
        client = SequenceGeminiClient([invalid, valid])

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            unit_path = base / "units" / unit.unit_id
            prior_failure = unit_path / "spoken-attempt2-FAILED.json"
            atomic_json(prior_failure, {"attempt": 2, "raw": "preserve me"})

            result = recover_spoken_units(
                base=base,
                units=[unit],
                verses={(1, 1): "النص الأول", (1, 2): "النص الثاني"},
                readers_by_unit={unit.unit_id: reader},
                shared_by_unit={unit.unit_id: "Large frozen evidence packet."},
                spoken_system="Spoken system.",
                client=client,  # type: ignore[arg-type]
                delay_seconds=0,
            )

            self.assertEqual(2, len(client.calls))
            self.assertEqual(1, result["recovered_count"])
            self.assertTrue(prior_failure.is_file())
            self.assertNotIn("Large frozen evidence packet.", client.calls[0]["user"])
            self.assertIn("النص الأول", client.calls[0]["user"])
            self.assertIn("The opening.", client.calls[0]["user"])
            artifact = json.loads((unit_path / "spoken.json").read_text())
            self.assertEqual(4, artifact["attempt"])
            self.assertEqual("sync-recovery", artifact["transport"])
            self.assertTrue(artifact["recovery"]["compact_assignment"])

            resumed_client = SequenceGeminiClient([])
            resumed = recover_spoken_units(
                base=base,
                units=[unit],
                verses={(1, 1): "النص الأول", (1, 2): "النص الثاني"},
                readers_by_unit={unit.unit_id: reader},
                shared_by_unit={unit.unit_id: "Large frozen evidence packet."},
                spoken_system="Spoken system.",
                client=resumed_client,  # type: ignore[arg-type]
                delay_seconds=0,
            )
            self.assertEqual([], resumed_client.calls)
            self.assertEqual(1, resumed["recovered_count"])


if __name__ == "__main__":
    unittest.main()
