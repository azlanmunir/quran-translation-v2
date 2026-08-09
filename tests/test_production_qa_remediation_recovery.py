from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quran_translate.production_qa_remediation import VERIFICATION_SCHEMA
from quran_translate.production_qa_remediation_recovery import (
    google_compatible_schema,
    recover_verification_bundles,
)


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


class ProductionQARemediationRecoveryTests(unittest.TestCase):
    def test_schema_recovery_writes_originally_compatible_bundle(self) -> None:
        provider_schema = google_compatible_schema(VERIFICATION_SCHEMA)
        self.assertNotIn("additionalProperties", json.dumps(provider_schema))
        target = {
            "ref": "1:1",
            "arabic": "النص",
            "candidate_english": "The text.",
            "local_context": [],
            "morphology": "",
        }
        valid = [{"ref": "1:1", "findings": [], "verdict": "pass"}]
        client = SequenceGeminiClient([valid])

        with tempfile.TemporaryDirectory() as temp:
            remediation = Path(temp)
            prior_failure = remediation / "verification" / "bundle-001-attempt1-FAILED.json"
            prior_failure.parent.mkdir(parents=True)
            prior_failure.write_text('{"preserve":true}', encoding="utf-8")

            result = recover_verification_bundles(
                remediation=remediation,
                targets=[target],
                target_bundle_numbers=[1],
                system="Critic system.",
                client=client,  # type: ignore[arg-type]
                delay_seconds=0,
            )

            self.assertEqual(1, result["recovered_count"])
            self.assertTrue(prior_failure.exists())
            self.assertNotIn(
                "additionalProperties",
                json.dumps(client.calls[0]["response_schema"]),
            )
            artifact = json.loads(
                (remediation / "verification" / "bundle-001.json").read_text()
            )
            self.assertEqual("sync-schema-recovery", artifact["transport"])
            self.assertEqual(valid, artifact["result"])

            resumed_client = SequenceGeminiClient([])
            resumed = recover_verification_bundles(
                remediation=remediation,
                targets=[target],
                target_bundle_numbers=[1],
                system="Critic system.",
                client=resumed_client,  # type: ignore[arg-type]
                delay_seconds=0,
            )
            self.assertEqual(1, resumed["recovered_count"])
            self.assertEqual([], resumed_client.calls)


if __name__ == "__main__":
    unittest.main()
