from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quran_translate.urdu_translation_opus46_extension import (
    EXTENSION_ID,
    FINALISTS,
    OPUS_46,
    _blind_key,
    _translation_input_hash,
)
from quran_translate.urdu_translation_bakeoff import build_passage_payloads


class UrduTranslationOpus46ExtensionTests(unittest.TestCase):
    def test_roster_is_exactly_new_model_and_three_frozen_finalists(self) -> None:
        self.assertEqual(OPUS_46.model_id, "claude-opus-4-6")
        self.assertEqual(
            {item.candidate_id for item in FINALISTS},
            {
                "anthropic-opus-4-6",
                "anthropic-opus-4-8",
                "anthropic-fable-5",
                "openrouter-muse-spark-12",
            },
        )

    def test_blind_key_is_private_complete_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _blind_key(root)
            first_bytes = (root / "PRIVATE_BLIND_KEY.json").read_bytes()
            second = _blind_key(root)
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, (root / "PRIVATE_BLIND_KEY.json").read_bytes())
            self.assertEqual(set(first), {item.candidate_id for item in FINALISTS})
            self.assertEqual(set(first.values()), {"Candidate A", "Candidate B", "Candidate C", "Candidate D"})
            self.assertEqual((root / "PRIVATE_BLIND_KEY.json").stat().st_mode & 0o777, 0o600)
            document = json.loads(first_bytes)
            self.assertEqual(document["bakeoff_id"], EXTENSION_ID)

    def test_recovery_budget_has_a_distinct_frozen_input_hash(self) -> None:
        payload = build_passage_payloads()[0]
        standard = _translation_input_hash(OPUS_46, payload, max_output_tokens=16_000)
        recovery = _translation_input_hash(OPUS_46, payload, max_output_tokens=32_000)
        self.assertNotEqual(standard, recovery)


if __name__ == "__main__":
    unittest.main()
