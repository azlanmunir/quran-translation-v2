from __future__ import annotations

import unittest

from quran_translate.urdu_opus46_config_probe import (
    BATCH_RATES,
    EFFORT,
    MAX_OUTPUT_TOKENS,
    TREATMENTS,
    estimate_batch_cost,
    revision_user,
    treatment_requests,
)
from quran_translate.urdu_translation_bakeoff import build_passage_payloads


class UrduOpus46ConfigProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = next(
            item
            for item in build_passage_payloads()
            if item["passage"]["passage_id"] == "fasting"
        )
        cls.muse_result = {
            "ayahs": [
                {"ayah": ayah, "urdu": f"آیت {ayah}", "review_flags": []}
                for ayah in range(177, 188)
            ]
        }

    def test_two_single_attempt_medium_effort_treatments_are_bounded(self) -> None:
        requests = treatment_requests(self.payload, self.muse_result)
        self.assertEqual(len(requests), 2)
        self.assertEqual({item["custom_id"] for item in requests}, set(TREATMENTS))
        for request in requests:
            params = request["params"]
            self.assertEqual(params["max_tokens"], MAX_OUTPUT_TOKENS)
            self.assertEqual(params["output_config"]["effort"], EFFORT)
            self.assertEqual(params["thinking"], {"type": "adaptive"})

    def test_revision_prompt_preserves_skeleton_without_interpolation(self) -> None:
        prompt = revision_user(self.payload, self.muse_result)
        self.assertIn("FROZEN MUSE SPARK BASE DRAFT", prompt)
        self.assertIn("semantic skeleton", prompt)
        self.assertIn("Do not add brackets", prompt)
        self.assertIn("taqwa expressions", prompt)

    def test_batch_cost_uses_registered_discounted_rates(self) -> None:
        usage = {
            "input_tokens": 1_000_000,
            "cache_creation_input_tokens": 1_000_000,
            "cache_read_input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
        }
        self.assertEqual(estimate_batch_cost(usage), sum(BATCH_RATES.values()))


if __name__ == "__main__":
    unittest.main()
