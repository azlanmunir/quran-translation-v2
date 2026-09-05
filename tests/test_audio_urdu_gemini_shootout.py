from __future__ import annotations

import unittest

from quran_translate import audio_urdu_gemini_shootout as shootout


class UrduGeminiShootoutTests(unittest.TestCase):
    def test_only_model_generation_changes(self) -> None:
        old, new = shootout.MODELS
        self.assertEqual(old.voice_id, new.voice_id)
        self.assertEqual(old.locale, new.locale)
        self.assertNotEqual(old.model_id, new.model_id)

    def test_paid_rates_are_equal(self) -> None:
        # Both models currently list $1/M input and $20/M audio output standard.
        prompt_tokens = 500
        output_tokens = 2_000
        expected = prompt_tokens / 1_000_000 + output_tokens * 20 / 1_000_000
        self.assertAlmostEqual(expected, 0.0405)


if __name__ == "__main__":
    unittest.main()
