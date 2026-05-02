"""Publication cleanup regression tests."""

import unittest

from quran_translate.publication import cleanup_translation


class PublicationCleanupTests(unittest.TestCase):
    def test_cleanup_preserves_case_for_banned_words(self) -> None:
        result = cleanup_translation("Thus the Messenger and the messengers came.")

        self.assertEqual(result.text, "So the Envoy and the envoys came.")
        self.assertIn("thus: 1", result.edits)
        self.assertIn("messenger: 1", result.edits)

    def test_lest_phrases_are_rewritten_without_fallback(self) -> None:
        result = cleanup_translation(
            "Do not track your vision for your brothers, lest they set a trap for you."
        )

        self.assertEqual(
            result.text,
            "Do not track your vision for your brothers, so they do not set a trap for you.",
        )
        self.assertNotIn("so that not", result.text)
