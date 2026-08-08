"""Publication cleanup regression tests."""

import json
import sqlite3
import unittest

from quran_translate.db import init_db
from quran_translate.publication import build_publication_layer, cleanup_translation


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

    def test_v24_publication_never_mechanically_rewrites_translation(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.row_factory = sqlite3.Row
        init_db(conn)
        conn.execute(
            """
            INSERT INTO source_ayahs VALUES
            ('1:1', 1, 1, 1, 'x', 'x', 'x', 'x', NULL)
            """
        )
        conn.execute(
            """
            INSERT INTO translation_runs VALUES
            ('run', 'claude-opus-4-6', 'production-v2.4-opus-gemini', 'hash',
             32, 6500, 3, 3, 'complete', 'now', 'now')
            """
        )
        original = "Thus the messenger speaks of grace."
        conn.execute(
            """
            INSERT INTO translations VALUES
            ('run', '1:1', ?, 'complete', ?, 'now', 'now')
            """,
            (original, json.dumps({"translation": original})),
        )
        conn.commit()

        result = build_publication_layer(conn, "run")
        published = conn.execute(
            "SELECT publication_translation FROM publication_translations"
        ).fetchone()[0]
        self.assertEqual(original, published)
        self.assertEqual(0, result["changed_ayahs"])
        self.assertEqual("identity_from_audited_production_text", result["policy"])
