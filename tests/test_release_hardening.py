from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quran_translate.book_pdf import EDITION_SUBTITLE, publication_note_parts
from quran_translate.db import init_db
from quran_translate.release_hardening import (
    ADJUDICATIONS_PATH,
    READING_NOTES_PATH,
    apply_release_adjudications,
    formula_violations,
    validate_adjudications,
    validate_reading_notes,
)


class ReleaseHardeningTests(unittest.TestCase):
    def test_registered_decisions_cover_formula_leaks_and_retained_titles(self) -> None:
        payload = json.loads(ADJUDICATIONS_PATH.read_text(encoding="utf-8"))
        decisions = validate_adjudications(payload, payload["run_id"])
        self.assertEqual(len(decisions), 34)
        self.assertEqual(sum(item["action"] == "change" for item in decisions), 32)
        self.assertEqual(
            {item["ref"] for item in decisions if item["action"] == "retain"},
            {"17:42", "70:3"},
        )

    def test_formula_gate_is_narrow(self) -> None:
        rows = [
            {"verse_key": "1:1", "translation": "Glory be to You!"},
            {"verse_key": "1:2", "translation": "Glory to the One who made it."},
            {"verse_key": "1:3", "translation": "He is full of glory."},
            {"verse_key": "1:4", "translation": "The Glorious."},
        ]
        violations = formula_violations(rows)  # type: ignore[arg-type]
        self.assertEqual([item["ref"] for item in violations], ["1:1", "1:2"])

    def test_reading_notes_are_unique_and_source_backed(self) -> None:
        payload = json.loads(READING_NOTES_PATH.read_text(encoding="utf-8"))
        refs = {entry["ref"] for entry in payload["notes"]}
        notes = validate_reading_notes(payload, refs)
        self.assertEqual(len(notes), 21)
        self.assertIn("4:34", refs)
        self.assertIn("112:2", refs)

    def test_front_matter_describes_the_actual_method(self) -> None:
        text = " ".join(publication_note_parts("book"))
        self.assertEqual(EDITION_SUBTITLE, "Evidence-Audited Modern English Translation")
        self.assertIn("sense before considering etymology", text)
        self.assertIn("AI-assisted translation", text)
        self.assertNotIn("physical root meanings", text)

    def test_adjudication_is_exact_guarded_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            adjudications = temp / "decisions.json"
            adjudications.write_text(
                json.dumps(
                    {
                        "version": "test-v1",
                        "run_id": "test-release",
                        "decisions": [
                            {
                                "ref": "1:1",
                                "action": "change",
                                "category": "subhana_formula",
                                "before": "Glory be to You!",
                                "after": "How perfect You are!",
                                "rationale": "Test correction.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            init_db(conn)
            conn.execute(
                """
                INSERT INTO translation_runs
                VALUES ('test-release', 'model', 'production-v2.4-test', 'hash',
                        1, 100, 0, 0, 'complete', 'now', 'now')
                """
            )
            conn.execute(
                """
                INSERT INTO source_ayahs
                VALUES ('1:1', 1, 1, 1, 'a', 'a', 'a', 'arabic', NULL)
                """
            )
            conn.execute(
                """
                INSERT INTO translations
                VALUES ('test-release', '1:1', 'Glory be to You!', 'complete',
                        '{}', 'now', 'now')
                """
            )
            conn.commit()

            with patch(
                "quran_translate.release_hardening.run_base",
                return_value=temp,
            ):
                first = apply_release_adjudications(
                    conn,
                    "test-release",
                    adjudications,
                )
                second = apply_release_adjudications(
                    conn,
                    "test-release",
                    adjudications,
                )
            translation = conn.execute(
                "SELECT translation FROM translations WHERE verse_key = '1:1'"
            ).fetchone()["translation"]
            self.assertEqual(translation, "How perfect You are!")
            self.assertEqual(first["changed"], 1)
            self.assertEqual(second["already_applied"], 1)
            conn.close()


if __name__ == "__main__":
    unittest.main()
