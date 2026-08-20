from __future__ import annotations

import unittest

from quran_translate.urdu_release import _reader_html, _urdu_digits


class UrduReleaseTests(unittest.TestCase):
    def test_urdu_digits(self) -> None:
        self.assertEqual(_urdu_digits(6236), "۶۲۳۶")

    def test_reader_html_preserves_translation_and_escapes_markup(self) -> None:
        rows = []
        names = {}
        for surah in range(1, 115):
            names[surah] = f"نام {surah}"
            count = 7 if surah == 1 else 1
            for ayah in range(1, count + 1):
                rows.append({"surah": surah, "ayah": ayah, "urdu": "الف < ب"})
        rendered = _reader_html(rows, names)
        self.assertIn("الف &lt; ب", rendered)
        self.assertIn("﴿۱﴾", rendered)
        self.assertEqual(rendered.count('class="surah"'), 114)


if __name__ == "__main__":
    unittest.main()
