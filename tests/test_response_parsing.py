from __future__ import annotations

import unittest

from quran_translate.prompt_builder import parse_translation_response


class ResponseParsingTests(unittest.TestCase):
    def test_valid_response_parses(self) -> None:
        raw = """
        {
              "translations": [
                {
                  "ref": "1:1",
                  "translation": "By the mark of Allah...",
                  "word_bank": [
                    {
                      "term": "Rahman",
                      "root": "R-H-M",
                      "rendering": "The Womb",
                      "physical_reality": "A mother's body protecting life."
                    }
                  ]
                }
              ]
            }
        """
        parsed = parse_translation_response(raw, ["1:1"])
        self.assertEqual(parsed[0]["ref"], "1:1")

    def test_missing_ref_fails(self) -> None:
        raw = '{"translations": []}'
        with self.assertRaises(ValueError):
            parse_translation_response(raw, ["1:1"])


if __name__ == "__main__":
    unittest.main()
