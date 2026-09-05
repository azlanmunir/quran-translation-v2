from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quran_translate.short_form_strategy import (
    SCORE_KEYS,
    ShortFormStrategyError,
    build_short_form_catalog,
)


class ShortFormStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.release = self.root / "release.json"
        self.strategy = self.root / "strategy.json"
        self.output = self.root / "catalog.json"
        rows = [
            {
                "ref": f"49:{ayah}",
                "surah": 49,
                "ayah": ayah,
                "surah_name_en": "Al-Hujurat",
                "surah_meaning_en": "The Chambers",
                "translation": f"Exact release text {ayah}.",
            }
            for ayah in range(1, 6)
        ]
        self.release.write_text(
            json.dumps({"run_id": "test-release", "ayahs": rows}), encoding="utf-8"
        )
        self.strategy_payload = {
            "strategy_name": "The Open Door",
            "context_ayahs_each_side": 1,
            "narration_words_per_minute": 120,
            "scoring_weights": {key: 12.5 for key in SCORE_KEYS},
            "candidates": [
                {
                    "candidate_id": "sf-001",
                    "start_ref": "49:2",
                    "end_ref": "49:3",
                    "pillar": "for_real_life",
                    "scores": {key: 4 for key in SCORE_KEYS},
                }
            ],
        }
        self.strategy.write_text(json.dumps(self.strategy_payload), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build(self) -> dict:
        return build_short_form_catalog(
            release_path=self.release,
            strategy_path=self.strategy,
            output_path=self.output,
        )

    def test_preserves_exact_text_and_attaches_context(self) -> None:
        payload = self.build()
        candidate = payload["candidates"][0]
        self.assertEqual(candidate["refs"], ["49:2", "49:3"])
        self.assertEqual(
            candidate["exact_translation"],
            "Exact release text 2. Exact release text 3.",
        )
        self.assertEqual([row["ref"] for row in candidate["context_before"]], ["49:1"])
        self.assertEqual([row["ref"] for row in candidate["context_after"]], ["49:4"])
        self.assertEqual(candidate["editorial_score_100"], 80.0)
        self.assertFalse(payload["rules"]["automated_publication_allowed"])

    def test_refuses_cross_surah_or_reverse_range(self) -> None:
        self.strategy_payload["candidates"][0]["end_ref"] = "48:1"
        self.strategy.write_text(json.dumps(self.strategy_payload), encoding="utf-8")
        with self.assertRaisesRegex(ShortFormStrategyError, "within one Surah"):
            self.build()

    def test_refuses_out_of_contract_score(self) -> None:
        self.strategy_payload["candidates"][0]["scores"]["clarity"] = 6
        self.strategy.write_text(json.dumps(self.strategy_payload), encoding="utf-8")
        with self.assertRaisesRegex(ShortFormStrategyError, "integer from 1 to 5"):
            self.build()


if __name__ == "__main__":
    unittest.main()
