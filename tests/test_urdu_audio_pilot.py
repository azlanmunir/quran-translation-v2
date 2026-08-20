from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from quran_translate.production_packets import atomic_json

from quran_translate.urdu_audio_pilot import (
    MAX_ESTIMATED_STANDARD_COST_USD,
    PASSAGES,
    _estimate,
    _passage_payloads,
    recover_empty_audio,
)


class UrduAudioPilotTests(unittest.TestCase):
    def _rows(self) -> list[dict[str, object]]:
        rows = []
        for spec in PASSAGES:
            for surah, first, last in spec.ranges:
                for ayah in range(first, last + 1):
                    text = "کٓھٰیٰعٓصٓ۔" if (surah, ayah) == (19, 1) else f"آیت {surah} {ayah}"
                    rows.append({"surah": surah, "ayah": ayah, "urdu": text})
        unique = {(int(row["surah"]), int(row["ayah"])): row for row in rows}
        return list(unique.values())

    def test_passages_are_exact_and_cover_final_adjudications(self) -> None:
        passages = _passage_payloads(self._rows())
        audit = next(item for item in passages if item["passage_id"] == "final-adjudications")
        self.assertIn("17:7", audit["refs"])
        self.assertIn("45:11", audit["refs"])
        self.assertIn("53:20", audit["refs"])
        self.assertIn("66:5", audit["refs"])
        maryam = next(item for item in passages if item["passage_id"] == "maryam-names")
        self.assertIn("کٓھٰیٰعٓصٓ۔", maryam["text"])
        self.assertIn("کاف، ہا، یا، عین، صاد۔", maryam["speech_text"])

    def test_estimate_stays_below_pilot_ceiling(self) -> None:
        passages = _passage_payloads(self._rows())
        estimate = _estimate(passages, 746_387)
        self.assertLess(estimate["pilot"]["standard_usd"], MAX_ESTIMATED_STANDARD_COST_USD)
        self.assertGreater(estimate["full_book"]["standard_usd"], 20)

    def test_recovery_rejects_non_empty_audio_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(
                root / "RUN.json",
                {
                    "jobs": [
                        {
                            "status": "failed",
                            "attempts": 1,
                            "last_error": "HTTP 429 quota",
                            "raw_path": str(root / "raw.wav"),
                            "normalized_path": str(root / "clip.mp3"),
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(Exception, "not eligible"):
                recover_empty_audio(root)


if __name__ == "__main__":
    unittest.main()
