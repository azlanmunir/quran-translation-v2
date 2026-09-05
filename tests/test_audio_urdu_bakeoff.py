from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from quran_translate import audio_urdu_bakeoff as bakeoff
from quran_translate.elevenlabs_tts import synthesize_text


class UrduBakeoffTests(unittest.TestCase):
    def test_candidates_are_male_urdu_shortlist(self) -> None:
        self.assertEqual(len(bakeoff.CANDIDATES), 6)
        self.assertEqual(
            {candidate.provider for candidate in bakeoff.CANDIDATES},
            {"edge", "elevenlabs", "gemini"},
        )
        self.assertTrue(all(candidate.locale.startswith("ur") for candidate in bakeoff.CANDIDATES))

    def test_passages_exercise_urdu_and_quranic_terms(self) -> None:
        text = " ".join(passage.text for passage in bakeoff.PASSAGES)
        for token in ("قرآن", "صراطِ مستقیم", "تقویٰ", "موسیٰ", "ذمّے داری"):
            self.assertIn(token, text)

    def test_prepare_is_resumable_and_fingerprinted(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first = bakeoff.prepare(root)
            second = bakeoff.prepare(root)
            self.assertEqual(first["input_fingerprint"], second["input_fingerprint"])
            self.assertEqual(len(first["jobs"]), 18)

    def test_elevenlabs_request_includes_language_code(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return b"audio"

        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return Response()

        with TemporaryDirectory() as temp, patch(
            "quran_translate.elevenlabs_tts.urllib.request.urlopen", fake_urlopen
        ):
            synthesize_text(
                text="اردو",
                voice_id="voice",
                output_path=Path(temp) / "clip.mp3",
                api_key="test",
                model_id="eleven_v3",
                language_code="ur",
            )
        self.assertEqual(captured["body"]["language_code"], "ur")


if __name__ == "__main__":
    unittest.main()
