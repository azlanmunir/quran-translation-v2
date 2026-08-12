import unittest

from quran_translate.audio_google_voice_audition import MODEL_ID, PASSAGE_ID, VOICES


class GoogleMaleVoiceAuditionTests(unittest.TestCase):
    def test_audition_is_compact_and_male_only(self) -> None:
        self.assertEqual(PASSAGE_ID, "consolation")
        self.assertEqual(MODEL_ID, "gemini-2.5-pro-preview-tts")
        self.assertEqual(
            [voice.voice_id for voice in VOICES],
            ["Charon", "Algenib", "Algieba", "Schedar"],
        )
        self.assertTrue(all("(male)" in voice.voice_label for voice in VOICES))


if __name__ == "__main__":
    unittest.main()
