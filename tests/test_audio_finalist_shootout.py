from __future__ import annotations

import json
import unittest
from pathlib import Path

from quran_translate.audio_finalist_shootout import (
    FINALISTS,
    PASSAGE_IDS,
    _pitch_command,
)


class AudioFinalistShootoutTests(unittest.TestCase):
    def test_finalists_isolate_voice_and_subtle_pitch(self) -> None:
        self.assertEqual(PASSAGE_IDS, ("consolation", "oaths", "mary"))
        self.assertEqual(len(FINALISTS), 4)
        self.assertEqual(
            [f.pitch_semitones for f in FINALISTS if f.mode == "pitch_nathan"],
            [-0.75, -1.25],
        )

    def test_pitch_command_preserves_formants_with_fine_engine(self) -> None:
        command = _pitch_command(Path("in.wav"), Path("out.wav"), -0.75)
        self.assertIn("--fine", command)
        self.assertIn("--formant", command)
        self.assertEqual(command[command.index("--pitch") + 1], "-0.75")

    def test_committed_voice_selection_matches_winning_treatment(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        selection = json.loads(
            (
                project_root
                / "releases"
                / "quran-translation-v2.4.1-audio-voice.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(selection["selection"]["blind_code"], "Voice C")
        self.assertEqual(selection["narration"]["model_id"], "eleven_v3")
        self.assertEqual(
            selection["narration"]["voice_id"], "lWDDHwXsJXJM7nv2YgHY"
        )
        self.assertEqual(selection["pitch_processing"]["pitch_semitones"], -1.25)
        self.assertTrue(selection["pitch_processing"]["formant_preservation"])
        self.assertEqual(selection["pacing_policy"]["global_time_stretch"], 1.0)


if __name__ == "__main__":
    unittest.main()
