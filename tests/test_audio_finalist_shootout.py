from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
