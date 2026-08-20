from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from quran_translate.urdu_audio_production import (
    HARD_ESTIMATED_BATCH_COST_USD,
    MODEL_ID,
    PRONUNCIATION_OVERRIDES,
    VOICE_ID,
    _build_units,
    _generation_config,
    _make_shards,
    _request_line,
    _speech_ayah,
    prepare,
)


class UrduAudioProductionTests(unittest.TestCase):
    def test_frozen_units_cover_every_ayah_once(self) -> None:
        units = _build_units()
        self.assertEqual(len(units), 343)
        self.assertEqual(sum(len(unit["refs"]) for unit in units), 6_236)
        self.assertEqual(units[0]["refs"][0], "1:1")
        self.assertEqual(units[-1]["refs"][-1], "114:6")
        self.assertLess(max(unit["speech_characters"] for unit in units), 6_000)
        self.assertTrue(all(unit["juz"] for unit in units))

    def test_pronunciation_overrides_are_exact_guarded(self) -> None:
        self.assertEqual(
            _speech_ayah("19:1", "کٓھٰیٰعٓصٓ۔"),
            "کاف، ہا، یا، عین، صاد۔",
        )
        self.assertEqual(
            _speech_ayah("11:1", "الر — یہ ایک کتاب ہے"),
            "الف، لام، را۔ یہ ایک کتاب ہے",
        )
        with self.assertRaisesRegex(Exception, "source guard"):
            _speech_ayah("19:1", "بدلا ہوا متن")
        self.assertGreaterEqual(len(PRONUNCIATION_OVERRIDES), 25)

    def test_batch_request_uses_audio_and_charon(self) -> None:
        unit = _build_units()[0]
        row = json.loads(_request_line(unit))
        self.assertEqual(row["key"], unit["unit_id"])
        config = row["request"]["generationConfig"]
        self.assertEqual(config["responseModalities"], ["AUDIO"])
        self.assertEqual(
            config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
            VOICE_ID,
        )
        self.assertEqual(MODEL_ID, "gemini-2.5-pro-preview-tts")

    def test_shards_isolate_canary_and_respect_character_target(self) -> None:
        units = _build_units()
        shards = _make_shards(units)
        self.assertEqual(len(shards[0]), 1)
        self.assertEqual(shards[0][0]["unit_index"], 2)
        flattened = [unit["unit_id"] for shard in shards for unit in shard]
        self.assertEqual(flattened, [unit["unit_id"] for unit in units[1:]])
        seeded_shards = _make_shards(units, seeded_units=2)
        seeded_flattened = [unit["unit_id"] for shard in seeded_shards for unit in shard]
        self.assertEqual(seeded_flattened, [unit["unit_id"] for unit in units[2:]])

    def test_prepare_is_immutable_and_below_cost_ceiling(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = prepare(root)
            self.assertEqual(len(state["jobs"]), 343)
            self.assertEqual(state["jobs"][0]["source"], "approved_pilot_reuse")
            self.assertEqual(state["jobs"][0]["status"], "complete")
            self.assertEqual(prepare(root)["input_fingerprint"], state["input_fingerprint"])
            estimate = json.loads((root / "COST_GUARD.json").read_text())
            self.assertLessEqual(
                estimate["estimated_batch_cost_usd"], HARD_ESTIMATED_BATCH_COST_USD
            )

    def test_generation_config_sets_urdu_language(self) -> None:
        config = _generation_config()
        self.assertEqual(config["speechConfig"]["languageCode"], "ur")


if __name__ == "__main__":
    unittest.main()
