from __future__ import annotations

import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quran_translate.audio_production import (
    DEFAULT_AUDIO_RUN_ID,
    EXPECTED_FINAL_TEXT_SHA256,
    AudioProductionError,
    _fixed_duration_groups,
    _juz_assignment,
    _pitch_and_encode,
    _provider_blocker_status,
    plan_chunks,
    prepare_audio_production,
)
from quran_translate.elevenlabs_tts import synthesize_text


class AudioProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = plan_chunks()

    def test_plan_is_release_pinned_complete_and_within_v3_limit(self) -> None:
        chunks = self.plan["chunks"]
        self.assertEqual(self.plan["ayah_count"], 6236)
        self.assertEqual(len(chunks), 313)
        self.assertEqual(chunks[0]["start_ref"], "1:1")
        self.assertEqual(chunks[-1]["end_ref"], "114:6")
        self.assertLessEqual(max(chunk["char_count"] for chunk in chunks), 3200)
        self.assertEqual(sum(chunk["ayah_count"] for chunk in chunks), 6236)

    def test_chunks_never_cross_surah_or_juz(self) -> None:
        for chunk in self.plan["chunks"]:
            start_surah = int(chunk["start_ref"].split(":", 1)[0])
            end_surah = int(chunk["end_ref"].split(":", 1)[0])
            self.assertEqual(start_surah, end_surah)
        starts = {juz["start_ref"] for juz in self.plan["juzs"]}
        chunk_starts = {chunk["start_ref"] for chunk in self.plan["chunks"]}
        self.assertTrue(starts.issubset(chunk_starts))

    def test_canonical_juz_ranges_cover_every_ayah_once(self) -> None:
        from quran_translate.audio_production import _release_rows

        rows, _ = _release_rows()
        assignment, juzs = _juz_assignment(rows)
        self.assertEqual(len(assignment), 6236)
        self.assertEqual(len(juzs), 30)
        self.assertEqual(juzs[0], {"juz": 1, "start_ref": "1:1", "end_ref": "2:141", "verse_count": 148})
        self.assertEqual(juzs[-1], {"juz": 30, "start_ref": "78:1", "end_ref": "114:6", "verse_count": 564})
        self.assertEqual(assignment["2:142"], 2)
        self.assertEqual(assignment["78:1"], 30)

    def test_prepare_is_immutable_and_carries_selected_treatment(self) -> None:
        def tool_version(command: list[str]) -> str:
            return "4.0.0" if command[0] == "rubberband" else "ffmpeg version test"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch("quran_translate.audio_production._run_root", return_value=root),
                mock.patch(
                    "quran_translate.audio_production._tool_version",
                    side_effect=tool_version,
                ),
            ):
                state = prepare_audio_production(audio_run_id=DEFAULT_AUDIO_RUN_ID)
                resumed = prepare_audio_production(audio_run_id=DEFAULT_AUDIO_RUN_ID)
            self.assertEqual(state["input_fingerprint"], resumed["input_fingerprint"])
            self.assertEqual(state["final_text_sha256"], EXPECTED_FINAL_TEXT_SHA256)
            self.assertEqual(state["voice_id"], "lWDDHwXsJXJM7nv2YgHY")
            self.assertEqual(state["model_id"], "eleven_v3")
            self.assertEqual(state["source_format"], "pcm_44100")
            self.assertEqual(state["pitch_processing"]["pitch_semitones"], -1.25)
            self.assertEqual(state["pitch_processing"]["global_time_stretch"], 1.0)
            self.assertEqual(len(state["jobs"]), 313)

            state_path = root / "RUN.json"
            tampered = json.loads(state_path.read_text(encoding="utf-8"))
            tampered["input_fingerprint"] = "tampered"
            state_path.write_text(json.dumps(tampered), encoding="utf-8")
            with (
                mock.patch("quran_translate.audio_production._run_root", return_value=root),
                mock.patch(
                    "quran_translate.audio_production._tool_version",
                    side_effect=tool_version,
                ),
                self.assertRaises(AudioProductionError),
            ):
                prepare_audio_production(audio_run_id=DEFAULT_AUDIO_RUN_ID)

    def test_fixed_tracks_cut_only_between_master_chunks(self) -> None:
        jobs = [
            {"job_id": "a", "duration_seconds": 1000},
            {"job_id": "b", "duration_seconds": 1000},
            {"job_id": "c", "duration_seconds": 1000},
            {"job_id": "d", "duration_seconds": 1000},
        ]
        groups = _fixed_duration_groups(jobs, 2400)
        self.assertEqual([[job["job_id"] for job in group] for group in groups], [["a", "b"], ["c", "d"]])

    def test_provider_blockers_are_classified_for_operator_action(self) -> None:
        self.assertEqual(
            _provider_blocker_status("subscription_required output_format_not_allowed"),
            "provider_blocked",
        )
        self.assertEqual(_provider_blocker_status("credits exhausted"), "quota_paused")
        self.assertEqual(_provider_blocker_status("401 unauthorized"), "authentication_blocked")
        self.assertEqual(_provider_blocker_status("payment failed"), "billing_blocked")

    def test_eleven_v3_omits_unsupported_context_fields(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"audio"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "clip.pcm"
            with mock.patch(
                "quran_translate.elevenlabs_tts.urllib.request.urlopen",
                return_value=response,
            ) as urlopen:
                synthesize_text(
                    text="Current passage.",
                    voice_id="voice-id",
                    output_path=output,
                    api_key="test-key",
                    model_id="eleven_v3",
                    output_format="pcm_44100",
                    previous_text="Previous passage.",
                    next_text="Following passage.",
                )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertNotIn("previous_text", payload)
        self.assertNotIn("next_text", payload)
        self.assertEqual(payload["model_id"], "eleven_v3")

    def test_lossless_pitch_and_mp3_processing_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = [
                int(7000 * math.sin(2 * math.pi * 220 * index / 44100))
                for index in range(44100)
            ]
            raw = root / "source.pcm"
            raw.write_bytes(struct.pack(f"<{len(samples)}h", *samples))
            job = {
                "job_id": "processing-smoke",
                "raw_path": str(raw),
                "source_wav_path": str(root / "work" / "source.wav"),
                "master_wav_path": str(root / "masters-wav" / "master.wav"),
                "master_mp3_path": str(root / "masters-mp3" / "master.mp3"),
            }
            result = _pitch_and_encode(job, "pcm_44100")
            self.assertGreater(result["bytes"], 1000)
            self.assertAlmostEqual(
                result["source_duration_seconds"], result["duration_seconds"], delta=0.05
            )
            self.assertEqual(len(result["master_wav_sha256"]), 64)
            self.assertEqual(len(result["master_mp3_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
