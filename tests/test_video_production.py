from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quran_translate.video_production import _catalog_chapters, _metadata, write_catalog_srt
from quran_translate.video_render import render_video


class VideoProductionTests(unittest.TestCase):
    @patch("quran_translate.video_render.file_sha256", return_value="hash")
    @patch(
        "quran_translate.video_render._loudness_measure",
        return_value={
            "input_i": -18.0,
            "input_tp": -2.0,
            "input_lra": 7.0,
            "input_thresh": -28.0,
            "target_offset": 0.0,
        },
    )
    @patch("quran_translate.video_render.subprocess.run")
    def test_render_caps_the_muxed_output_duration(
        self,
        run_mock,
        _loudness_mock,
        _hash_mock,
    ) -> None:
        display = {
            "selection": {"clip_start": 0.0, "duration": 10.0},
            "events": [{"start": 0.0, "end": 10.0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.mp3"
            output = root / "video.mp4"
            render_video(
                display=display,
                frames=[root / "frame.png"],
                audio_path=audio,
                output_path=output,
            )

        command = run_mock.call_args.args[0]
        audio_index = command.index(str(audio))
        duration_caps = [index for index, value in enumerate(command) if value == "-t"]
        self.assertEqual(len(duration_caps), 2)
        self.assertGreater(duration_caps[-1], audio_index)
        self.assertEqual(command[duration_caps[-1] + 1], "10.000")
        self.assertEqual(command[-2], "-shortest")
        self.assertEqual(command[-1], str(output))

    def test_catalog_srt_offsets_each_canonical_segment(self) -> None:
        displays = [
            {
                "events": [
                    {"start": 0.0, "end": 2.0, "caption": "First ayah."},
                    {"start": 2.0, "end": 5.0, "caption": "Second ayah."},
                ]
            },
            {
                "events": [
                    {"start": 0.0, "end": 3.0, "caption": "Third ayah."},
                ]
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "catalog.srt"
            write_catalog_srt(
                displays=displays,
                segment_durations=[5.1, 3.0],
                output_path=output,
            )
            content = output.read_text(encoding="utf-8")
        self.assertIn("00:00:05,100 --> 00:00:08,100", content)
        self.assertIn("Third ayah.", content)

    def test_juz_chapters_follow_surah_boundaries(self) -> None:
        jobs = [
            {
                "surah_number": 1,
                "surah_name": "Al-Fatihah",
                "start_ref": "1:1",
            },
            {
                "surah_number": 2,
                "surah_name": "Al-Baqarah",
                "start_ref": "2:1",
            },
            {
                "surah_number": 3,
                "surah_name": "Ali Imran",
                "start_ref": "3:1",
            },
        ]
        displays = [
            {"events": [{"start": 0.0, "end": 10.0, "active_ref": "1:1"}]},
            {"events": [{"start": 0.0, "end": 10.0, "active_ref": "2:1"}]},
            {"events": [{"start": 0.0, "end": 10.0, "active_ref": "3:1"}]},
        ]
        chapters = _catalog_chapters(
            kind="juz",
            jobs=jobs,
            displays=displays,
            segment_durations=[12.0, 14.0, 16.0],
        )
        self.assertEqual([row["timestamp"] for row in chapters], ["0:00", "0:12", "0:26"])
        self.assertEqual(chapters[1]["label"], "Surah 2: Al-Baqarah")

    def test_juz_metadata_leads_with_para_and_keeps_juz_alias(self) -> None:
        metadata = _metadata(
            kind="juz",
            entry={"number": 2},
            jobs=[{"start_ref": "2:142", "end_ref": "2:252"}],
            chapters=[],
            listening={},
        )

        self.assertEqual(
            metadata["title"],
            "Quran Para 2 of 30 (Juz 2) | English Listening Edition",
        )
        self.assertEqual(metadata["thumbnail_label"], "PARA 02 OF 30")
        self.assertEqual(metadata["playlist"], "The Quran by Para - English Listening Edition")
        self.assertTrue(metadata["description"].startswith("Para 2 of 30 (Juz 2)"))
        self.assertNotIn("AI", metadata["description"])

    def test_short_catalog_does_not_emit_invalid_chapters(self) -> None:
        chapters = _catalog_chapters(
            kind="surah",
            jobs=[{"surah_number": 112, "surah_name": "Al-Ikhlas", "start_ref": "112:1"}],
            displays=[
                {
                    "events": [
                        {"start": 0.0, "end": 2.0, "active_ref": "112:1"},
                        {"start": 2.0, "end": 4.0, "active_ref": "112:2"},
                    ]
                }
            ],
            segment_durations=[8.0],
        )
        self.assertEqual(chapters, [])
