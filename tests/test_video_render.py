from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from quran_translate.video_render import (
    build_display_events,
    build_youtube_chapters,
    format_youtube_chapters,
    validate_video,
    validate_srt_identity,
    write_mobile_review,
    write_srt,
)


class DisplayEventTests(unittest.TestCase):
    def _video_probe(self, duration: float) -> Mock:
        return Mock(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": 1920,
                            "height": 1080,
                            "pix_fmt": "yuv420p",
                            "codec_name": "h264",
                        },
                        {
                            "codec_type": "audio",
                            "codec_name": "aac",
                            "sample_rate": "44100",
                            "channels": 1,
                        },
                    ],
                    "format": {"duration": str(duration)},
                }
            )
        )

    @patch(
        "quran_translate.video_render._loudness_measure",
        return_value={"input_i": -18.38, "input_tp": -1.5},
    )
    @patch("quran_translate.video_render.subprocess.run")
    def test_duration_contract_allows_visual_tail_without_speech_cutoff(
        self, run_mock, _loudness_mock
    ) -> None:
        run_mock.side_effect = [self._video_probe(229.2), Mock()]
        checks = validate_video(
            Path("video.mp4"),
            229.42,
            minimum_content_duration=229.17,
        )
        self.assertEqual(checks["duration_delta_seconds"], 0.22)
        self.assertEqual(checks["content_end_margin_seconds"], 0.03)
        self.assertEqual(checks["loudness_tolerance_lufs"], 0.5)
        self.assertTrue(checks["decode_passed"])

    @patch(
        "quran_translate.video_render._loudness_measure",
        return_value={"input_i": -18.6, "input_tp": -1.5},
    )
    @patch("quran_translate.video_render.subprocess.run")
    def test_loudness_contract_rejects_material_target_miss(
        self, run_mock, _loudness_mock
    ) -> None:
        run_mock.return_value = self._video_probe(20.0)
        with self.assertRaisesRegex(Exception, "outside the -18 LUFS"):
            validate_video(Path("video.mp4"), 20.0)

    @patch("quran_translate.video_render.subprocess.run")
    def test_duration_contract_rejects_aligned_speech_cutoff(self, run_mock) -> None:
        run_mock.return_value = self._video_probe(229.2)
        with self.assertRaisesRegex(Exception, "cuts off aligned speech"):
            validate_video(
                Path("video.mp4"),
                229.42,
                minimum_content_duration=229.3,
            )

    def alignment(self):
        texts = ["One short line.", "Another short line.", "A third short line."]
        spans = []
        words = []
        cursor = 0
        time = 1.0
        for index, text in enumerate(texts, start=1):
            start_char = cursor
            for word in text.rstrip(".").split():
                offset = text.find(word)
                words.append(
                    {
                        "text": word,
                        "start_char": start_char + offset,
                        "end_char": start_char + offset + len(word),
                        "start": time,
                        "end": time + 0.3,
                    }
                )
                time += 0.35
            spans.append(
                {
                    "kind": "ayah",
                    "ref": f"81:{index}",
                    "surah": 81,
                    "ayah": index,
                    "start_char": start_char,
                    "end_char": start_char + len(text),
                    "text": text,
                    "start": time - 1.0,
                    "end": time,
                }
            )
            cursor += len(text) + 1
            time += 0.4
        return {
            "engine": "test",
            "audio_sha256": "a",
            "transcript_sha256": "t",
            "spans": spans,
            "words": words,
        }

    def test_groups_short_ayahs_but_advances_active_highlight(self):
        display = build_display_events(self.alignment(), start_ref="81:1", end_ref="81:3")
        self.assertEqual(len(display["events"]), 3)
        self.assertEqual([len(event["lines"]) for event in display["events"]], [3, 3, 3])
        self.assertEqual(
            [event["active_ref"] for event in display["events"]],
            ["81:1", "81:2", "81:3"],
        )

    def test_srt_uses_same_timeline(self):
        display = build_display_events(self.alignment(), start_ref="81:1", end_ref="81:3")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.srt"
            write_srt(display, path)
            content = path.read_text(encoding="utf-8")
            self.assertIn("00:00:00,120", content)
            self.assertIn("One short line.", content)
            self.assertIn("Another short line.", content)
            self.assertTrue(validate_srt_identity(display, path)["timeline_identity"])

    def test_srt_identity_refuses_drift(self):
        display = build_display_events(self.alignment(), start_ref="81:1", end_ref="81:3")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.srt"
            write_srt(display, path)
            path.write_text(path.read_text(encoding="utf-8").replace("One", "One drifted", 1))
            with self.assertRaisesRegex(Exception, "drifted text"):
                validate_srt_identity(display, path)

    def test_mobile_review_includes_first_and_last_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for index in range(15):
                path = root / f"frame-{index + 1:04d}.png"
                Image.new("RGB", (1920, 1080), (index, index, index)).save(path)
                frames.append(path)
            payload = write_mobile_review(
                frames=frames,
                output_dir=root / "mobile",
                max_samples=5,
            )
            self.assertEqual(payload["sample_count"], 5)
            self.assertEqual(payload["previews"][0]["frame_index"], 1)
            self.assertEqual(payload["previews"][-1]["frame_index"], 15)
            self.assertTrue((root / "mobile" / "contact-sheet.png").exists())

    def test_youtube_chapters_are_ayah_snapped_and_valid(self):
        spans = []
        for index, start in enumerate((0, 300, 610, 930, 1235), start=1):
            spans.append(
                {
                    "kind": "ayah",
                    "ref": f"2:{index}",
                    "surah": 2,
                    "ayah": index,
                    "start": float(start),
                    "end": float(start + 60),
                }
            )
        chapters = build_youtube_chapters(
            alignment={"spans": spans},
            start_ref="2:1",
            end_ref="2:5",
            catalog="surah",
        )
        self.assertEqual([row["ref"] for row in chapters], ["2:1", "2:3", "2:5"])
        self.assertEqual(chapters[0]["timestamp"], "0:00")
        self.assertIn("10:10 Ayah 2:3", format_youtube_chapters(chapters))

    def test_short_video_does_not_emit_invalid_chapter_set(self):
        chapters = build_youtube_chapters(
            alignment={"spans": self.alignment()["spans"]},
            start_ref="81:1",
            end_ref="81:3",
            catalog="surah",
        )
        self.assertEqual(chapters, [])
