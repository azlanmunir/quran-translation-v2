from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from quran_translate.config import file_sha256
from quran_translate.video_alignment import (
    AlignmentError,
    compare_alignments,
    normalize_forced_alignment,
    normalize_whisper_alignment,
    request_forced_alignment,
    write_alignment_review,
)


class AlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.audio = self.root / "audio.mp3"
        self.audio.write_bytes(b"fake-audio")
        self.transcript = "First words.\nSecond words."
        self.spans = [
            {
                "kind": "ayah",
                "ref": "1:1",
                "surah": 1,
                "ayah": 1,
                "start_char": 0,
                "end_char": 12,
                "text": "First words.",
            },
            {
                "kind": "ayah",
                "ref": "1:2",
                "surah": 1,
                "ayah": 2,
                "start_char": 13,
                "end_char": 26,
                "text": "Second words.",
            },
        ]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_normalizes_whisper_words_to_ayah_spans(self) -> None:
        raw = {
            "segments": [
                {
                    "words": [
                        {"word": " First", "start": 0.1, "end": 0.4, "probability": 0.9},
                        {"word": " words.", "start": 0.4, "end": 0.8, "probability": 0.9},
                        {"word": " Second", "start": 1.0, "end": 1.4, "probability": 0.9},
                        {"word": " words.", "start": 1.4, "end": 1.8, "probability": 0.9},
                    ]
                }
            ]
        }
        raw_path = self.root / "whisper.json"
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        result = normalize_whisper_alignment(
            raw_payload=raw,
            transcript=self.transcript,
            spans=self.spans,
            audio_path=self.audio,
            raw_path=raw_path,
            output_path=self.root / "normalized-whisper.json",
        )
        self.assertEqual(result["metrics"]["exact_word_coverage"], 1.0)
        self.assertEqual(result["spans"][0]["start"], 0.1)
        self.assertEqual(result["spans"][1]["end"], 1.8)

    def test_normalizes_forced_characters_to_ayah_spans(self) -> None:
        characters = []
        for index, char in enumerate(self.transcript):
            characters.append({"text": char, "start": index / 10, "end": (index + 1) / 10})
        raw = {"characters": characters, "words": [], "loss": 0.02}
        raw_path = self.root / "forced.json"
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        result = normalize_forced_alignment(
            raw_payload=raw,
            transcript=self.transcript,
            spans=self.spans,
            audio_path=self.audio,
            raw_path=raw_path,
            output_path=self.root / "normalized-forced.json",
        )
        self.assertEqual(result["spans"][0]["start"], 0.0)
        self.assertEqual(result["spans"][1]["end"], 2.6)

    def test_forced_alignment_refuses_transcript_drift(self) -> None:
        raw = {"characters": [{"text": "x", "start": 0, "end": 1}]}
        raw_path = self.root / "forced.json"
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(AlignmentError, "do not reproduce"):
            normalize_forced_alignment(
                raw_payload=raw,
                transcript=self.transcript,
                spans=self.spans,
                audio_path=self.audio,
                raw_path=raw_path,
                output_path=self.root / "normalized-forced.json",
            )

    def test_compares_only_identical_audio_and_transcript(self) -> None:
        first = {
            "engine": "one",
            "audio_sha256": "a",
            "transcript_sha256": "t",
            "spans": [{"kind": "ayah", "ref": "1:1", "surah": 1, "start": 0, "end": 1}],
        }
        second = {
            "engine": "two",
            "audio_sha256": "a",
            "transcript_sha256": "t",
            "spans": [{"kind": "ayah", "ref": "1:1", "surah": 1, "start": 0.2, "end": 1.1}],
        }
        result = compare_alignments(
            first=first, second=second, output_path=self.root / "comparison.json"
        )
        self.assertEqual(result["metrics"]["spans"], 1)
        self.assertEqual(result["metrics"]["max_boundary_delta_seconds"], 0.2)

    def test_writes_portable_boundary_review(self) -> None:
        sha = file_sha256(self.audio)
        first = {
            "engine": "engine-a",
            "audio_sha256": sha,
            "transcript_sha256": "text-sha",
            "spans": [
                {
                    "kind": "ayah",
                    "ref": "81:1",
                    "surah": 81,
                    "text": "When the sun is folded up,",
                    "start": 1.0,
                    "end": 2.0,
                }
            ],
        }
        second = {
            **first,
            "engine": "engine-b",
            "spans": [{**first["spans"][0], "start": 1.4, "end": 2.3}],
        }
        comparison = compare_alignments(
            first=first,
            second=second,
            output_path=self.root / "comparison.json",
        )
        payload = write_alignment_review(
            first=first,
            second=second,
            comparison=comparison,
            audio_path=self.audio,
            output_dir=self.root / "review",
        )
        self.assertEqual(payload["review_row_count"], 1)
        html = (self.root / "review" / "review.html").read_text(encoding="utf-8")
        self.assertIn("A start", html)
        self.assertIn("When the sun is folded up", html)
        self.assertEqual(file_sha256(self.root / "review" / "audio.mp3"), sha)

    @mock.patch("quran_translate.video_alignment.requests.post")
    def test_forced_alignment_wraps_network_failures(self, post: mock.Mock) -> None:
        post.side_effect = requests.ConnectionError("connection dropped")
        with self.assertRaisesRegex(AlignmentError, "network failure"):
            request_forced_alignment(
                audio_path=self.audio,
                transcript=self.transcript,
                api_key="test-key",
                output_path=self.root / "forced.json",
            )
