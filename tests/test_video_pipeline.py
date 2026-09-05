from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quran_translate.config import file_sha256, text_sha256
from quran_translate.video_pipeline import (
    VideoPipelineError,
    build_narration_manifest,
    build_video_catalog_plan,
)


class NarrationManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.listening = self.root / "quran-listening-edition.json"
        self.run = self.root / "RUN.json"
        self.output = self.root / "NARRATION_MANIFEST.json"
        self.rows = [
            {
                "ref": "81:1",
                "surah": 81,
                "ayah": 1,
                "surah_name_en": "At-Takwir",
                "translation": "When the sun is folded up,",
            },
            {
                "ref": "81:2",
                "surah": 81,
                "ayah": 2,
                "surah_name_en": "At-Takwir",
                "translation": "and when the stars fall away,",
            },
        ]
        self.listening.write_text(
            json.dumps({"ayahs": self.rows}, ensure_ascii=False), encoding="utf-8"
        )
        transcript = (
            "Surah 81. At-Takwir.\n\nWhen the sun is folded up,\n"
            "and when the stars fall away,"
        )
        input_path = self.inputs / "0001-81_1-81_2.txt"
        input_path.write_text(transcript, encoding="utf-8")
        final_hash = text_sha256(
            "\n".join(f"{row['ref']}\t{row['translation']}" for row in self.rows)
        )
        self.state = {
            "version": "quran-audio-production-run-v1",
            "audio_run_id": "test-audio",
            "release_version": "test-release",
            "final_text_sha256": final_hash,
            "input_fingerprint": "f" * 64,
            "jobs": [
                {
                    "chunk_index": 1,
                    "job_id": "test-audio-0001",
                    "start_ref": "81:1",
                    "end_ref": "81:2",
                    "ayah_count": 2,
                    "juz_number": 30,
                    "surah_number": 81,
                    "surah_name": "At-Takwir",
                    "input_path": "/old/machine/inputs/0001-81_1-81_2.txt",
                    "text_sha256": text_sha256(transcript),
                    "char_count": len(transcript),
                    "master_mp3_path": "/old/machine/masters-mp3/0001-81_1-81_2.mp3",
                    "master_mp3_sha256": "a" * 64,
                    "duration_seconds": 12.5,
                }
            ],
        }
        self.run.write_text(json.dumps(self.state), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build(self) -> dict:
        return build_narration_manifest(
            run_path=self.run,
            inputs_dir=self.inputs,
            listening_edition_path=self.listening,
            output_path=self.output,
        )

    def test_builds_portable_manifest_with_exact_spans(self) -> None:
        payload = self.build()
        self.assertEqual(payload["totals"]["ayahs"], 2)
        self.assertEqual(payload["jobs"][0]["input_path"], "inputs/0001-81_1-81_2.txt")
        self.assertEqual(
            payload["jobs"][0]["master_mp3_path"],
            "masters-mp3/0001-81_1-81_2.mp3",
        )
        spans = payload["jobs"][0]["spans"]
        self.assertEqual([span["kind"] for span in spans], ["surah_announcement", "ayah", "ayah"])
        transcript = (self.inputs / "0001-81_1-81_2.txt").read_text(encoding="utf-8")
        for span in spans:
            self.assertEqual(
                transcript[span["start_char"] : span["end_char"]], span["text"]
            )
        self.assertEqual(payload["source_artifacts"]["run_json"]["sha256"], file_sha256(self.run))

    def test_refuses_changed_narration_script(self) -> None:
        path = self.inputs / "0001-81_1-81_2.txt"
        path.write_text(path.read_text(encoding="utf-8") + " drift", encoding="utf-8")
        with self.assertRaisesRegex(VideoPipelineError, "does not reproduce release"):
            self.build()

    def test_refuses_audio_release_hash_mismatch(self) -> None:
        self.state["final_text_sha256"] = "0" * 64
        self.run.write_text(json.dumps(self.state), encoding="utf-8")
        with self.assertRaisesRegex(VideoPipelineError, "text hashes disagree"):
            self.build()

    def test_catalog_plan_reuses_each_segment_for_both_catalogs(self) -> None:
        self.state["jobs"][0]["juz_number"] = 1
        self.run.write_text(json.dumps(self.state), encoding="utf-8")
        manifest = self.build()
        plan = build_video_catalog_plan(
            narration_manifest=manifest,
            output_path=self.root / "CATALOG_PLAN.json",
            expected_juz_count=1,
            expected_surah_count=0,
        )
        self.assertEqual(plan["totals"], {"segments": 1, "juz_videos": 1, "surah_videos": 1})
        self.assertEqual(plan["juz"][0]["segment_indices"], [1])
        self.assertEqual(plan["surahs"][0]["segment_indices"], [1])
        self.assertTrue(plan["invariants"]["each_segment_rendered_once"])

    def test_catalog_plan_refuses_cross_surah_chunk(self) -> None:
        manifest = self.build()
        manifest["jobs"][0]["spans"].append(
            {
                "kind": "ayah",
                "ref": "82:1",
                "surah": 82,
                "ayah": 1,
                "start_char": 0,
                "end_char": 1,
                "text": "x",
            }
        )
        with self.assertRaisesRegex(VideoPipelineError, "crosses a Surah boundary"):
            build_video_catalog_plan(
                narration_manifest=manifest,
                output_path=self.root / "CATALOG_PLAN.json",
                expected_juz_count=1,
                expected_surah_count=1,
            )
