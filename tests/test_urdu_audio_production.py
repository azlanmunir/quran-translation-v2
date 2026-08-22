from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from quran_translate.urdu_audio_production import (
    HARD_ESTIMATED_BATCH_COST_USD,
    MODEL_ID,
    PRONUNCIATION_OVERRIDES,
    VOICE_ID,
    _build_units,
    _generation_config,
    _fragment_unit,
    _make_shards,
    _request_line,
    _speech_ayah,
    prepare,
    prepare_failed_unit_recovery,
    prepare_fragment_recovery,
    prepare_smaller_fragment_recovery,
    resume_quota_wave,
)
from quran_translate.production_packets import atomic_json


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

    def test_quota_recovery_preserves_failure_and_opens_one_wave(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(
                root / "RUN.json",
                {
                    "status": "blocked",
                    "quota_recovery_waves": 0,
                    "batches": [
                        {"shard": 1, "status": "collected", "batch_id": "batches/one"},
                        {
                            "shard": 2,
                            "status": "submission_blocked",
                            "batch_id": None,
                            "last_error": "429 RESOURCE_EXHAUSTED",
                            "uploaded_file_name": "files/two",
                        },
                    ],
                },
            )
            with mock.patch(
                "quran_translate.urdu_audio_production.run", return_value={"status": "started"}
            ) as run_mock:
                result = resume_quota_wave(root, poll_seconds=1)
            self.assertEqual(result, {"status": "started"})
            updated = json.loads((root / "RUN.json").read_text())
            recovered = updated["batches"][1]
            self.assertEqual(recovered["status"], "prepared")
            self.assertEqual(len(recovered["failure_history"]), 1)
            self.assertEqual(updated["quota_recovery_waves"], 1)
            run_mock.assert_called_once_with(root, poll_seconds=1)

    def test_failed_unit_recovery_is_targeted_preserved_and_idempotent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = prepare(root)
            source = state["batches"][0]
            failed_ids = source["unit_ids"][:2]
            source.update(
                {
                    "status": "collection_blocked",
                    "batch_id": "batches/source",
                    "last_error": f"Failed units: {', '.join(failed_ids)}",
                }
            )
            jobs = {job["unit_id"]: job for job in state["jobs"]}
            for unit_id in failed_ids:
                jobs[unit_id].update(
                    {"status": "failed", "last_error": "provider item failed"}
                )
            atomic_json(root / "RUN.json", state)

            prepare_failed_unit_recovery(root)
            updated = json.loads((root / "RUN.json").read_text())
            recovery = next(
                batch for batch in updated["batches"] if batch.get("failed_unit_recovery")
            )
            self.assertEqual(recovery["unit_ids"], failed_ids)
            self.assertEqual(recovery["status"], "prepared")
            self.assertEqual(updated["batches"][0]["status"], "collected_with_failures")
            updated_jobs = {job["unit_id"]: job for job in updated["jobs"]}
            for unit_id in failed_ids:
                self.assertEqual(updated_jobs[unit_id]["status"], "pending")
                self.assertEqual(len(updated_jobs[unit_id]["failure_history"]), 1)

            later_source = next(
                batch
                for batch in updated["batches"]
                if not batch.get("failed_unit_recovery") and batch["status"] == "prepared"
            )
            later_id = later_source["unit_ids"][0]
            later_source["status"] = "collected_with_failures"
            updated_jobs[later_id].update(
                {"status": "failed", "last_error": "later provider item failed"}
            )
            atomic_json(root / "RUN.json", updated)

            prepare_failed_unit_recovery(root)
            merged = json.loads((root / "RUN.json").read_text())
            merged_recovery = next(
                batch for batch in merged["batches"] if batch.get("failed_unit_recovery")
            )
            self.assertEqual(merged_recovery["unit_ids"], failed_ids + [later_id])
            self.assertEqual(
                next(job for job in merged["jobs"] if job["unit_id"] == later_id)["status"],
                "pending",
            )

            prepare_failed_unit_recovery(root)
            unchanged = json.loads((root / "RUN.json").read_text())
            self.assertEqual(
                sum(bool(batch.get("failed_unit_recovery")) for batch in unchanged["batches"]),
                1,
            )

            first_recovery = next(
                batch for batch in unchanged["batches"] if batch.get("failed_unit_recovery")
            )
            first_recovery.update(
                {"status": "collected_with_failures", "batch_id": "batches/recovery-one"}
            )
            repeated_id = failed_ids[0]
            repeated_job = next(
                job for job in unchanged["jobs"] if job["unit_id"] == repeated_id
            )
            repeated_job.update(
                {"status": "failed", "last_error": "recovery provider item failed"}
            )
            atomic_json(root / "RUN.json", unchanged)

            prepare_failed_unit_recovery(root)
            second = json.loads((root / "RUN.json").read_text())
            recoveries = [
                batch for batch in second["batches"] if batch.get("failed_unit_recovery")
            ]
            self.assertEqual(len(recoveries), 2)
            second_recovery = next(
                batch for batch in recoveries if batch["recovery_attempt"] == 2
            )
            self.assertEqual(second_recovery["unit_ids"], [repeated_id])
            self.assertEqual(second_recovery["status"], "prepared")

            second_recovery.update(
                {"status": "collected_with_failures", "batch_id": "batches/recovery-two"}
            )
            repeated_job = next(job for job in second["jobs"] if job["unit_id"] == repeated_id)
            repeated_job.update(
                {"status": "failed", "last_error": "third provider item failure"}
            )
            atomic_json(root / "RUN.json", second)
            with self.assertRaisesRegex(Exception, "attempt ceiling"):
                prepare_failed_unit_recovery(root)

    def test_fragment_unit_preserves_ayah_order_and_reduces_request_size(self) -> None:
        unit = max(_build_units(), key=lambda item: len(item["refs"]))
        fragments = _fragment_unit(unit)
        self.assertGreater(len(fragments), 1)
        self.assertEqual(
            [ref for fragment in fragments for ref in fragment["refs"]],
            unit["refs"],
        )
        self.assertEqual(
            "\n".join(fragment["speech_text"] for fragment in fragments),
            unit["speech_text"],
        )
        self.assertTrue(
            all(len(fragment["refs"]) < len(unit["refs"]) for fragment in fragments)
        )
        self.assertEqual(
            len({fragment["fragment_id"] for fragment in fragments}),
            len(fragments),
        )

    def test_fragment_recovery_is_targeted_frozen_and_idempotent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = prepare(root)
            source = state["batches"][0]
            failed_id = source["unit_ids"][0]
            source["status"] = "collected_with_failures"
            failed_job = next(
                job for job in state["jobs"] if job["unit_id"] == failed_id
            )
            failed_job.update({"status": "failed", "last_error": "three attempts failed"})
            atomic_json(root / "RUN.json", state)

            prepare_fragment_recovery(root)
            updated = json.loads((root / "RUN.json").read_text())
            recovery = json.loads((root / "FRAGMENT_RECOVERY.json").read_text())
            fragment_batches = [
                batch for batch in updated["batches"] if batch.get("fragment_recovery")
            ]
            self.assertEqual(recovery["original_unit_ids"], [failed_id])
            self.assertTrue(fragment_batches)
            self.assertEqual(
                [
                    fragment["fragment_id"]
                    for fragment in recovery["fragments"]
                ],
                [
                    fragment_id
                    for batch in fragment_batches
                    for fragment_id in batch["unit_ids"]
                ],
            )
            recovered_job = next(
                job for job in updated["jobs"] if job["unit_id"] == failed_id
            )
            self.assertEqual(recovered_job["status"], "pending")
            self.assertEqual(len(recovered_job["failure_history"]), 1)

            prepare_fragment_recovery(root)
            unchanged = json.loads((root / "RUN.json").read_text())
            self.assertEqual(
                sum(bool(batch.get("fragment_recovery")) for batch in unchanged["batches"]),
                len(fragment_batches),
            )

    def test_smaller_fragment_recovery_reuses_success_and_splits_failures(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = prepare(root)
            first_id = state["batches"][0]["unit_ids"][0]
            first_job = next(job for job in state["jobs"] if job["unit_id"] == first_id)
            first_job.update({"status": "failed", "last_error": "full unit failed"})
            state["batches"][0]["status"] = "collected_with_failures"
            atomic_json(root / "RUN.json", state)
            prepare_fragment_recovery(root)

            state = json.loads((root / "RUN.json").read_text())
            recovery = json.loads((root / "FRAGMENT_RECOVERY.json").read_text())
            first_fragments = [
                fragment
                for fragment in recovery["fragments"]
                if fragment["original_unit_id"] == first_id
            ]
            first_fragments[0]["status"] = "complete"
            for fragment in first_fragments[1:]:
                fragment.update({"status": "failed", "last_error": "no audio"})
            second_id = next(
                job["unit_id"]
                for job in state["jobs"]
                if job["unit_id"] != first_id and job["status"] == "pending"
            )
            for job in state["jobs"]:
                if job["unit_id"] in {first_id, second_id}:
                    job.update({"status": "failed", "last_error": "provider failed"})
            for batch in state["batches"]:
                batch["status"] = "collected"
            atomic_json(root / "RUN.json", state)
            atomic_json(root / "FRAGMENT_RECOVERY.json", recovery)

            prepare_smaller_fragment_recovery(root)
            updated = json.loads((root / "RUN.json").read_text())
            smaller = json.loads((root / "FRAGMENT_RECOVERY.json").read_text())
            original_first = [
                fragment
                for fragment in smaller["fragments"]
                if fragment["fragment_id"] in {
                    item["fragment_id"] for item in first_fragments
                }
            ]
            self.assertEqual(original_first[0]["status"], "complete")
            self.assertTrue(
                all(item["status"] == "superseded" for item in original_first[1:])
            )
            micro = [
                fragment
                for fragment in smaller["fragments"]
                if fragment.get("recovery_level") == 2
            ]
            self.assertTrue(micro)
            self.assertTrue(all(fragment["status"] == "pending" for fragment in micro))
            self.assertTrue(all(len(fragment["refs"]) <= 3 for fragment in micro))
            self.assertEqual(
                {
                    fragment["original_unit_id"]
                    for fragment in micro
                },
                {first_id, second_id},
            )
            jobs = {job["unit_id"]: job for job in updated["jobs"]}
            self.assertEqual(jobs[first_id]["status"], "pending")
            self.assertEqual(jobs[second_id]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
