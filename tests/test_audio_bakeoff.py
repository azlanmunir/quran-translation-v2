from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quran_translate.audio_bakeoff import (
    CANDIDATES,
    EXPECTED_FINAL_TEXT_SHA256,
    PASSAGES,
    BakeoffError,
    _blind_key,
    passage_payloads,
    prepare_bakeoff,
)


class AudioBakeoffTests(unittest.TestCase):
    def test_passages_are_release_pinned_and_representative(self) -> None:
        passages = passage_payloads()
        self.assertEqual(len(passages), 6)
        self.assertEqual({item["passage_id"] for item in passages}, {p.passage_id for p in PASSAGES})
        self.assertTrue(all(item["text_sha256"] for item in passages))
        self.assertGreater(sum(item["char_count"] for item in passages), 3500)

    def test_prepare_is_idempotent_and_contains_no_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = prepare_bakeoff(root)
            second = prepare_bakeoff(root)
            self.assertEqual(first["input_fingerprint"], second["input_fingerprint"])
            serialized = (root / "RUN.json").read_text(encoding="utf-8")
            self.assertNotIn("API_KEY", serialized)
            self.assertEqual(first["final_text_sha256"], EXPECTED_FINAL_TEXT_SHA256)
            self.assertEqual(len(first["jobs"]), len(PASSAGES) * len(CANDIDATES))

    def test_prepare_refuses_input_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state = prepare_bakeoff(root)
            state["input_fingerprint"] = "stale"
            (root / "RUN.json").write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(BakeoffError, "mixed-version"):
                prepare_bakeoff(root)

    def test_blind_key_is_stable_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate_ids = [candidate.candidate_id for candidate in CANDIDATES]
            first = _blind_key(root, candidate_ids)
            second = _blind_key(root, candidate_ids)
            self.assertEqual(first, second)
            self.assertEqual(len(set(first.values())), len(candidate_ids))
            self.assertEqual((root / "PRIVATE_BLIND_KEY.json").stat().st_mode & 0o777, 0o600)

    def test_release_hash_guard_fails_closed(self) -> None:
        with patch(
            "quran_translate.audio_bakeoff.EXPECTED_FINAL_TEXT_SHA256",
            "0" * 64,
        ):
            with self.assertRaisesRegex(BakeoffError, "approved v2.4.1"):
                passage_payloads()


if __name__ == "__main__":
    unittest.main()
