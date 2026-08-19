from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quran_translate.production_packets import ProductionUnit, atomic_json
from quran_translate.urdu_draft_recovery import (
    RecoveryProviderBlocked,
    _recover_one,
    freeze_targets,
    retry_exhausted_rate_limit,
)
from quran_translate.urdu_production import BudgetLedger
from quran_translate.urdu_translation_bakeoff import file_hash


class UrduDraftRecoveryTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[ProductionUnit, dict[str, object]]:
        unit = ProductionUnit("s001_001_001", 1, 1, 1, 1, 1, 1)
        unit_root = root / "units" / unit.unit_id
        (root / "evidence").mkdir(parents=True)
        (root / "evidence" / f"{unit.unit_id}.md").write_text(
            "Evidence packet", encoding="utf-8"
        )
        path = unit_root / "draft.json"
        atomic_json(
            path,
            {
                "version": "urdu-production-stage-v1",
                "stage": "draft",
                "unit_id": unit.unit_id,
                "input_hash": "placeholder",
                "status": "failed",
                "attempts": 2,
                "usage": {},
                "errors": ["Provider HTTP 403: Key limit exceeded (total limit)"],
                "raw_text": "",
                "raw_response": None,
            },
        )
        return unit, {"unit_id": unit.unit_id, "failed_summary_sha256": file_hash(path)}

    def test_recovery_preserves_failure_and_reuses_success(self) -> None:
        calls = 0

        def provider(*_args):
            nonlocal calls
            calls += 1
            result = {
                "ayahs": [{"ayah": 1, "urdu": "اللہ ایک ہے۔", "review_flags": []}]
            }
            return json.dumps(result, ensure_ascii=False), {"cost": 0.01}, {"ok": True}

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            unit, target = self._fixture(base)
            verses = {(1, 1): "بِسْمِ اللَّهِ"}
            bismillah = {1: None}
            from quran_translate.urdu_draft_recovery import _draft_input_hash

            _, _, input_hash = _draft_input_hash(base, unit, verses, bismillah)
            path = base / "units" / unit.unit_id / "draft.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["input_hash"] = input_hash
            atomic_json(path, document)
            target["failed_summary_sha256"] = file_hash(path)
            with patch.dict(
                "quran_translate.urdu_draft_recovery.PROVIDER_CALLS",
                {"openrouter": provider},
            ):
                first = _recover_one(
                    base=base,
                    unit=unit,
                    target=target,
                    verses=verses,
                    bismillah=bismillah,
                    budget=BudgetLedger(base, 1),
                )
                second = _recover_one(
                    base=base,
                    unit=unit,
                    target=target,
                    verses=verses,
                    bismillah=bismillah,
                    budget=BudgetLedger(base, 1),
                )
            archive = path.with_name("draft-pre-recovery-FAILED.json")
            self.assertEqual(target["failed_summary_sha256"], file_hash(archive))
            self.assertEqual("complete", first["status"])
            self.assertEqual("reused", second["status"])
            self.assertEqual(1, calls)

    def test_provider_block_does_not_replace_original_failure(self) -> None:
        def blocked(*_args):
            raise RuntimeError("Key limit exceeded (total limit)")

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            unit, target = self._fixture(base)
            verses = {(1, 1): "بِسْمِ اللَّهِ"}
            bismillah = {1: None}
            from quran_translate.urdu_draft_recovery import _draft_input_hash

            _, _, input_hash = _draft_input_hash(base, unit, verses, bismillah)
            path = base / "units" / unit.unit_id / "draft.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["input_hash"] = input_hash
            atomic_json(path, document)
            target["failed_summary_sha256"] = file_hash(path)
            before = path.read_bytes()
            with patch.dict(
                "quran_translate.urdu_draft_recovery.PROVIDER_CALLS",
                {"openrouter": blocked},
            ):
                with self.assertRaises(RecoveryProviderBlocked):
                    _recover_one(
                        base=base,
                        unit=unit,
                        target=target,
                        verses=verses,
                        bismillah=bismillah,
                        budget=BudgetLedger(base, 1),
                    )
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(
                1,
                len(list(path.parent.glob("draft-recovery-provider-block-*-FAILED.json"))),
            )

    def test_frozen_targets_survive_partial_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            unit, _ = self._fixture(base)
            marker = freeze_targets(base, [unit])
            path = base / "units" / unit.unit_id / "draft.json"
            archive = path.with_name("draft-pre-recovery-FAILED.json")
            archive.write_bytes(path.read_bytes())
            atomic_json(path, {"status": "complete"})

            reused = freeze_targets(base, [unit])

            self.assertEqual(marker, reused)

    def test_controlled_rate_limit_retry_is_single_and_preserves_evidence(self) -> None:
        calls = 0

        def provider(*_args):
            nonlocal calls
            calls += 1
            result = {
                "ayahs": [{"ayah": 1, "urdu": "اللہ ایک ہے۔", "review_flags": []}]
            }
            return json.dumps(result, ensure_ascii=False), {"cost": 0.01}, {"ok": True}

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            unit, _ = self._fixture(base)
            verses = {(1, 1): "بِسْمِ اللَّهِ"}
            bismillah = {1: None}
            from quran_translate.urdu_draft_recovery import _draft_input_hash

            _, _, input_hash = _draft_input_hash(base, unit, verses, bismillah)
            draft_path = base / "units" / unit.unit_id / "draft.json"
            document = json.loads(draft_path.read_text(encoding="utf-8"))
            document["input_hash"] = input_hash
            atomic_json(draft_path, document)
            marker = freeze_targets(base, [unit])
            archive = draft_path.with_name("draft-pre-recovery-FAILED.json")
            archive.write_bytes(draft_path.read_bytes())
            rate_error = "Provider HTTP 429: rate_limit_exceeded"
            for attempt in (1, 2):
                atomic_json(
                    draft_path.with_name(
                        f"draft-recovery-attempt{attempt}-FAILED.json"
                    ),
                    {
                        "input_hash": input_hash,
                        "attempt": attempt,
                        "errors": [rate_error],
                        "usage": {},
                        "raw_text": "",
                    },
                )
            atomic_json(
                draft_path.with_name("draft-recovery-FAILED.json"),
                {
                    "input_hash": input_hash,
                    "status": "failed",
                    "errors": [rate_error, rate_error],
                },
            )
            preserved = {
                path.name: path.read_bytes()
                for path in draft_path.parent.glob("*FAILED.json")
            }
            with patch.dict(
                "quran_translate.urdu_draft_recovery.PROVIDER_CALLS",
                {"openrouter": provider},
            ), patch(
                "quran_translate.urdu_draft_recovery.RATE_LIMIT_COOLDOWN_SECONDS",
                0,
            ):
                result = retry_exhausted_rate_limit(
                    base,
                    [unit],
                    verses,
                    bismillah,
                    marker,
                    budget=BudgetLedger(base, 1),
                    unit_id=unit.unit_id,
                )

            completed = json.loads(draft_path.read_text(encoding="utf-8"))
            self.assertTrue(result["complete"])
            self.assertEqual(1, result["recovered"])
            self.assertEqual(3, completed["attempts"])
            self.assertEqual(1, calls)
            self.assertTrue((base / "DRAFT_RECOVERY_COMPLETE.json").exists())
            for name, content in preserved.items():
                self.assertEqual(content, (draft_path.parent / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
