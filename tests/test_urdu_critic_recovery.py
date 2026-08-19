from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import quran_translate.urdu_production as urdu_production
from quran_translate.production_packets import ProductionUnit, atomic_json
from quran_translate.urdu_critic_recovery import (
    amend_quality_manifest,
    freeze_targets,
    run_revalidation,
)
from quran_translate.urdu_translation_bakeoff import file_hash


class UrduCriticRecoveryTests(unittest.TestCase):
    def test_manifest_amendment_changes_only_quality_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            manifest = {
                "version": "quran-urdu-production-v1",
                "runner_sha256": file_hash(Path(urdu_production.__file__)),
                "quality_sha256": "old-quality-hash",
                "provider_sha256": file_hash(
                    Path(urdu_production.__file__).with_name(
                        "urdu_translation_bakeoff.py"
                    )
                ),
                "inputs": {},
            }
            atomic_json(base / "MANIFEST.json", manifest)
            before = (base / "MANIFEST.json").read_bytes()

            amendment = amend_quality_manifest(base)
            repeated = amend_quality_manifest(base)
            amended = json.loads(
                (base / "MANIFEST.json").read_text(encoding="utf-8")
            )

            self.assertEqual(amendment, repeated)
            self.assertEqual(
                before,
                (base / "MANIFEST_PRE_CRITIC_ORTHOGRAPHY.json").read_bytes(),
            )
            self.assertEqual(
                file_hash(Path(urdu_production.__file__).with_name("urdu_quality.py")),
                amended.pop("quality_sha256"),
            )
            manifest.pop("quality_sha256")
            self.assertEqual(manifest, amended)

    def test_revalidates_stored_uthmani_ground_without_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            unit = ProductionUnit("s001_001_001", 1, 1, 1, 1, 1, 1)
            root = base / "units" / unit.unit_id
            draft = {
                "ayahs": [
                    {
                        "ayah": 1,
                        "urdu": "قسم ہے سورج کی اور اس کی روشنی کی۔",
                        "review_flags": [],
                    }
                ]
            }
            atomic_json(
                root / "draft.json",
                {"status": "complete", "result": draft},
            )
            failed_summary = {
                "status": "failed",
                "stage": "critic",
                "unit_id": unit.unit_id,
                "input_hash": "frozen-input",
                "model": {"model_id": "critic"},
                "attempts": 2,
                "errors": ["response failed strict stage contract"] * 2,
                "terminal_provider_failure": False,
            }
            atomic_json(root / "critic.json", failed_summary)
            raw = {
                "ayahs": [
                    {
                        "ayah": 1,
                        "findings": [
                            {
                                "type": "addition",
                                "severity": "significant",
                                "where": "روشنی",
                                "arabic_ground": "ٱلْءَاخِرَةِ ... أَزْوَاجِكُمْ",
                                "explanation": "The wording adds specificity.",
                                "suggestion": "Use a broader expression.",
                            }
                        ],
                        "verdict": "revise",
                    }
                ]
            }
            for attempt in (1, 2):
                atomic_json(
                    root / f"critic-attempt{attempt}-FAILED.json",
                    {
                        "attempt": attempt,
                        "input_hash": "frozen-input",
                        "errors": ["response failed strict stage contract"],
                        "usage": {"cost": 0.01},
                        "cost_usd": 0.01,
                        "raw_text": json.dumps(raw, ensure_ascii=False),
                        "raw_response": {"attempt": attempt},
                    },
                )
            preserved = {
                path.name: path.read_bytes() for path in root.glob("*FAILED.json")
            }

            marker = freeze_targets(base, [unit])
            result = run_revalidation(
                base,
                [unit],
                {(1, 1): "فِى الاخِرَةِ مِن أَزوٰجِكُم"},
                marker,
            )

            recovered = json.loads((root / "critic.json").read_text(encoding="utf-8"))
            self.assertTrue(result["complete"])
            self.assertEqual(1, result["recovered"])
            self.assertEqual("complete", recovered["status"])
            self.assertNotIn("cost_usd", recovered)
            self.assertEqual(0.0, recovered["recovery"]["additional_provider_cost_usd"])
            for name, content in preserved.items():
                self.assertEqual(content, (root / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
