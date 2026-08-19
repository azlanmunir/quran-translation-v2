from __future__ import annotations

import json
import unittest

from quran_translate.urdu_quality import (
    deterministic_quality_gate,
    finding_records,
    validate_critic,
    validate_revision,
    validate_verification,
)


class UrduQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arabic = {1: "وَٱلشَّمْسِ وَضُحَىٰهَا"}
        self.urdu = {1: "قسم ہے سورج کی اور اس کی روشنی کی۔"}
        self.finding = {
            "type": "addition",
            "severity": "significant",
            "where": "روشنی",
            "arabic_ground": "وَضُحَىٰهَا",
            "explanation": "The Urdu narrows the image beyond the supplied evidence.",
            "suggestion": "Use a less narrowing expression.",
        }

    def test_critic_requires_exact_grounds_and_consistent_verdict(self) -> None:
        document = {
            "ayahs": [
                {"ayah": 1, "findings": [self.finding], "verdict": "revise"}
            ]
        }
        clean = validate_critic(
            document,
            expected=[1],
            arabic_by_ayah=self.arabic,
            urdu_by_ayah=self.urdu,
        )
        self.assertIsNotNone(clean)
        self.assertEqual("f-1-0", finding_records(clean)[0]["finding_id"])

        bad_ground = json.loads(json.dumps(document))
        bad_ground["ayahs"][0]["findings"][0]["arabic_ground"] = "ليس هنا"
        self.assertIsNone(
            validate_critic(
                bad_ground,
                expected=[1],
                arabic_by_ayah=self.arabic,
                urdu_by_ayah=self.urdu,
            )
        )
        wrong_verdict = json.loads(json.dumps(document))
        wrong_verdict["ayahs"][0]["verdict"] = "pass"
        self.assertIsNone(
            validate_critic(
                wrong_verdict,
                expected=[1],
                arabic_by_ayah=self.arabic,
                urdu_by_ayah=self.urdu,
            )
        )

    def test_critic_accepts_equivalent_uthmani_ground_spelling(self) -> None:
        document = {
            "ayahs": [
                {
                    "ayah": 1,
                    "findings": [
                        {
                            **self.finding,
                            "arabic_ground": "ٱلْءَاخِرَةِ ... أَزْوَاجِكُمْ",
                        }
                    ],
                    "verdict": "revise",
                }
            ]
        }
        clean = validate_critic(
            document,
            expected=[1],
            arabic_by_ayah={1: "فِى الاخِرَةِ مِن أَزوٰجِكُم"},
            urdu_by_ayah=self.urdu,
        )
        self.assertIsNotNone(clean)

        unrelated = json.loads(json.dumps(document))
        unrelated["ayahs"][0]["findings"][0]["arabic_ground"] = "لَيْسَ هُنَا"
        self.assertIsNone(
            validate_critic(
                unrelated,
                expected=[1],
                arabic_by_ayah={1: "فِى الاخِرَةِ مِن أَزوٰجِكُم"},
                urdu_by_ayah=self.urdu,
            )
        )

    def test_revision_and_verification_contracts_are_strict(self) -> None:
        revision = {
            "ayahs": [
                {"ayah": 1, "urdu": "قسم ہے سورج کی اور اس کی چمک کی۔", "review_flags": []}
            ],
            "decisions": [
                {"finding_id": "f-1-0", "decision": "applied", "reason": "Repairs the narrowing."}
            ],
        }
        self.assertIsNotNone(
            validate_revision(revision, expected=[1], finding_ids=["f-1-0"])
        )
        duplicate = json.loads(json.dumps(revision))
        duplicate["decisions"].append(duplicate["decisions"][0])
        self.assertIsNone(
            validate_revision(duplicate, expected=[1], finding_ids=["f-1-0"])
        )

        proposed = {1: revision["ayahs"][0]["urdu"]}
        accepted = {
            "ayahs": [
                {"ayah": 1, "accept": True, "findings": [], "reason": "Faithful repair."}
            ]
        }
        self.assertIsNotNone(
            validate_verification(
                accepted,
                expected=[1],
                arabic_by_ayah=self.arabic,
                proposed_by_ayah=proposed,
            )
        )
        inconsistent = json.loads(json.dumps(accepted))
        inconsistent["ayahs"][0]["findings"] = [
            {**self.finding, "where": "چمک"}
        ]
        self.assertIsNone(
            validate_verification(
                inconsistent,
                expected=[1],
                arabic_by_ayah=self.arabic,
                proposed_by_ayah=proposed,
            )
        )

    def test_deterministic_gate_catches_ledger_and_refrain_failures(self) -> None:
        verses = {
            (2, 179): "وَلَكُمْ فِى ٱلْقِصَاصِ حَيَوٰةٌ يَـٰٓأُولِى ٱلْأَلْبَـٰبِ لَعَلَّكُمْ تَتَّقُونَ",
            (2, 187): "أُحِلَّ لَكُمْ لَيْلَةَ ٱلصِّيَامِ ٱلرَّفَثُ إِلَىٰ نِسَآئِكُمْ",
            (55, 13): "فَبِأَىِّ ءَالَآءِ رَبِّكُمَا تُكَذِّبَانِ",
            (55, 16): "فَبِأَىِّ ءَالَآءِ رَبِّكُمَا تُكَذِّبَانِ",
        }
        translations = {
            (2, 179): "قصاص میں تمہارے لیے زندگی ہے تاکہ تم تقویٰ اختیار کرو۔",
            (2, 187): "صبح تک میل جول حلال ہے اور اعتکاف میں ان سے نہ ملو۔",
            (55, 13): "تم دونوں اپنے رب کی کون سی نعمتیں جھٹلاؤ گے؟",
            (55, 16): "تم دونوں اپنے رب کی کون سی بخششیں جھٹلاؤ گے؟",
        }
        result = deterministic_quality_gate(translations, verses)
        self.assertFalse(result["passed"])
        codes = {issue["code"] for issue in result["issues"]}
        self.assertIn("ledger_required", codes)
        self.assertIn("ledger_forbidden", codes)
        self.assertIn("refrain_divergence", codes)
        self.assertIn("untranslated_taqwa_verb", codes)

        translations[(2, 179)] = "قصاص میں تمہارے لیے زندگی ہے تاکہ تم بچ جاؤ۔"
        translations[(2, 187)] = (
            "روزے کی رات صحبت حلال ہے، فجر تک، اور اعتکاف میں مباشرت نہ کرو۔"
        )
        translations[(55, 16)] = translations[(55, 13)]
        self.assertTrue(deterministic_quality_gate(translations, verses)["passed"])


if __name__ == "__main__":
    unittest.main()
