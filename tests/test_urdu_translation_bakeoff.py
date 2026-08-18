from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quran_translate.production_packets import atomic_json
from quran_translate.urdu_translation_bakeoff import (
    CANDIDATES,
    PASSAGES,
    UrduBakeoffError,
    _openai_output_text,
    _private_blind_key,
    _private_result_path,
    _translation_input_hash,
    audit_schema,
    build_passage_payloads,
    package_blind_workbook,
    prepare_bakeoff,
    run_generation,
    seed_generation,
    validate_audit,
    validate_translation,
)


class UrduTranslationBakeoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payloads = build_passage_payloads()

    def test_corpus_is_exactly_ninety_unique_ayahs(self) -> None:
        refs = {
            (passage.surah, ayah)
            for passage in PASSAGES
            for ayah in passage.expected_ayahs
        }
        self.assertEqual(len(PASSAGES), 11)
        self.assertEqual(len(refs), 90)
        self.assertEqual(len(self.payloads), len(PASSAGES))
        self.assertTrue(all(payload["input_sha256"] for payload in self.payloads))
        serialized = json.dumps(self.payloads, ensure_ascii=False)
        self.assertNotIn('"english"', serialized)

    def test_translation_contract_requires_urdu_and_exact_ayahs(self) -> None:
        valid = {
            "ayahs": [
                {"ayah": 1, "urdu": "اللہ ایک ہے۔", "review_flags": []},
                {"ayah": 2, "urdu": "سب اسی کے محتاج ہیں۔", "review_flags": ["rare_word"]},
            ]
        }
        self.assertIsNotNone(validate_translation(valid, [1, 2]))
        devanagari = json.loads(json.dumps(valid, ensure_ascii=False))
        devanagari["ayahs"][0]["urdu"] = "ईश्वर एक है"
        self.assertIsNone(validate_translation(devanagari, [1, 2]))
        missing = json.loads(json.dumps(valid, ensure_ascii=False))
        missing["ayahs"].pop()
        self.assertIsNone(validate_translation(missing, [1, 2]))

    def test_audit_contract_checks_quotes_and_scores(self) -> None:
        codes = ["Candidate A"]
        urdu = {"Candidate A": {1: "اللہ ایک ہے۔"}}
        arabic = {1: "قُل هُوَ اللَّهُ أَحَدٌ"}
        valid = {
            "candidates": [
                {
                    "code": "Candidate A",
                    "ayahs": [
                        {
                            "ayah": 1,
                            "findings": [],
                            "verdict": "pass",
                        }
                    ],
                    "scores": {
                        "clarity": 5,
                        "naturalness": 5,
                        "pakistani_urdu": 5,
                        "source_force": 4,
                        "spoken_cadence": 5,
                    },
                    "summary": "صاف اور رواں",
                }
            ]
        }
        self.assertIsNotNone(
            validate_audit(
                valid,
                codes=codes,
                expected=[1],
                urdu_by_code=urdu,
                arabic_by_ayah=arabic,
            )
        )
        invalid = json.loads(json.dumps(valid, ensure_ascii=False))
        invalid["candidates"][0]["ayahs"][0] = {
            "ayah": 1,
            "findings": [
                {
                    "type": "addition",
                    "severity": "significant",
                    "where": "غلط اقتباس",
                    "arabic_ground": "اللَّهُ",
                    "explanation": "unsupported",
                    "suggestion": "",
                }
            ],
            "verdict": "revise",
        }
        self.assertIsNone(
            validate_audit(
                invalid,
                codes=codes,
                expected=[1],
                urdu_by_code=urdu,
                arabic_by_ayah=arabic,
            )
        )

    def test_audit_accepts_grounded_orthographic_variants_and_minor_pass(self) -> None:
        document = {
            "candidates": [
                {
                    "code": "Candidate A",
                    "ayahs": [
                        {
                            "ayah": 1,
                            "findings": [
                                {
                                    "type": "register_error",
                                    "severity": "minor",
                                    "where": "\"اللہ ایک ہے\"",
                                    "arabic_ground": "ٱللَّهُ أَحَدٌ",
                                    "explanation": "معمولی اسلوبی کھردرا پن",
                                    "suggestion": "",
                                }
                            ],
                            "verdict": "pass",
                        }
                    ],
                    "scores": {
                        "clarity": 4,
                        "naturalness": 4,
                        "pakistani_urdu": 4,
                        "source_force": 4,
                        "spoken_cadence": 4,
                    },
                    "summary": "قابل قبول",
                }
            ]
        }
        self.assertIsNotNone(
            validate_audit(
                document,
                codes=["Candidate A"],
                expected=[1],
                urdu_by_code={"Candidate A": {1: "اللہ ایک ہے۔"}},
                arabic_by_ayah={1: "اللَّهُ أَحَدٌ"},
            )
        )

    def test_schema_is_strict_at_every_object_level(self) -> None:
        schema = audit_schema(["Candidate A"])
        self.assertFalse(schema["additionalProperties"])
        candidate = schema["properties"]["candidates"]["items"]
        self.assertFalse(candidate["additionalProperties"])
        ayah = candidate["properties"]["ayahs"]["items"]
        self.assertFalse(ayah["additionalProperties"])
        finding = ayah["properties"]["findings"]["items"]
        self.assertFalse(finding["additionalProperties"])

    def test_prepare_is_idempotent_and_key_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = prepare_bakeoff(root)
            first_manifest = (root / "MANIFEST.json").read_bytes()
            first_key = (root / "PRIVATE_BLIND_KEY.json").read_bytes()
            second = prepare_bakeoff(root)
            self.assertEqual(first["bakeoff_id"], second["bakeoff_id"])
            self.assertEqual(first_manifest, (root / "MANIFEST.json").read_bytes())
            self.assertEqual(first_key, (root / "PRIVATE_BLIND_KEY.json").read_bytes())
            self.assertEqual((root / "PRIVATE_BLIND_KEY.json").stat().st_mode & 0o777, 0o600)
            mapping = _private_blind_key(root)
            self.assertEqual(len(mapping), len(CANDIDATES))

    def test_manifest_guard_refuses_code_or_parameter_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepare_bakeoff(root)
            with patch("quran_translate.urdu_translation_bakeoff.MAX_OUTPUT_TOKENS", 999):
                with self.assertRaisesRegex(UrduBakeoffError, "manifest"):
                    prepare_bakeoff(root)

    def test_generation_rejects_unknown_or_empty_candidate_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(UrduBakeoffError, "Unknown candidate IDs"):
                run_generation(root, candidate_ids={"not-a-candidate"})
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(UrduBakeoffError, "At least one candidate"):
                run_generation(root, candidate_ids=set())

    def test_blind_package_contains_no_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepare_bakeoff(root)
            payload_by_id = {
                payload["passage"]["passage_id"]: payload for payload in self.payloads
            }
            for passage in PASSAGES:
                payload = payload_by_id[passage.passage_id]
                for candidate in CANDIDATES:
                    rows = [
                        {
                            "ayah": ayah,
                            "urdu": f"یہ آیت {ayah} کا آزمائشی اردو متن ہے۔",
                            "review_flags": [],
                        }
                        for ayah in passage.expected_ayahs
                    ]
                    atomic_json(
                        _private_result_path(root, candidate, passage),
                        {
                            "input_hash": _translation_input_hash(candidate, payload),
                            "status": "complete",
                            "result": {"ayahs": rows},
                        },
                    )
            manifest = package_blind_workbook(root)
            self.assertEqual(len(manifest["candidate_codes"]), len(CANDIDATES))
            serialized = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (root / "blind").rglob("*")
                if path.is_file()
            )
            for candidate in CANDIDATES:
                self.assertNotIn(candidate.model_id, serialized)
                self.assertNotIn(candidate.private_label, serialized)

    def test_seed_generation_copies_validated_outputs_without_editing(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as destination_dir:
            source = Path(source_dir)
            destination = Path(destination_dir)
            prepare_bakeoff(source)
            payload_by_id = {
                payload["passage"]["passage_id"]: payload for payload in self.payloads
            }
            for passage in PASSAGES:
                payload = payload_by_id[passage.passage_id]
                for candidate in CANDIDATES:
                    document = {
                        "input_hash": _translation_input_hash(candidate, payload),
                        "status": "complete",
                        "result": {
                            "ayahs": [
                                {
                                    "ayah": ayah,
                                    "urdu": f"یہ آیت {ayah} کا آزمائشی اردو متن ہے۔",
                                    "review_flags": [],
                                }
                                for ayah in passage.expected_ayahs
                            ]
                        },
                    }
                    atomic_json(_private_result_path(source, candidate, passage), document)
            status = seed_generation(source, destination)
            self.assertEqual(status["generation"]["complete"], 77)
            for passage in PASSAGES:
                for candidate in CANDIDATES:
                    self.assertEqual(
                        _private_result_path(source, candidate, passage).read_bytes(),
                        _private_result_path(destination, candidate, passage).read_bytes(),
                    )

    def test_openai_output_text_collects_only_output_blocks(self) -> None:
        response = {
            "output": [
                {"content": [{"type": "reasoning", "text": "private"}]},
                {"content": [{"type": "output_text", "text": '{"ayahs": []}'}]},
            ]
        }
        self.assertEqual(_openai_output_text(response), '{"ayahs": []}')


if __name__ == "__main__":
    unittest.main()
