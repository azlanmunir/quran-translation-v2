from __future__ import annotations

from quran_translate.urdu_video_production import (
    _alignment_recovery_prompt,
    _catalog_metadata,
    _normalized_urdu_word,
    _spans,
)
from quran_translate.urdu_video_render import build_urdu_display_events


def test_urdu_word_normalization_unifies_arabic_letter_forms() -> None:
    assert _normalized_urdu_word("كِتاب") == _normalized_urdu_word("کتاب")
    assert _normalized_urdu_word("عَلَيْهِ") == _normalized_urdu_word("علیہ")
    assert _normalized_urdu_word("يه") == _normalized_urdu_word("یہ")


def test_spans_use_spoken_offsets_but_published_text() -> None:
    unit = {
        "speech_text": "الف، لام، میم۔\nوہ کتاب ہے۔",
        "text": "الم۔\nوہ کتاب ہے۔",
        "refs": ["2:1", "2:2"],
    }
    spans = _spans(unit)
    assert spans[0]["text"] == "الم۔"
    assert unit["speech_text"][spans[0]["start_char"] : spans[0]["end_char"]] == "الف، لام، میم۔"
    assert [row["ref"] for row in spans] == ["2:1", "2:2"]


def test_short_ayahs_share_a_stable_panel() -> None:
    alignment = {
        "engine": "mlx-whisper:test",
        "audio_sha256": "audio",
        "transcript_sha256": "text",
        "spans": [
            {"kind": "ayah", "ref": "1:1", "text": "پہلی آیت", "start": 0.0, "end": 2.0},
            {"kind": "ayah", "ref": "1:2", "text": "دوسری آیت", "start": 2.0, "end": 4.0},
            {"kind": "ayah", "ref": "1:3", "text": "تیسری آیت", "start": 4.0, "end": 6.0},
        ],
    }
    display = build_urdu_display_events(alignment)
    assert len(display["events"]) == 3
    assert all(len(event["lines"]) == 3 for event in display["events"])
    assert [event["active_ref"] for event in display["events"]] == ["1:1", "1:2", "1:3"]


def test_public_metadata_uses_para_number_and_has_no_ai_label() -> None:
    units = [{"refs": ["1:1", "1:7"]}]
    metadata = {
        "1:1": {
            "surah_name_en": "Al-Fatihah",
            "surah_name_ar": "الفاتحة",
        }
    }
    payload = _catalog_metadata(
        kind="para",
        number=1,
        units=units,
        chapters=[],
        metadata=metadata,
    )
    assert "Para 1/30" in payload["title"]
    assert "AI" not in payload["description"]
    assert payload["visibility"] == "public"


def test_alignment_recovery_prompt_is_short_and_whitespace_normalized() -> None:
    prompt = _alignment_recovery_prompt(("  پہلا   دوسرا\n" * 80).strip())
    assert len(prompt) <= 220
    assert "\n" not in prompt
    assert "  " not in prompt
