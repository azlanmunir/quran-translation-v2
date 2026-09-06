from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from quran_translate.short_form_production import (
    ShortFormProductionError,
    _audit_audio_semantics,
    validate_episode_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs" / "short_form_episode_001.json"
SPEC_002_PATH = ROOT / "configs" / "short_form_episode_002.json"
SPEC_003_PATH = ROOT / "configs" / "short_form_episode_003.json"
SPEC_004_PATH = ROOT / "configs" / "short_form_episode_004.json"
SPEC_005_PATH = ROOT / "configs" / "short_form_episode_005.json"


@pytest.mark.production_assets
def test_episode_001_matches_catalog_and_alignment() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    validated = validate_episode_spec(ROOT, spec)
    assert validated["candidate"]["start_ref"] == "5:8"
    assert validated["span"]["text"] == spec["source"]["exact_translation"]


@pytest.mark.production_assets
def test_episode_rejects_changed_quote() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    spec["source"]["exact_translation"] += " Changed."
    with pytest.raises(ShortFormProductionError, match="canonical catalog"):
        validate_episode_spec(ROOT, spec)


@pytest.mark.production_assets
def test_episode_002_matches_catalog_and_alignment() -> None:
    spec = json.loads(SPEC_002_PATH.read_text(encoding="utf-8"))
    validated = validate_episode_spec(ROOT, spec)
    assert validated["candidate"]["start_ref"] == "17:36"
    assert validated["span"]["text"] == spec["source"]["exact_translation"]


@pytest.mark.production_assets
def test_episode_003_matches_catalog_alignment_and_segment() -> None:
    spec = json.loads(SPEC_003_PATH.read_text(encoding="utf-8"))
    validated = validate_episode_spec(ROOT, spec)
    assert validated["candidate"]["start_ref"] == "25:63"
    assert validated["span"]["text"] == spec["source"]["exact_translation"]
    assert validated["segment_qa_path"].parent.name == "0163"


@pytest.mark.production_assets
def test_episode_004_matches_catalog_alignment_and_segment() -> None:
    spec = json.loads(SPEC_004_PATH.read_text(encoding="utf-8"))
    validated = validate_episode_spec(ROOT, spec)
    assert validated["candidate"]["start_ref"] == "30:21"
    assert validated["span"]["text"] == spec["source"]["exact_translation"]
    assert validated["segment_qa_path"].parent.name == "0183"


@pytest.mark.production_assets
def test_episode_005_matches_multi_ayah_catalog_range() -> None:
    spec = json.loads(SPEC_005_PATH.read_text(encoding="utf-8"))
    validated = validate_episode_spec(ROOT, spec)
    assert validated["candidate"]["start_ref"] == "103:1"
    assert validated["candidate"]["end_ref"] == "103:3"
    assert [span["ref"] for span in validated["spans"]] == [
        "103:1",
        "103:2",
        "103:3",
    ]
    assert validated["span"]["text"] == spec["source"]["exact_translation"]
    assert validated["segment_qa_path"].parent.name == "0302"


@pytest.mark.production_assets
def test_episode_rejects_mismatched_alignment_and_video_timeline() -> None:
    spec = json.loads(SPEC_002_PATH.read_text(encoding="utf-8"))
    spec["source"]["video_path"] = (
        "output/video/runs/quran-v2.4.1-youtube-production-v1/"
        "surahs/surah-017-al-isra.mp4"
    )
    with pytest.raises(ShortFormProductionError, match="alignment segment"):
        validate_episode_spec(ROOT, spec)


@pytest.mark.production_assets
def test_episode_rejects_post_roll_into_next_verse() -> None:
    spec = json.loads(SPEC_002_PATH.read_text(encoding="utf-8"))
    spec["source"]["post_roll_seconds"] = 0.8
    with pytest.raises(ShortFormProductionError, match="next spoken verse"):
        validate_episode_spec(ROOT, spec)


@pytest.mark.production_assets
def test_pre_roll_cannot_include_previous_spoken_word(monkeypatch):
    from quran_translate import short_form_production as production
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    spec["source"]["pre_roll_seconds"] = 0.4
    read_json = production._read_json

    def with_close_previous_word(path):
        data = read_json(path)
        if "words" in data and "spans" in data:
            start = spec["source"]["clip_start_seconds"]
            data["words"].append({"start": start - 0.5, "end": start - 0.2})
        return data

    monkeypatch.setattr(production, "_read_json", with_close_previous_word)
    with pytest.raises(ShortFormProductionError, match="previous spoken verse"):
        validate_episode_spec(ROOT, spec)


def test_spoken_audio_gate_rejects_missing_first_word(monkeypatch):
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    missing_first = spec["source"]["exact_translation"].split(" ", 1)[1]
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(
        transcribe=lambda *args, **kwargs: {"text": missing_first}
    ))
    with pytest.raises(ShortFormProductionError, match="does not match"):
        _audit_audio_semantics(spec, Path("unused.wav"))



def test_spoken_audio_gate_requires_expected_terminal_words(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = json.loads(SPEC_002_PATH.read_text(encoding="utf-8"))
    fake_whisper = SimpleNamespace(
        transcribe=lambda *args, **kwargs: {
            "text": "Finds spread open. Read your record. Whoever is"
        }
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_whisper)
    with pytest.raises(ShortFormProductionError, match="final words"):
        _audit_audio_semantics(spec, Path("unused.wav"))


def test_spoken_audio_gate_accepts_canonical_verse(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = json.loads(SPEC_002_PATH.read_text(encoding="utf-8"))
    fake_whisper = SimpleNamespace(
        transcribe=lambda *args, **kwargs: {"text": spec["source"]["exact_translation"]}
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_whisper)
    result = _audit_audio_semantics(spec, Path("unused.wav"))
    assert result["passed"] is True
    assert result["word_sequence_similarity"] == 1.0
