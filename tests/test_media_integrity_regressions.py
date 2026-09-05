from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from quran_translate import catalog_cache as cache
from quran_translate import short_form_production as short
from quran_translate import urdu_video_production as urdu
from quran_translate.config import file_sha256, text_sha256
from quran_translate.production_packets import atomic_json
from quran_translate.publication_receipts import record_publication_state
from quran_translate.state_safety import exclusive_lock


@pytest.mark.parametrize("observed", [
    "Do pursue what you have no knowledge of.",
    "Do not pursue what you have knowledge of.",
    "Do not pursue what you have no knowledge of whoever.",
    "Give three measures to Mary.",
    "Give two measures to John.",
])
def test_spoken_gate_rejects_meaning_changes(monkeypatch, observed):
    expected = ("Give two measures to Mary." if observed.startswith("Give")
                else "Do not pursue what you have no knowledge of.")
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(
        transcribe=lambda *a, **k: {"text": observed}
    ))
    with pytest.raises(short.ShortFormProductionError, match="canonical"):
        short._audit_audio_semantics({"source": {"exact_translation": expected}}, Path("fixture"))


def test_spoken_gate_accepts_only_punctuation_and_case_changes(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(
        transcribe=lambda *a, **k: {"text": "DO NOT PURSUE."}
    ))
    assert short._audit_audio_semantics(
        {"source": {"exact_translation": "Do not pursue!"}}, Path("fixture")
    )["passed"]


def test_encoded_master_is_audited_even_when_wav_passed(tmp_path, monkeypatch):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"fixture")
    monkeypatch.setattr(short, "_ffprobe", lambda _: {
        "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920},
                    {"codec_type": "audio", "codec_name": "aac"}],
        "format": {"duration": 10},
    })
    monkeypatch.setattr(short, "_run", lambda _: None)
    audit = Mock(side_effect=short.ShortFormProductionError("encoded truncation"))
    monkeypatch.setattr(short, "_audit_audio_semantics", audit)
    spec = {"render": {"width": 1080, "height": 1920, "duration_seconds": 10}}
    with pytest.raises(short.ShortFormProductionError, match="encoded truncation"):
        short._audit_video(spec, tmp_path, master, {}, {"passed": True})
    assert audit.call_args.args[1] == master


@pytest.fixture
def episode(tmp_path, monkeypatch):
    for name in ("catalog", "alignment", "video", "qa", "background"):
        (tmp_path / name).write_text(name)
    spec = {"episode_id": "episode-1", "source": {
        "catalog_path": "catalog", "alignment_path": "alignment", "video_path": "video"
    }, "visual": {"background_path": "background"}}
    spec_path = tmp_path / "spec.json"
    atomic_json(spec_path, spec)
    monkeypatch.setattr(short, "validate_episode_spec",
                        lambda *a: {"segment_qa_path": tmp_path / "qa"})

    def render(_root, _spec, directory):
        directory.mkdir(parents=True)
        (directory / "master.mp4").write_bytes(b"immutable media")
        atomic_json(directory / "QA.json", {"passed": True, "path": str(directory / "master.mp4")})
        atomic_json(directory / "PUBLICATION_STATE.json", {"instagram": {"status": "not_uploaded"}})
        return {"output_dir": str(directory)}

    renderer = Mock(side_effect=render)
    monkeypatch.setattr(short, "_render_short_form_episode", renderer)
    return tmp_path, spec_path, spec, renderer


def test_identical_render_preserves_partial_publication(episode):
    root, spec_path, _spec, renderer = episode
    result = short.render_short_form_episode(root, spec_path)
    directory = Path(result["output_dir"])
    state = {"instagram": {"status": "published", "url": "https://example.test/existing"},
             "tiktok": {"status": "pending"}}
    atomic_json(directory / "PUBLICATION_STATE.json", state)
    receipt = (directory / "PUBLICATION_STATE.json").read_bytes()
    again = short.render_short_form_episode(root, spec_path)
    assert again["reused"] and again["publication"] == state
    assert (directory / "PUBLICATION_STATE.json").read_bytes() == receipt
    assert json.loads((directory / "QA.json").read_text())["path"] == str(directory / "master.mp4")
    assert renderer.call_count == 1
    history = [json.loads(path.read_text()) for path in (directory / "publication-history").glob("*.json")]
    assert state in history and len(history) == 2


def test_publication_updates_preserve_history_and_reject_stale_writes(episode):
    root, path, _spec, _renderer = episode
    directory = Path(short.render_short_form_episode(root, path)["output_dir"])
    state_path = directory / "PUBLICATION_STATE.json"
    original_hash = file_sha256(state_path)
    state = {"instagram": {"status": "published", "id": "existing"}}
    record_publication_state(directory, state, expected_state_sha256=original_hash)
    with pytest.raises(ValueError, match="reconcile"):
        record_publication_state(directory, {"instagram": {}}, expected_state_sha256=original_hash)
    assert json.loads(state_path.read_text()) == state
    assert len(list((directory / "publication-history").glob("*.json"))) == 2


def test_changed_episode_requires_linked_new_version(episode):
    root, spec_path, spec, renderer = episode
    first = short.render_short_form_episode(root, spec_path)
    receipt = Path(first["output_dir"]) / "PUBLICATION_STATE.json"
    before = receipt.read_bytes()
    spec["changed"] = True
    atomic_json(spec_path, spec)
    with pytest.raises(short.ShortFormProductionError, match="new episode ID"):
        short.render_short_form_episode(root, spec_path)
    spec.update(episode_id="episode-1-v2", replaces_episode_id="episode-1")
    atomic_json(spec_path, spec)
    replacement = short.render_short_form_episode(root, spec_path)
    assert receipt.read_bytes() == before
    assert json.loads((Path(replacement["output_dir"]) / "RENDER_COMPLETE.json").read_text())[
        "replaces_episode_id"] == "episode-1"
    assert renderer.call_count == 2


def test_legacy_episode_never_has_receipts_reset(episode):
    root, path, _spec, renderer = episode
    receipt = root / "output/short-form/episodes/episode-1/PUBLICATION_STATE.json"
    atomic_json(receipt, {"instagram": {"status": "published", "id": "existing"}})
    before = receipt.read_bytes()
    with pytest.raises(short.ShortFormProductionError, match="preserve"):
        short.render_short_form_episode(root, path)
    assert receipt.read_bytes() == before
    renderer.assert_not_called()


@pytest.mark.parametrize("failure_point", ["media", "qa", "publication"])
def test_interrupted_render_never_promotes_incomplete_episode(episode, failure_point):
    root, path, _spec, renderer = episode
    real = renderer.side_effect
    def fail(_root, _spec_path, directory):
        directory.mkdir(parents=True)
        (directory / "master.mp4").write_bytes(b"partial")
        if failure_point in {"qa", "publication"}:
            atomic_json(directory / "QA.json", {"passed": True})
        if failure_point == "publication":
            atomic_json(directory / "PUBLICATION_STATE.json", {"instagram": {}})
        raise KeyboardInterrupt
    renderer.side_effect = fail
    with pytest.raises(KeyboardInterrupt):
        short.render_short_form_episode(root, path)
    assert not (root / "output/short-form/episodes/episode-1").exists()
    renderer.side_effect = real
    assert Path(short.render_short_form_episode(root, path)["output_dir"]).is_dir()
    assert len(list((root / "output/short-form/episodes/.render-attempts").iterdir())) == 1


def test_exclusive_lock_rejects_concurrent_worker(tmp_path):
    with exclusive_lock(tmp_path / "worker.lock"):
        with pytest.raises(RuntimeError, match="Another worker"):
            with exclusive_lock(tmp_path / "worker.lock"):
                pytest.fail("second worker entered")


def alignment_fixture(tmp_path, monkeypatch, missing=(), first_ayah_words=20):
    words = [f"word{index}" for index in range(20)]
    transcript = " ".join(words[:first_ayah_words])
    refs = ["1:1"]
    if first_ayah_words < 20:
        transcript += "\n" + " ".join(words[first_ayah_words:])
        refs.append("1:2")
    unit = {"unit_id": "fixture", "speech_text": transcript, "text": transcript,
            "speech_text_sha256": text_sha256(transcript), "refs": refs}
    audio = tmp_path / "audio"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(urdu, "_audio_duration", lambda _: 10)
    expected = urdu._transcript_words(transcript)
    mapping = {i: (i * .3, i * .3 + .2) for i in range(20) if i not in missing}
    timings = urdu._interpolate_word_times(expected, mapping)
    aligned = [dict(row, start=timings[i][0], end=timings[i][1],
                    timing_source="mapped_word" if i in mapping else "interpolated_word")
               for i, row in enumerate(expected)]
    spans = []
    for span in urdu._spans(unit):
        selected = [w for w in aligned if w["start_char"] < span["end_char"]
                    and w["end_char"] > span["start_char"]]
        spans.append(dict(span, start=selected[0]["start"], end=selected[-1]["end"]))
    payload = {"engine": "mlx-whisper:test", "audio_sha256": file_sha256(audio),
               "transcript_sha256": unit["speech_text_sha256"], "words": aligned, "spans": spans,
               "metrics": {"mapped_word_coverage": 1.0}}
    return unit, payload, audio


def test_alignment_accepts_small_measured_internal_gap(tmp_path, monkeypatch):
    unit, payload, audio = alignment_fixture(tmp_path, monkeypatch, missing=[5])
    urdu._validate_alignment(unit, payload, audio)


@pytest.mark.parametrize("mapping", [
    {0: (0, .2), 1: (.3, .5)},
    {0: (0, .2), 7: (2.1, 2.3)},
    {0: (0, .2), 2: (5, 5.2), 3: (5.3, 5.5)},
])
def test_alignment_prohibits_unsupported_boundaries_and_long_gaps(mapping):
    with pytest.raises(urdu.AlignmentError):
        urdu._interpolate_word_times([{}] * (4 if 3 in mapping else 8), mapping)


def test_alignment_recomputes_coverage_instead_of_trusting_metrics(tmp_path, monkeypatch):
    unit, payload, audio = alignment_fixture(tmp_path, monkeypatch, missing=[5, 10])
    with pytest.raises(urdu.UrduVideoProductionError, match="coverage"):
        urdu._validate_alignment(unit, payload, audio)


def test_alignment_requires_evidence_in_each_ayah(tmp_path, monkeypatch):
    unit, payload, audio = alignment_fixture(tmp_path, monkeypatch, missing=[1], first_ayah_words=2)
    with pytest.raises(urdu.UrduVideoProductionError, match="ayah"):
        urdu._validate_alignment(unit, payload, audio)


@pytest.mark.parametrize("mutation", ["beyond_audio", "nan", "span_offsets", "word_text"])
def test_alignment_rejects_timing_or_transcript_drift(tmp_path, monkeypatch, mutation):
    unit, payload, audio = alignment_fixture(tmp_path, monkeypatch)
    if mutation == "beyond_audio":
        payload["words"][-1]["end"] = 100
    elif mutation == "nan":
        payload["words"][-1]["end"] = float("nan")
    elif mutation == "span_offsets":
        payload["spans"][0]["start_char"] = 3
    else:
        payload["words"][0]["text"] = "changed"
    with pytest.raises(urdu.UrduVideoProductionError):
        urdu._validate_alignment(unit, payload, audio)


@pytest.fixture
def catalog(tmp_path):
    output = tmp_path / "catalog.mp4"
    qa = output.with_suffix(".qa.json")
    captions, metadata = output.with_suffix(".ur.srt"), output.with_suffix(".metadata.json")
    for path in (output, captions, metadata):
        path.write_text(path.name)
    contract = cache.catalog_contract([output], [captions], {"qa": "v2"})
    prior = {"input_contract": contract, "decode_passed": True, "sha256": file_sha256(output),
             "captions_sha256": file_sha256(captions), "metadata_sha256": file_sha256(metadata)}
    atomic_json(qa, prior)
    return output, qa, captions, metadata, contract, prior


@pytest.mark.parametrize("which,action", [
    (2, "missing"), (3, "missing"), (2, "changed"), (3, "changed"), (0, "changed"),
])
def test_catalog_cache_checks_all_delivery_artifacts(catalog, which, action):
    path = catalog[which]
    path.unlink() if action == "missing" else path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        cache.validate_cached_catalog(catalog[0], catalog[1], catalog[4], "ur")


def test_catalog_cache_rejects_source_or_qa_contract_drift(catalog):
    output, qa, _captions, _metadata, contract, _prior = catalog
    for key in contract:
        changed = copy.deepcopy(contract)
        changed[key] = "changed"
        with pytest.raises(ValueError, match="inputs"):
            cache.validate_cached_catalog(output, qa, changed, "ur")


def test_catalog_recovers_only_byte_identical_sidecars(catalog):
    output, qa, captions, metadata, contract, prior = catalog
    media_before = output.read_bytes()
    caption_bytes, metadata_bytes = captions.read_bytes(), metadata.read_bytes()
    captions.unlink()
    metadata.unlink()
    def restore(first, second):
        first.write_bytes(caption_bytes)
        second.write_bytes(metadata_bytes)
    assert cache.validate_cached_catalog(output, qa, contract, "ur", restore) == prior
    assert output.read_bytes() == media_before
    assert captions.read_bytes() == caption_bytes and metadata.read_bytes() == metadata_bytes


def test_catalog_recovery_rejects_changed_generated_sidecar(catalog):
    output, qa, captions, metadata, contract, _prior = catalog
    captions.unlink()
    def restore(first, second):
        first.write_bytes(b"incorrect")
        second.write_bytes(metadata.read_bytes())
    with pytest.raises(ValueError, match="verified original"):
        cache.validate_cached_catalog(output, qa, contract, "ur", restore)
    assert not captions.exists()
