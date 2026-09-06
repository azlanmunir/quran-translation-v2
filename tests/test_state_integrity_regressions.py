from __future__ import annotations

import io
import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest

from quran_translate import audio_pipeline as audio
from quran_translate import batching, db, production_clients as clients, production_runner as runner
from quran_translate.config import DEFAULT_SOURCE_XML
from quran_translate.production_packets import ProductionUnit, atomic_json
from quran_translate.source_import import import_tanzil_xml


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    db.init_db(connection)
    import_tanzil_xml(connection)
    batching.prepare_run(connection, run_id="review")
    yield connection
    connection.close()


def test_identical_reimport_preserves_translation_and_batch(conn):
    now = db.utc_now()
    conn.execute(
        "INSERT INTO translations VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("review", "1:1", "Preserve this.", "complete", "{}", now, now),
    )
    conn.execute("UPDATE translation_batches SET status='complete'")
    conn.commit()
    before = list(conn.execute("SELECT * FROM source_files"))[0]
    import_tanzil_xml(conn)
    assert conn.execute("SELECT translation FROM translations").fetchone()[0] == "Preserve this."
    assert tuple(conn.execute("SELECT * FROM source_files").fetchone()) == tuple(before)
    assert conn.execute("SELECT COUNT(*) FROM translation_batches WHERE status='complete'").fetchone()[0] > 0


def test_changed_source_with_runs_is_rejected(conn, tmp_path):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(DEFAULT_SOURCE_XML.read_bytes())
    root.find("sura/aya").set("text", "changed source")
    path = tmp_path / "source.xml"
    path.write_bytes(ET.tostring(root))
    before = conn.execute("SELECT arabic_uthmani_min FROM source_ayahs WHERE verse_key='1:1'").fetchone()[0]
    with pytest.raises(ValueError, match="new database"):
        import_tanzil_xml(conn, path)
    assert conn.execute("SELECT arabic_uthmani_min FROM source_ayahs WHERE verse_key='1:1'").fetchone()[0] == before


def test_incomplete_source_cannot_replace_database(conn, tmp_path):
    path = tmp_path / "incomplete.xml"
    path.write_text("<quran/>")
    with pytest.raises(ValueError, match="114"):
        import_tanzil_xml(conn, path)
    assert conn.execute("SELECT COUNT(*) FROM source_ayahs").fetchone()[0] == 6236


def test_post_response_loss_is_not_retried(monkeypatch):
    send = Mock(side_effect=urllib.error.URLError("lost response"))
    monkeypatch.setattr(urllib.request, "urlopen", send)
    monkeypatch.setattr(clients, "_retry_delay", lambda _: None)
    with pytest.raises(clients.ProviderError):
        clients._json_request(urllib.request.Request("https://invalid.test", data=b"{}", method="POST"), timeout=1)
    assert send.call_count == 1


def test_get_response_loss_can_retry(monkeypatch):
    send = Mock(side_effect=[urllib.error.URLError("lost response"), io.BytesIO(b'{"ok": true}')])
    monkeypatch.setattr(urllib.request, "urlopen", send)
    monkeypatch.setattr(clients, "_retry_delay", lambda _: None)
    assert clients._json_request(urllib.request.Request("https://invalid.test"), timeout=1)["ok"]
    assert send.call_count == 2


def test_submission_receipt_reuses_known_batch(tmp_path):
    client = Mock()
    client.submit.return_value = clients.BatchState("known", "ended", {})
    path = tmp_path / "job.json"
    first = clients.submit_batch_once(client, [{"id": 1}], path)
    assert clients.submit_batch_once(client, [{"id": 1}], path) == first
    assert client.submit.call_count == 1
    with pytest.raises(clients.ProviderError, match="changed"):
        clients.submit_batch_once(client, [{"id": 2}], path)


def test_uncertain_submission_is_never_replayed(tmp_path):
    client = Mock()
    client.submit.side_effect = KeyboardInterrupt
    path = tmp_path / "job.json"
    with pytest.raises(KeyboardInterrupt):
        clients.submit_batch_once(client, [{"id": 1}], path)
    with pytest.raises(clients.ProviderError, match="reconcile"):
        clients.submit_batch_once(client, [{"id": 1}], path)
    assert client.submit.call_count == 1


def test_english_resume_submits_only_never_submitted_shard(tmp_path):
    units = [ProductionUnit("first", 1, 1, 1, 1, 1, 1), ProductionUnit("second", 2, 1, 2, 2, 2, 2)]
    requests = [{"custom_id": "draft-first", "params": runner._anthropic_params(
        system=[], user="test", response_schema=runner.anthropic_schema_for_stage("draft"))}]
    atomic_json(tmp_path / "jobs/draft-a1-s001.json", {
        "request_hash": runner.stable_hash(requests), "batch_id": "first-job", "custom_ids": ["draft-first"],
    })
    atomic_json(runner.artifact_path(tmp_path, units[0], "draft"), {
        "input_hash": runner._stage_input_hash("draft", units[0], {}), "result": {},
    })
    client = Mock()
    client.submit.return_value = clients.BatchState("second-job", "ended", {})
    client.results.return_value = [{"custom_id": "draft-second", "result": {
        "type": "succeeded", "message": {"content": [{"type": "text", "text": "{}"}], "usage": {}},
    }}]
    runner.run_anthropic_stage(
        base=tmp_path, stage="draft", units=units, shard_size=1, poll_seconds=0, system=[],
        assignment=lambda *_: ("test", {}, lambda value: value), client=client,
    )
    assert client.submit.call_count == 1
    assert client.submit.call_args[0][0][0]["custom_id"] == "draft-second"
    assert runner.artifact_path(tmp_path, units[1], "draft").is_file()
    runner.run_anthropic_stage(
        base=tmp_path, stage="draft", units=units, shard_size=1, poll_seconds=0, system=[],
        assignment=lambda *_: ("test", {}, lambda value: value), client=client,
    )
    assert client.submit.call_count == 1


def prepare_audio(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(audio, "publication_rows", lambda *_: [
        {"surah_number": 1, "verse_key": "1:1", "translation": "Original wording."},
    ])
    kwargs = dict(audio_run_id="audio", translation_run_id="review", voice_id="voice",
                  model_id="model", output_format="mp3_44100_128")
    audio.prepare_audio_chunks(conn, **kwargs)
    return kwargs, conn.execute("SELECT * FROM audio_chunks").fetchone()


def test_restart_after_acceptance_before_job_write_reuses_batch(tmp_path, monkeypatch):
    unit = ProductionUnit("first", 1, 1, 1, 1, 1, 1)
    client = Mock()
    client.submit.return_value = clients.BatchState("accepted-once", "ended", {})
    client.results.return_value = [{"custom_id": "draft-first", "result": {
        "type": "succeeded", "message": {"content": [{"type": "text", "text": "{}"}], "usage": {}}
    }}]
    real_write = runner.atomic_json
    def interrupt(path, payload):
        if path.name == "draft-a1-s001.json":
            raise KeyboardInterrupt
        real_write(path, payload)
    kwargs = dict(base=tmp_path, stage="draft", units=[unit], shard_size=1, poll_seconds=0,
                  system=[], assignment=lambda *_: ("test", {}, lambda value: value), client=client)
    monkeypatch.setattr(runner, "atomic_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_anthropic_stage(**kwargs)
    monkeypatch.setattr(runner, "atomic_json", real_write)
    runner.run_anthropic_stage(**kwargs)
    assert client.submit.call_count == 1
    assert runner.artifact_path(tmp_path, unit, "draft").is_file()


@pytest.mark.parametrize("change", ["text", "voice"])
def test_forced_audio_input_change_requires_new_run(conn, tmp_path, monkeypatch, change):
    kwargs, chunk = prepare_audio(conn, tmp_path, monkeypatch)
    if change == "text":
        monkeypatch.setattr(audio, "publication_rows", lambda *_: [
            {"surah_number": 1, "verse_key": "1:1", "translation": "Different wording."},
        ])
    else:
        kwargs["voice_id"] = "other-voice"
    with pytest.warns(FutureWarning, match="deprecated"), pytest.raises(ValueError, match="new audio_run_id"):
        audio.prepare_audio_chunks(conn, **kwargs, force=True)
    assert conn.execute("SELECT text_sha256 FROM audio_chunks").fetchone()[0] == chunk["text_sha256"]


def test_audio_reuse_requires_provenance(conn, tmp_path, monkeypatch):
    _, chunk = prepare_audio(conn, tmp_path, monkeypatch)
    path = Path(chunk["output_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"old audio")
    with pytest.raises(audio.ElevenLabsError, match="provenance"):
        audio.maybe_mark_existing_complete(conn, chunk)


def test_audio_reuse_checks_hash_and_missing_file(conn, tmp_path, monkeypatch):
    _, chunk = prepare_audio(conn, tmp_path, monkeypatch)
    path = Path(chunk["output_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    monkeypatch.setattr(audio, "probe_duration", lambda _: 1.0)
    audio.mark_chunk_complete(conn, chunk, path)
    assert audio.maybe_mark_existing_complete(conn, chunk)
    path.write_bytes(b"changed")
    with pytest.raises(audio.ElevenLabsError, match="checksum"):
        audio.maybe_mark_existing_complete(conn, chunk)
    path.unlink()
    with pytest.raises(audio.ElevenLabsError, match="missing"):
        audio.maybe_mark_existing_complete(conn, chunk)
    with pytest.raises(audio.ElevenLabsError, match="missing"):
        audio.require_all_chunks_complete(conn, "audio")
    with pytest.raises(audio.ElevenLabsError, match="missing"):
        audio.synthesize_audio_chunks(conn, audio_run_id="audio")
