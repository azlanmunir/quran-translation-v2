from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from quran_translate import urdu_production as urdu
from quran_translate.production_clients import BatchState
from quran_translate.production_packets import ProductionUnit, atomic_json
from quran_translate.urdu_costs import usage_cost


@pytest.fixture
def revision(tmp_path, monkeypatch):
    units = [ProductionUnit("first", 1, 1, 1, 1, 1, 1),
             ProductionUnit("second", 2, 1, 2, 2, 2, 2)]
    monkeypatch.setattr(urdu, "_units_requiring_revision", lambda *_: units)
    monkeypatch.setattr(urdu, "load_environment", lambda: None)
    monkeypatch.setattr(urdu, "_revision_assignment", lambda *_: ("system", "user", []))
    monkeypatch.setattr(urdu, "validate_revision", lambda value, **k: value)
    monkeypatch.setattr(urdu, "CONTRACT_ATTEMPTS", 1)
    monkeypatch.setattr(urdu, "estimate_request_ceiling", lambda *_: .1)
    usage = {"input_tokens": 1000, "output_tokens": 1000}
    rows = [{"custom_id": unit.unit_id, "result": {
        "type": "succeeded", "message": {"content": [{"type": "text", "text": "{}"}], "usage": usage}
    }} for unit in units]
    client = Mock()
    client.submit.return_value = BatchState("existing-job", "ended", {})
    client.retrieve.return_value = BatchState("existing-job", "ended", {})
    client.results.return_value = rows
    monkeypatch.setattr(urdu, "AnthropicBatchClient", lambda *_: client)
    return tmp_path, units, client, rows, usage


def test_urdu_partial_collection_resumes_original_membership(revision, monkeypatch):
    base, units, client, _rows, _usage = revision
    real_write = urdu.atomic_json
    second_path = urdu.artifact_path(base, units[1], "revision")
    def interrupt(path, value):
        if path == second_path:
            raise KeyboardInterrupt
        real_write(path, value)
    monkeypatch.setattr(urdu, "atomic_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        urdu.run_revisions(base, units, {}, {}, budget=urdu.BudgetLedger(base, 100))
    first_path = urdu.artifact_path(base, units[0], "revision")
    first_bytes = first_path.read_bytes()
    assert urdu.BudgetLedger(base, 100).report()["reserved_usd"] == .2
    monkeypatch.setattr(urdu, "atomic_json", real_write)
    urdu.run_revisions(base, units, {}, {}, budget=urdu.BudgetLedger(base, 100))
    assert client.submit.call_count == 1
    assert client.retrieve.call_count == 1
    assert first_path.read_bytes() == first_bytes
    assert second_path.exists()
    assert urdu.BudgetLedger(base, 100).report()["reserved_usd"] == 0


def test_malformed_paid_response_keeps_usage_and_cost(revision):
    base, units, _client, rows, usage = revision
    for row in rows:
        row["result"]["message"]["content"][0]["text"] = "not valid json"
    with pytest.raises(urdu.UrduProductionError, match="exhausted"):
        urdu.run_revisions(base, units, {}, {}, budget=urdu.BudgetLedger(base, 100))
    expected = usage_cost(urdu.REVISION_MODEL.model_id, usage)
    for unit in units:
        failure = json.loads((urdu.unit_dir(base, unit) / "revision-attempt1-FAILED.json").read_text())
        assert failure["usage"] == usage
        assert failure["cost_usd"] == expected
    assert urdu.BudgetLedger(base, 100).spent() == pytest.approx(expected * 2)


def test_historical_failed_usage_is_reconciled_without_rewriting(tmp_path):
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    path = tmp_path / "revision-FAILED.json"
    atomic_json(path, {"stage": "revision", "raw": {"result": {"message": {"usage": usage}}}})
    before = path.read_bytes()
    assert urdu.BudgetLedger(tmp_path, 100).spent() == usage_cost(urdu.REVISION_MODEL.model_id, usage)
    assert path.read_bytes() == before


def test_valid_json_without_usage_keeps_budget_hold(revision):
    base, units, _client, rows, _usage = revision
    for row in rows:
        row["result"]["message"]["usage"] = {}
    with pytest.raises(urdu.UrduProductionError, match="exhausted"):
        urdu.run_revisions(base, units, {}, {}, budget=urdu.BudgetLedger(base, 100))
    assert urdu.BudgetLedger(base, 100).report()["reserved_usd"] == .2
    assert not any(urdu.artifact_path(base, unit, "revision").exists() for unit in units)


@pytest.mark.parametrize("value", [float("nan"), -1.0])
def test_invalid_cost_evidence_fails_closed(tmp_path, value):
    atomic_json(tmp_path / "failure.json", {"cost_usd": value})
    with pytest.raises(urdu.BudgetExceeded, match="Invalid cost"):
        urdu.BudgetLedger(tmp_path, 100).reserve(.01)


def test_uncertain_batch_keeps_budget_hold_and_never_resubmits(revision):
    base, units, client, _rows, _usage = revision
    client.submit.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        urdu.run_revisions(base, units, {}, {}, budget=urdu.BudgetLedger(base, 100))
    assert urdu.BudgetLedger(base, 100).report()["reserved_usd"] == .2
    with pytest.raises(urdu.BudgetExceeded):
        urdu.BudgetLedger(base, .1).reserve(.01)
    with pytest.raises(urdu.ProviderError, match="reconcile"):
        urdu.run_revisions(base, units, {}, {}, budget=urdu.BudgetLedger(base, 100))
    assert client.submit.call_count == 1


@pytest.fixture
def repair(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/repair_urdu_audio_content_unit.py"
    spec = importlib.util.spec_from_file_location("repair_regression_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PRODUCTION_ROOT", tmp_path)
    atomic_json(tmp_path / "RUN.json", {"jobs": [{"actual_batch_cost_usd": 1}]})
    return module


def test_unadopted_repair_attempts_count_toward_ceiling(repair):
    for index, cost in enumerate([2, 3], 1):
        atomic_json(repair.PRODUCTION_ROOT / f"content-repairs/unit/attempt-{index:04d}/ayahs/01/COMPLETE.json",
                    {"actual_standard_cost_usd": cost})
    assert repair._repair_spend() == 6


def test_adopted_repair_spend_is_not_counted_twice(repair):
    atomic_json(repair.PRODUCTION_ROOT / "RUN.json", {"jobs": [{
        "actual_batch_cost_usd": 3, "content_integrity_repairs": [{"repair_standard_cost_usd": 2}]
    }]})
    atomic_json(repair.PRODUCTION_ROOT / "content-repairs/unit/attempt-0001/ayahs/01/COMPLETE.json",
                {"actual_standard_cost_usd": 2})
    assert repair._repair_spend() == 3


def test_uncertain_repair_call_blocks_further_requests(repair):
    first = repair.PRODUCTION_ROOT / "content-repairs/unit/attempt-0001/ayahs/01"
    second = first.with_name("02")
    repair._reserve_repair_call(first, "fixture")
    with pytest.raises(repair.ContentRepairError, match="Uncertain"):
        repair._reserve_repair_call(second, "fixture")
    with pytest.raises(repair.ContentRepairError, match="replayed"):
        repair._reserve_repair_call(first, "fixture")
    assert not (second / "CALL.json").exists()


def test_missing_usage_does_not_release_repair_reservation(repair):
    directory = repair.PRODUCTION_ROOT / "content-repairs/unit/attempt-0001/ayahs/01"
    call = repair._reserve_repair_call(directory, "fixture")
    with pytest.raises(repair.ContentRepairError, match="usage missing"):
        repair._account_repair_call(call, SimpleNamespace())
    assert json.loads(call.read_text())["state"] == "submitting"


def test_repair_stops_before_second_call_when_first_exceeds_ceiling(repair, monkeypatch):
    unit = {"unit_id": "unit", "unit_index": 1, "refs": ["1:1", "1:2"],
            "speech_text": "first line\nsecond line", "speech_text_sha256": "fixture", "speech_characters": 22}
    atomic_json(repair.PRODUCTION_ROOT / "UNITS.json", [unit])
    atomic_json(repair.PRODUCTION_ROOT / "RUN.json", {"jobs": [{
        "unit_id": "unit", "status": "complete", "actual_batch_cost_usd": 1,
        "probe": {"duration_seconds": 2}
    }]})
    monkeypatch.setattr(repair, "load_dotenv", lambda: None)
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-not-a-real-key")
    response = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=1, candidates_token_count=30_000_000
    ))
    provider = Mock()
    provider.models.generate_content.return_value = response
    monkeypatch.setattr(repair.genai, "Client", lambda **k: provider)
    with pytest.raises(repair.ContentRepairError, match="no further calls"):
        repair.generate("unit")
    assert provider.models.generate_content.call_count == 1
    assert repair._repair_spend() > 20
    call = next(repair.PRODUCTION_ROOT.rglob("CALL.json"))
    assert json.loads(call.read_text())["state"] == "accounted"


def test_repair_reservation_blocks_before_provider_at_limit(repair, monkeypatch):
    atomic_json(repair.PRODUCTION_ROOT / "RUN.json", {"jobs": [{"actual_batch_cost_usd": 19.99}]})
    with pytest.raises(repair.ContentRepairError, match="ceiling"):
        repair._reserve_repair_call(
            repair.PRODUCTION_ROOT / "content-repairs/unit/attempt-0001/ayahs/01", "fixture"
        )
    assert not list(repair.PRODUCTION_ROOT.rglob("CALL.json"))
