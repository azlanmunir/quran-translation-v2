from unittest.mock import Mock

from quran_translate import production_clients as clients, production_runner as runner
from quran_translate.production_packets import ProductionUnit, atomic_json


def test_partial_cached_accepted_shard_is_not_resubmitted(tmp_path):
    units = [ProductionUnit("first", 1, 1, 1, 1, 1, 1),
             ProductionUnit("second", 2, 1, 2, 2, 2, 2)]
    requests = [{"custom_id": "draft-" + unit.unit_id,
                 "params": runner._anthropic_params(
                     system=[], user="test",
                     response_schema=runner.anthropic_schema_for_stage("draft"))}
                for unit in units]
    atomic_json(tmp_path / "jobs/draft-a1-plan.json",
                {"shard_size": 2, "groups": [["first", "second"]]})
    atomic_json(tmp_path / "jobs/draft-a1-s001.json",
                {"request_hash": runner.stable_hash(requests),
                 "batch_id": "accepted",
                 "custom_ids": [r["custom_id"] for r in requests]})
    cached = runner.artifact_path(tmp_path, units[0], "draft")
    atomic_json(cached, {"input_hash": runner._stage_input_hash("draft", units[0], {}),
                        "result": {"preserve": True}})
    prior = cached.read_bytes()
    client = Mock()
    client.submit.side_effect = AssertionError("Unexpected duplicate submit")
    client.retrieve.return_value = clients.BatchState("accepted", "ended", {})
    client.results.return_value = [
        {"custom_id": "draft-" + unit.unit_id,
         "result": {"type": "succeeded",
                    "message": {"content": [{"type": "text", "text": "{}"}],
                                "usage": {}}}} for unit in units]
    runner.run_anthropic_stage(
        base=tmp_path, stage="draft", units=units, shard_size=2,
        poll_seconds=0, system=[],
        assignment=lambda *_: ("test", {}, lambda value: value), client=client)
    client.submit.assert_not_called()
    client.retrieve.assert_called_once_with("accepted")
    assert cached.read_bytes() == prior
    assert runner.artifact_path(tmp_path, units[1], "draft").exists()
