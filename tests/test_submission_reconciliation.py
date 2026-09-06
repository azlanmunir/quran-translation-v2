import hashlib
import json
from unittest.mock import Mock, patch

import pytest

from quran_translate.production_clients import (
    BatchState, ProviderError, reconcile_batch_submission, submit_batch_once,
)
from quran_translate.production_packets import atomic_json


@pytest.fixture
def uncertain(tmp_path):
    requests = [{"custom_id": "example", "params": {"model": "frozen"}}]
    digest = hashlib.sha256(json.dumps(requests, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    job = tmp_path / "job.json"
    receipt = tmp_path / "job.json.submission.json"
    atomic_json(receipt, {"request_hash": digest, "state": "submitting"})
    evidence = tmp_path / "evidence.json"
    atomic_json(evidence, {"request_hash": digest, "batch_id": "known",
                          "reviewer": "test operator", "payload_association": "mock exact payload",
                          "provider_evidence": "mock support confirmation"})
    client = Mock()
    client.retrieve.return_value = BatchState("known", "ended", {"id": "known"})
    client.submit.side_effect = AssertionError("No paid submissions allowed")
    return requests, job, receipt, evidence, client


def test_adoption_preserves_intent_and_reuses_id(uncertain):
    requests, job, receipt, evidence, client = uncertain
    original = json.loads(receipt.read_text())
    result = reconcile_batch_submission(client, requests, job, batch_id="known", evidence_path=evidence)
    assert reconcile_batch_submission(client, requests, job, batch_id="known", evidence_path=evidence) == result
    assert submit_batch_once(client, requests, job) == result
    audit = json.loads(job.with_name("job.json.reconciliation.json").read_text())
    assert audit["original_receipt"] == original
    client.submit.assert_not_called()
    client.retrieve.assert_called_once_with("known")


@pytest.mark.parametrize("defect", ["hash", "evidence", "provider_id", "accepted_id"])
def test_unsafe_adoption_preserves_receipt(uncertain, defect):
    requests, job, receipt, evidence, client = uncertain
    if defect == "hash":
        atomic_json(receipt, {"request_hash": "wrong", "state": "submitting"})
    elif defect == "evidence":
        atomic_json(evidence, {})
    elif defect == "provider_id":
        client.retrieve.return_value = BatchState("known", "ended", {"id": "different"})
    else:
        value = json.loads(receipt.read_text())
        value["batch_id"] = "other"
        atomic_json(receipt, value)
    original = receipt.read_bytes()
    with pytest.raises(ProviderError):
        reconcile_batch_submission(client, requests, job, batch_id="known", evidence_path=evidence)
    assert receipt.read_bytes() == original
    client.submit.assert_not_called()


def test_interrupted_adoption_can_resume_without_submission(uncertain):
    requests, job, receipt, evidence, client = uncertain

    def interrupted_write(path, value):
        if path == receipt:
            raise KeyboardInterrupt()
        atomic_json(path, value)

    with patch("quran_translate.production_clients.atomic_json", interrupted_write):
        with pytest.raises(KeyboardInterrupt):
            reconcile_batch_submission(client, requests, job, batch_id="known", evidence_path=evidence)
    with pytest.raises(ProviderError, match="Uncertain"):
        submit_batch_once(client, requests, job)
    assert reconcile_batch_submission(client, requests, job, batch_id="known", evidence_path=evidence).batch_id == "known"
    client.submit.assert_not_called()
