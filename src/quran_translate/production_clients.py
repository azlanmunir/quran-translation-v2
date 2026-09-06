"""Provider clients used by the resumable v2.4 production pipeline."""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .production_packets import atomic_json
from .state_safety import exclusive_lock


RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504, 529}


class ProviderError(RuntimeError):
    """A provider request could not be completed safely."""


@dataclass(frozen=True)
class BatchState:
    batch_id: str
    state: str
    raw: dict[str, Any]

    @property
    def ended(self) -> bool:
        return self.state in {
            "ended",
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
            "JOB_STATE_PARTIALLY_SUCCEEDED",
        }

    @property
    def succeeded(self) -> bool:
        return self.state in {
            "ended",
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_PARTIALLY_SUCCEEDED",
        }


def _retry_delay(attempt: int) -> None:
    time.sleep(min(15 * (attempt + 1), 90))


def _json_request(
    request: urllib.request.Request,
    *,
    timeout: int,
    attempts: int = 4,
) -> dict[str, Any]:
    if request.get_method() not in {"GET", "HEAD"}:
        attempts = 1
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ProviderError("Provider returned a non-object JSON response")
            return payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            if exc.code in RETRYABLE_HTTP and attempt + 1 < attempts:
                _retry_delay(attempt)
                continue
            raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ssl.SSLError,
            http.client.IncompleteRead,
            ConnectionError,
            json.JSONDecodeError,
        ) as exc:
            if attempt + 1 >= attempts:
                raise ProviderError(f"Provider request failed: {exc}") from exc
            _retry_delay(attempt)
    raise AssertionError("unreachable")



def submit_batch_once(client: Any, requests: list[dict[str, Any]], job_path: Path) -> BatchState:
    """Persist submission intent before a mutation and reuse only a known receipt."""
    receipt_path = job_path.with_name(job_path.name + ".submission.json")
    request_hash = hashlib.sha256(json.dumps(
        requests, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(receipt_path.with_suffix(".lock")):
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("request_hash") != request_hash:
                raise ProviderError(f"Submission input changed: {receipt_path}")
            if not receipt.get("batch_id"):
                raise ProviderError(
                    f"Uncertain prior submission; reconcile before retrying: {receipt_path}"
                )
            return BatchState(receipt["batch_id"], receipt["state"], receipt.get("raw", {}))
        atomic_json(receipt_path, {"request_hash": request_hash, "state": "submitting"})
        # Any exception, including interruption, deliberately leaves a blocking intent.
        state = client.submit(requests)
        if not state.batch_id:
            raise ProviderError("Submitted batch lacks an ID; reconciliation is required")
        atomic_json(receipt_path, {
            "request_hash": request_hash, "batch_id": state.batch_id,
            "state": state.state, "raw": state.raw,
        })
        return state


def reconcile_batch_submission(
    client: Any, requests: list[dict[str, Any]], job_path: Path,
    *, batch_id: str, evidence_path: Path,
) -> BatchState:
    """Adopt an operator-proven batch using read-only provider retrieval, never replay."""
    receipt_path = job_path.with_name(job_path.name + ".submission.json")
    audit_path = job_path.with_name(job_path.name + ".reconciliation.json")
    request_hash = hashlib.sha256(json.dumps(
        requests, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (not isinstance(evidence, dict)
            or evidence.get("request_hash") != request_hash
            or evidence.get("batch_id") != batch_id
            or not batch_id
            or not all(isinstance(evidence.get(key), str) and evidence[key].strip()
                       for key in ("reviewer", "payload_association", "provider_evidence"))):
        raise ProviderError("Reconciliation requires reviewed evidence for this exact batch and payload")
    with exclusive_lock(receipt_path.with_suffix(".lock")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("request_hash") != request_hash:
            raise ProviderError("Reconciliation input changed")
        if receipt.get("batch_id"):
            if receipt["batch_id"] != batch_id:
                raise ProviderError("Cannot replace an accepted batch ID")
            return BatchState(batch_id, receipt["state"], receipt.get("raw", {}))
        if receipt.get("state") != "submitting":
            raise ProviderError("Only an uncertain submitting intent can be reconciled")
        state = client.retrieve(batch_id)
        if (state.batch_id != batch_id or not state.state
                or state.raw.get("id", state.raw.get("name")) != batch_id):
            raise ProviderError("Provider did not confirm the requested batch ID")
        audit = {"original_receipt": receipt, "evidence": evidence,
                 "provider_state": state.raw}
        if audit_path.exists():
            prior = json.loads(audit_path.read_text(encoding="utf-8"))
            if prior.get("original_receipt") != receipt or prior.get("evidence") != evidence:
                raise ProviderError("Conflicting reconciliation history; review before proceeding")
        else:
            atomic_json(audit_path, audit)
        atomic_json(receipt_path, {
            "request_hash": request_hash, "batch_id": batch_id,
            "state": state.state, "raw": state.raw,
            "reconciliation_path": str(audit_path),
        })
        return state


class AnthropicBatchClient:
    """Small dependency-free client for Anthropic Message Batches."""

    base_url = "https://api.anthropic.com/v1/messages/batches"

    def __init__(self, api_key: str, *, timeout: int = 900) -> None:
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is missing")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def submit(self, requests: list[dict[str, Any]]) -> BatchState:
        if not requests:
            raise ValueError("Cannot submit an empty Anthropic batch")
        body = json.dumps({"requests": requests}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        payload = _json_request(request, timeout=self.timeout)
        batch_id = str(payload.get("id") or "")
        state = str(payload.get("processing_status") or "")
        if not batch_id or not state:
            raise ProviderError(f"Malformed Anthropic batch response: {payload}")
        return BatchState(batch_id=batch_id, state=state, raw=payload)

    def retrieve(self, batch_id: str) -> BatchState:
        request = urllib.request.Request(
            f"{self.base_url}/{batch_id}", headers=self._headers(), method="GET"
        )
        payload = _json_request(request, timeout=self.timeout)
        state = str(payload.get("processing_status") or "")
        if not state:
            raise ProviderError(f"Malformed Anthropic batch status: {payload}")
        return BatchState(batch_id=batch_id, state=state, raw=payload)

    def results(self, batch_id: str) -> list[dict[str, Any]]:
        request = urllib.request.Request(
            f"{self.base_url}/{batch_id}/results",
            headers=self._headers(),
            method="GET",
        )
        for attempt in range(4):
            try:
                rows: list[dict[str, Any]] = []
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    for raw_line in response:
                        if not raw_line.strip():
                            continue
                        row = json.loads(raw_line.decode("utf-8"))
                        if not isinstance(row, dict):
                            raise ProviderError("Anthropic batch result row is not an object")
                        rows.append(row)
                return rows
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                if exc.code in RETRYABLE_HTTP and attempt < 3:
                    _retry_delay(attempt)
                    continue
                raise ProviderError(f"Anthropic results HTTP {exc.code}: {detail}") from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                ssl.SSLError,
                http.client.IncompleteRead,
                ConnectionError,
                json.JSONDecodeError,
            ) as exc:
                if attempt == 3:
                    raise ProviderError(f"Anthropic results failed: {exc}") from exc
                _retry_delay(attempt)
        raise AssertionError("unreachable")


def anthropic_result_text(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    result = row.get("result")
    if not isinstance(result, dict) or result.get("type") != "succeeded":
        raise ProviderError(f"Anthropic item did not succeed: {result}")
    message = result.get("message")
    if not isinstance(message, dict):
        raise ProviderError("Anthropic succeeded item lacks a message")
    content = message.get("content")
    if not isinstance(content, list):
        raise ProviderError("Anthropic message lacks content blocks")
    text = "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    if not text.strip():
        raise ProviderError("Anthropic message contains no text")
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    metadata = {
        "model": message.get("model"),
        "stop_reason": message.get("stop_reason"),
        "usage": usage,
    }
    return text, metadata


class GeminiBatchClient:
    """Thin wrapper around the official google-genai file Batch API."""

    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
    }

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ProviderError("GOOGLE_API_KEY is missing")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - exercised by environment setup.
            raise ProviderError("Install google-genai before running production") from exc
        self.client = genai.Client(api_key=api_key)

    def submit_file(
        self,
        *,
        model: str,
        input_path: Path,
        display_name: str,
    ) -> BatchState:
        if not input_path.is_file() or input_path.stat().st_size == 0:
            raise ValueError(f"Gemini batch input is missing or empty: {input_path}")
        try:
            uploaded = self.client.files.upload(
                file=input_path,
                config={"mime_type": "application/jsonl", "display_name": display_name},
            )
            job = self.client.batches.create(
                model=model,
                src=uploaded.name,
                config={"display_name": display_name},
            )
        except Exception as exc:  # SDK error hierarchy changes across releases.
            raise ProviderError(f"Gemini batch submission failed: {exc}") from exc
        batch_id = str(getattr(job, "name", "") or "")
        state = _gemini_state_name(job)
        if not batch_id:
            raise ProviderError(f"Gemini batch response lacks a name: {job}")
        return BatchState(batch_id=batch_id, state=state, raw=_model_dump(job))

    def retrieve(self, batch_id: str) -> BatchState:
        try:
            job = self.client.batches.get(name=batch_id)
        except Exception as exc:
            raise ProviderError(f"Gemini batch status failed: {exc}") from exc
        return BatchState(
            batch_id=batch_id,
            state=_gemini_state_name(job),
            raw=_model_dump(job),
        )

    def file_results(self, batch_id: str) -> list[dict[str, Any]]:
        try:
            job = self.client.batches.get(name=batch_id)
        except Exception as exc:
            raise ProviderError(f"Gemini batch result retrieval failed: {exc}") from exc
        state = _gemini_state_name(job)
        if state not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
            raise ProviderError(
                f"Gemini batch {batch_id} is not successful: {state}"
            )
        dest = getattr(job, "dest", None)
        file_name = getattr(dest, "file_name", None) if dest else None
        if not file_name:
            raise ProviderError("Gemini file batch has no output file")
        try:
            payload = self.client.files.download(file=file_name)
        except Exception as exc:
            raise ProviderError(f"Gemini result file download failed: {exc}") from exc
        rows: list[dict[str, Any]] = []
        for line in bytes(payload).decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ProviderError("Gemini result row is not an object")
            rows.append(row)
        return rows


class GeminiSynchronousClient:
    """Checkpoint-friendly synchronous Gemini transport for projects without batch quota."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ProviderError("GOOGLE_API_KEY is missing")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - environment setup.
            raise ProviderError("Install google-genai before running production") from exc
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_schema: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=user,
                config={
                    "system_instruction": system,
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                    "max_output_tokens": max_output_tokens,
                    "temperature": 0,
                },
            )
        except Exception as exc:  # SDK error hierarchy changes across releases.
            raise ProviderError(f"Gemini synchronous request failed: {exc}") from exc
        return {"response": _model_dump(response)}


def _gemini_state_name(job: Any) -> str:
    state = getattr(job, "state", None)
    name = getattr(state, "name", None)
    return str(name or state or "JOB_STATE_UNSPECIFIED")


def _model_dump(value: Any) -> dict[str, Any]:
    for method_name in ("model_dump", "to_json_dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            result = method()
            if isinstance(result, dict):
                return _json_safe(result)
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        payload = {"repr": repr(value)}
    return _json_safe(payload) if isinstance(payload, dict) else {"value": payload}


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
