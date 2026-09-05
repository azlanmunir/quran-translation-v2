"""Append-only publication snapshots with compare-and-swap state updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .production_packets import atomic_json
from .state_safety import exclusive_lock


def _snapshot(directory: Path, state: dict[str, Any]) -> None:
    digest = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = directory / "publication-history" / f"{digest}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != state:
            raise ValueError("Publication history integrity failure")
    else:
        atomic_json(path, state)


def preserve_publication_receipt(directory: Path) -> None:
    with exclusive_lock(directory.parent / ".locks" / f"{directory.name}.publication.lock"):
        path = directory / "PUBLICATION_STATE.json"
        if path.exists():
            _snapshot(directory, json.loads(path.read_text(encoding="utf-8")))


def record_publication_state(
    directory: Path, state: dict[str, Any], *, expected_state_sha256: str
) -> None:
    """Caller must supply the hash of the state it read before publishing."""
    with exclusive_lock(directory.parent / ".locks" / f"{directory.name}.publication.lock"):
        path = directory / "PUBLICATION_STATE.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_state_sha256:
            raise ValueError("Publication state changed; reconcile before updating")
        previous = json.loads(raw)
        if previous.get("episode_id") != state.get("episode_id"):
            raise ValueError("Publication receipt episode changed")
        _snapshot(directory, previous)
        _snapshot(directory, state)
        atomic_json(path, state)
