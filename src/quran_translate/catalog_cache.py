"""Validate whole catalog packages before reusing assembled media."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from .config import file_sha256


def catalog_contract(inputs: list[Path], displays: list[Path], context: Any) -> dict[str, Any]:
    return {
        "version": "catalog-input-contract-v2",
        "source_sha256s": [file_sha256(path) for path in inputs],
        "display_sha256s": [file_sha256(path) for path in displays],
        "context_sha256": hashlib.sha256(json.dumps(
            context, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")).hexdigest(),
    }


def validate_cached_catalog(
    output: Path, qa_path: Path, contract: dict[str, Any], language: str,
    restore_sidecars: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any] | None:
    captions = output.with_suffix(f".{language}.srt")
    metadata = output.with_suffix(".metadata.json")
    paths = (output, qa_path, captions, metadata)
    if not any(path.exists() for path in paths):
        return None
    if not output.is_file() or not qa_path.is_file():
        raise ValueError("Incomplete existing catalog; preserve media and repair sidecars explicitly")
    prior = json.loads(qa_path.read_text(encoding="utf-8"))
    if prior.get("input_contract") != contract:
        raise ValueError("Catalog inputs are unverified or changed; explicit versioned repair required")
    for key, path in (("sha256", output), ("captions_sha256", captions), ("metadata_sha256", metadata)):
        if not path.exists() and path != output:
            continue
        if prior.get(key) != file_sha256(path):
            raise ValueError(f"Catalog artifact integrity failure: {path}")
    if prior.get("decode_passed") is not True:
        raise ValueError("Cached catalog has not passed full decode")
    if not captions.is_file() or not metadata.is_file():
        if restore_sidecars is None:
            raise ValueError("Incomplete existing catalog; no validated sidecar recovery supplied")
        with TemporaryDirectory(prefix=".sidecar-recovery-", dir=output.parent) as directory:
            generated_captions = Path(directory) / captions.name
            generated_metadata = Path(directory) / metadata.name
            restore_sidecars(generated_captions, generated_metadata)
            pairs = [("captions_sha256", generated_captions, captions),
                     ("metadata_sha256", generated_metadata, metadata)]
            for key, generated, _destination in pairs:
                if not generated.is_file() or file_sha256(generated) != prior.get(key):
                    raise ValueError("Recovered sidecar differs from the verified original")
            for _key, generated, destination in pairs:
                if not destination.exists():
                    os.link(generated, destination)
    return prior
