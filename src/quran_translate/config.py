"""Project paths and small configuration helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
WORK_DIR = DATA_DIR / "work"
OUTPUT_DIR = PROJECT_ROOT / "output"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DEFAULT_DB_PATH = WORK_DIR / "quran.sqlite"
DEFAULT_SOURCE_XML = SOURCE_DIR / "quran-uthmani-min.xml"
DEFAULT_PHILOLOGICAL_PROMPT = PROMPTS_DIR / "philological-v3.md"
DEFAULT_OUTPUT_CONTRACT_PROMPT = PROMPTS_DIR / "output-contract-v2.md"

DEFAULT_MODEL = "gemini-3-pro-preview"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_TARGET_CHARS = 3200
DEFAULT_CONTEXT_BEFORE = 1
DEFAULT_CONTEXT_AFTER = 1


def ensure_dirs() -> None:
    for path in (SOURCE_DIR, WORK_DIR, OUTPUT_DIR, PROMPTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_dotenv(path: Path | None = None) -> None:
    """Load a simple KEY=VALUE .env file without requiring python-dotenv."""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def read_prompt_files() -> tuple[str, str, str]:
    philological = DEFAULT_PHILOLOGICAL_PROMPT.read_text(encoding="utf-8").strip()
    contract = DEFAULT_OUTPUT_CONTRACT_PROMPT.read_text(encoding="utf-8").strip()
    prompt_hash = text_sha256(philological + "\n\n---\n\n" + contract)
    return philological, contract, prompt_hash
