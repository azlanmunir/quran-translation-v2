#!/usr/bin/env python3
"""Record verified platform receipts without losing prior publication state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quran_translate.config import file_sha256
from quran_translate.publication_receipts import record_publication_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("state_json", type=Path)
    parser.add_argument("--expected-state-sha256", required=True)
    args = parser.parse_args()
    state = json.loads(args.state_json.read_text(encoding="utf-8"))
    record_publication_state(
        args.episode_dir, state, expected_state_sha256=args.expected_state_sha256
    )
    print(json.dumps({"saved": True, "state_sha256": file_sha256(
        args.episode_dir / "PUBLICATION_STATE.json"
    )}))


if __name__ == "__main__":
    main()
