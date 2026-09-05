#!/usr/bin/env python3
"""Build the first short-form editorial catalog from the canonical release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quran_translate.short_form_strategy import build_short_form_catalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=PROJECT_ROOT
        / "output/release/quran-translation-v2.4.1/quran-listening-edition.json",
    )
    parser.add_argument(
        "--strategy",
        type=Path,
        default=PROJECT_ROOT / "configs/short_form_strategy_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "output/short-form/strategy/SHORT_FORM_CANDIDATE_CATALOG.json",
    )
    args = parser.parse_args()
    payload = build_short_form_catalog(
        release_path=args.release,
        strategy_path=args.strategy,
        output_path=args.output,
    )
    print(
        json.dumps(
            {"output": str(args.output), "totals": payload["totals"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
