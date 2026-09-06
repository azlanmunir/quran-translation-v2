#!/usr/bin/env python3
"""Plan, measure and preflight daily episodes. Does not upload or buy anything."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quran_translate.production_packets import atomic_json
from quran_translate.short_form_workflow import (
    build_plan,
    inventory,
    learning_summary,
    read_json,
    save_observation,
    validate_ready,
)
from quran_translate.state_safety import exclusive_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument(
        "--start", required=True, help="Local YYYY-MM-DD for this 30-day learning cycle"
    )
    record = sub.add_parser("observe")
    record.add_argument(
        "snapshot", type=Path, help="One evidence-backed post observation JSON"
    )
    sub.add_parser("status")
    sub.add_parser("summary")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("episode_dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "output/short-form"
    policy = read_json(root / "configs/short_form_strategy_v2.json")
    if args.command == "preflight":
        result = validate_ready(args.episode_dir.resolve())
    elif args.command == "observe":
        result = {
            "saved": str(
                save_observation(output / "analytics/v2", read_json(args.snapshot))
            )
        }
    elif args.command == "summary":
        rows = [
            read_json(path)
            for path in sorted((output / "analytics/v2/observations").glob("*.json"))
        ]
        result = learning_summary(rows, policy)
        atomic_json(output / "analytics/v2/ROLLING_SUMMARY.json", result)
    else:
        result = inventory(root, datetime.now(timezone.utc))
        if args.command == "plan":
            catalog = read_json(output / "strategy/SHORT_FORM_CANDIDATE_CATALOG.json")
            result = build_plan(catalog, policy, set(result["used_refs"]), args.start)
            destination = output / "strategy/v2" / f"PLAN-{args.start}.json"
            with exclusive_lock(output / "strategy/v2/.plan.lock"):
                if destination.exists():
                    result = read_json(destination)
                else:
                    atomic_json(destination, result)
        else:
            atomic_json(output / "strategy/v2/STATUS.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
