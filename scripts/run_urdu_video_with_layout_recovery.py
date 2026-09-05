#!/usr/bin/env python3
"""Run Urdu video production with bounded recovery for known text-fit failures."""

from __future__ import annotations

import argparse

from quran_translate import urdu_video_production as production
from scripts.repair_urdu_video_layout_unit import repair


def failed_layout_unit(run_id: str) -> int | None:
    state = production._read_json(production._run_root(run_id) / "RUN.json")
    failed = [job for job in state.get("jobs", []) if job.get("render") == "failed"]
    if not failed:
        return None
    if len(failed) != 1:
        raise RuntimeError(f"Expected at most one failed render, found {len(failed)}")
    job = failed[0]
    error = str(job.get("render_error", ""))
    if "Urdu text cannot fit the 16:9 panel at a readable size" not in error:
        raise RuntimeError(
            f"Non-layout render failure at unit {job.get('unit_index')}: {error}"
        )
    return int(job["unit_index"])


def run(run_id: str) -> None:
    repaired = set()
    while True:
        unit_index = failed_layout_unit(run_id)
        if unit_index is not None:
            if unit_index in repaired:
                raise RuntimeError(f"Layout repair exhausted for unit {unit_index}")
            repaired.add(unit_index)
            print(f"bounded layout recovery {unit_index:04d}", flush=True)
            repair(run_id=run_id, unit_index=unit_index)

        try:
            result = production.pipeline(run_id=run_id)
        except production.UrduVideoProductionError:
            if failed_layout_unit(run_id) is not None:
                continue
            raise
        if (
            result.get("run_id") == run_id
            and result.get("segments") == 343
            and result.get("para_videos") == 30
            and result.get("surah_videos") == 114
            and result.get("decode_checked") is True
        ):
            return
        raise RuntimeError(f"Pipeline did not return a validated completion: {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.run_id)


if __name__ == "__main__":
    main()
