#!/usr/bin/env python3
"""Repair one failed Urdu video layout without changing frozen production code."""

from __future__ import annotations

import argparse
import shutil

from quran_translate import urdu_video_production as production
from quran_translate import urdu_video_render as renderer


MAX_CLAUSE_CHARACTERS = 145
ORIGINAL_SPLITTER = renderer._split_clauses


def split_oversized_clauses(text: str) -> list[str]:
    pieces = ORIGINAL_SPLITTER(text)
    result: list[str] = []
    for piece in pieces:
        if len(piece) <= MAX_CLAUSE_CHARACTERS:
            result.append(piece)
            continue
        pending: list[str] = []
        for word in piece.split():
            candidate = " ".join([*pending, word])
            if pending and len(candidate) > MAX_CLAUSE_CHARACTERS:
                result.append(" ".join(pending))
                pending = []
            pending.append(word)
        if pending:
            result.append(" ".join(pending))
    return result


def build_repaired_display(alignment: dict[str, object]) -> dict[str, object]:
    display = ORIGINAL_DISPLAY_BUILDER(alignment)
    for event in display["events"]:
        lines = event["lines"]
        if event["kind"] != "clause" or len(lines) <= 2:
            continue
        active = next(index for index, line in enumerate(lines) if line["active"])
        start = min(max(0, active), len(lines) - 2)
        event["lines"] = [dict(line) for line in lines[start : start + 2]]
        active = next(index for index, line in enumerate(event["lines"]) if line["active"])
        if not any(line.get("ref") for line in event["lines"]):
            event["lines"][active]["ref"] = event["active_ref"]
    return display


ORIGINAL_DISPLAY_BUILDER = production.build_urdu_display_events


def repair(*, run_id: str, unit_index: int) -> dict[str, object]:
    root = production._run_root(run_id)
    state = production._read_json(root / "RUN.json")
    jobs = state.get("jobs", [])
    if not 1 <= unit_index <= len(jobs):
        raise RuntimeError(f"Invalid unit index: {unit_index}")
    job = jobs[unit_index - 1]
    if job.get("render") != "failed" or "cannot fit" not in str(job.get("render_error", "")):
        raise RuntimeError("Layout recovery requires a failed text-fit render")

    marker_path = root / f"RENDER_LAYOUT_REPAIR_{unit_index:04d}.json"
    if marker_path.exists():
        raise RuntimeError(f"Layout recovery was already attempted: {marker_path}")

    work = root / "segment-work" / f"{unit_index:04d}"
    archive_parent = root / "render-recovery"
    prior_archives = sorted(archive_parent.glob(f"{unit_index:04d}-failed-layout-attempt-*"))
    if not work.is_dir() or len(prior_archives) >= 2:
        raise RuntimeError("Failed render work is missing or bounded recovery is exhausted")
    archive = archive_parent / (
        f"{unit_index:04d}-failed-layout-attempt-{len(prior_archives) + 1:04d}"
    )
    archive_parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(work), str(archive))

    original_splitter = renderer._split_clauses
    original_builder = production.build_urdu_display_events
    renderer._split_clauses = split_oversized_clauses
    production.build_urdu_display_events = build_repaired_display
    try:
        result = production.render(run_id=run_id, unit_index=unit_index)
    finally:
        renderer._split_clauses = original_splitter
        production.build_urdu_display_events = original_builder

    state = production._read_json(root / "RUN.json")
    job = state["jobs"][unit_index - 1]
    if job.get("render") != "complete":
        raise RuntimeError("Layout recovery did not produce a complete render")

    qa_path = work / "QA.json"
    qa = production._read_json(qa_path)
    repair_evidence = {
        "version": "quran-urdu-video-layout-repair-v1",
        "repaired_at": production.utc_now(),
        "run_id": run_id,
        "unit_index": unit_index,
        "unit_id": job["unit_id"],
        "strategy": "split-oversized-clauses-and-cap-visible-clause-context-at-two",
        "failed_work_archives": [str(path) for path in [*prior_archives, archive]],
        "production_implementation_unchanged": True,
    }
    qa["layout_repair"] = repair_evidence
    production.atomic_json(qa_path, qa)
    production.atomic_json(marker_path, repair_evidence)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--unit-index", required=True, type=int)
    args = parser.parse_args()
    print(repair(run_id=args.run_id, unit_index=args.unit_index))


if __name__ == "__main__":
    main()
