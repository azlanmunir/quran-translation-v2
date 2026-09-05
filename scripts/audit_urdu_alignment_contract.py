#!/usr/bin/env python3
"""Read-only audit of preserved Urdu alignments against the current QA contract."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quran_translate.urdu_video_production import AUDIO_RUN_RELATIVE, _validate_alignment


def audit(source_root: Path, video_root: Path) -> dict:
    source_root, video_root = source_root.resolve(), video_root.resolve()
    audio_root = source_root / AUDIO_RUN_RELATIVE
    units = json.loads((audio_root / "UNITS.json").read_text(encoding="utf-8"))
    state = json.loads((audio_root / "RUN.json").read_text(encoding="utf-8"))
    jobs = {job["unit_id"]: job for job in state["jobs"]}
    results = []
    for unit in units:
        path = video_root / "alignments" / f"{unit['unit_index']:04d}" / "normalized.json"
        audio = Path(jobs[unit["unit_id"]]["normalized_path"])
        record = {"unit_id": unit["unit_id"], "unit_index": unit["unit_index"],
                  "refs": unit["refs"], "alignment": str(path), "audio": str(audio)}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record["mapped_word_coverage"] = payload.get("metrics", {}).get("mapped_word_coverage")
            _validate_alignment(unit, payload, audio)
        except Exception as exc:
            record.update(status="review_required", reason=f"{type(exc).__name__}: {exc}")
        else:
            record["status"] = "passed"
        results.append(record)
    return {
        "version": "urdu-alignment-contract-audit-v2",
        "read_only": True,
        "note": "A blocked alignment is not proof of missing narration. Do not regenerate without listening review.",
        "counts": dict(Counter(row["status"] for row in results)),
        "total": len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source_root, args.video_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
