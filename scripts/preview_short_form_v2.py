#!/usr/bin/env python3
"""Non-publishing renderer smoke test using an already validated source episode."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quran_translate.production_packets import atomic_json
from quran_translate.short_form_production import (
    _audit_audio_semantics,
    _audit_video,
    _extract_audio,
    _render_layers,
    _render_video,
    _write_platform_copy,
    validate_episode_spec,
)
from quran_translate.short_form_workflow import VERSION, read_json, validate_timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    original = read_json(args.spec)
    validated = validate_episode_spec(root, original)
    spec = copy.deepcopy(original)
    spec["version"] = VERSION
    spec["production_status"] = "diagnostic_only_not_for_publication"
    spec["episode_id"] = "preview-v2-" + uuid.uuid4().hex[:12]
    spec["creative"].update(
        quote_start_seconds=1.2, hook_end_seconds=1.2, closing_question=""
    )
    spec["creative"]["hook"] = "DOES IT COUNT?"
    spec["creative"]["closing"] = "SMALL STILL COUNTS."
    spec["render"]["duration_seconds"] = round(
        spec["source"]["clip_end_seconds"] - spec["source"]["clip_start_seconds"] + 2.2,
        3,
    )
    spec["platform_copy"].setdefault("youtube", "Diagnostic, not for publication")
    spec["platform_copy"].setdefault("youtube_title", "Diagnostic, not for publication")
    validate_timing(spec)
    output = root / "output/short-form/previews" / spec["episode_id"]
    output.mkdir(parents=True)
    atomic_json(output / "DIAGNOSTIC_SPEC.json", spec)
    layers = _render_layers(spec, output / "layers")
    wav = output / "quote.wav"
    _extract_audio(spec, validated["video_path"], wav)
    spoken = _audit_audio_semantics(spec, wav)
    master = output / "preview.mp4"
    _render_video(spec, root, layers, wav, master)
    _write_platform_copy(spec, output)
    qa = _audit_video(
        spec,
        output,
        master,
        dict.fromkeys(("instagram", "tiktok", "youtube"), master),
        spoken,
    )
    qa["diagnostic_only_not_for_publication"] = True
    atomic_json(output / "QA.json", qa)
    print(json.dumps({"output": str(output), "qa": qa}, indent=2))


if __name__ == "__main__":
    main()
