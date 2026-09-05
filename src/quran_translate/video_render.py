"""Adaptive, text-first Quran video rendering and publication QA."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

from .config import file_sha256
from .production_packets import atomic_json, atomic_text
from .video_alignment import AlignmentError


WIDTH = 3840
HEIGHT = 2160
FINAL_WIDTH = 1920
FINAL_HEIGHT = 1080
BACKGROUND = (15, 17, 19)
PRIMARY = (240, 236, 226)
SECONDARY = (126, 131, 137)
MUTED = (88, 92, 98)
ACCENT = (211, 178, 91)
FONT_PATH = Path("/System/Library/Fonts/Avenir Next.ttc")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?;:—])\s+|(?<=,)\s+")
MAX_DURATION_DELTA_SECONDS = 0.3
MIN_CONTENT_END_MARGIN_SECONDS = -0.05
TARGET_LOUDNESS_LUFS = -18.0
MAX_LOUDNESS_DELTA_LUFS = 0.5


def _ref_key(ref: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{1,3}):(\d{1,3})", ref)
    if not match:
        raise AlignmentError(f"Malformed Quran reference: {ref}")
    return int(match.group(1)), int(match.group(2))


def _is_selected(ref: str, start_ref: str | None, end_ref: str | None) -> bool:
    key = _ref_key(ref)
    return (start_ref is None or key >= _ref_key(start_ref)) and (
        end_ref is None or key <= _ref_key(end_ref)
    )


def _clauses(span: dict[str, Any], words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = str(span["text"])
    pieces = [piece for piece in CLAUSE_SPLIT_RE.split(text) if piece]
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + 1 + len(piece) <= 180:
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    if len(merged) == 1 and len(text) > 210:
        raw_words = text.split()
        merged = []
        current: list[str] = []
        for word in raw_words:
            if current and len(" ".join(current + [word])) > 165:
                merged.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            merged.append(" ".join(current))

    clauses: list[dict[str, Any]] = []
    search_at = 0
    span_start = int(span["start_char"])
    for piece in merged:
        local_start = text.find(piece, search_at)
        if local_start < 0:
            raise AlignmentError(f"Cannot locate clause text inside {span.get('ref')}")
        local_end = local_start + len(piece)
        search_at = local_end
        global_start = span_start + local_start
        global_end = span_start + local_end
        relevant = [
            word
            for word in words
            if int(word["start_char"]) < global_end
            and int(word["end_char"]) > global_start
        ]
        if relevant:
            start = float(relevant[0]["start"])
            end = float(relevant[-1]["end"])
        else:
            fraction_start = local_start / max(1, len(text))
            fraction_end = local_end / max(1, len(text))
            duration = float(span["end"]) - float(span["start"])
            start = float(span["start"]) + duration * fraction_start
            end = float(span["start"]) + duration * fraction_end
        clauses.append(
            {
                "text": piece,
                "start": start,
                "end": end,
                "start_char": global_start,
                "end_char": global_end,
            }
        )
    return clauses


def _short_group(span: dict[str, Any]) -> bool:
    return len(str(span["text"])) <= 95 and float(span["end"]) - float(span["start"]) <= 10


def build_display_events(
    alignment: dict[str, Any],
    *,
    start_ref: str | None = None,
    end_ref: str | None = None,
) -> dict[str, Any]:
    """Turn ayah timing into stable grouped panels and clause advances."""

    spans = alignment.get("spans")
    words = alignment.get("words")
    if not isinstance(spans, list) or not isinstance(words, list):
        raise AlignmentError("Normalized alignment is missing spans or words")
    ayahs = [
        span
        for span in spans
        if span.get("kind") == "ayah"
        and isinstance(span.get("ref"), str)
        and _is_selected(str(span["ref"]), start_ref, end_ref)
    ]
    if not ayahs:
        raise AlignmentError("Selected pilot range has no aligned ayahs")

    include_announcement = start_ref is None
    announcements = [span for span in spans if span.get("kind") == "surah_announcement"]
    units: list[list[dict[str, Any]]] = []
    pending_short: list[dict[str, Any]] = []

    def flush_short() -> None:
        if pending_short:
            units.append(list(pending_short))
            pending_short.clear()

    for span in ayahs:
        if _short_group(span):
            candidate_chars = sum(len(str(item["text"])) for item in pending_short) + len(
                str(span["text"])
            )
            if pending_short and (len(pending_short) >= 4 or candidate_chars > 250):
                flush_short()
            pending_short.append(span)
        else:
            flush_short()
            units.append([span])
    flush_short()

    absolute_events: list[dict[str, Any]] = []
    if include_announcement and announcements:
        announcement = announcements[0]
        absolute_events.append(
            {
                "kind": "title",
                "start": 0.0,
                "source_end": float(announcement["end"]),
                "active_ref": None,
                "caption": str(announcement["text"]),
                "lines": [],
            }
        )

    for unit in units:
        span = unit[0]
        is_long = len(str(span["text"])) > 340 or float(span["end"]) - float(span["start"]) > 24
        if len(unit) == 1 and is_long:
            clauses = _clauses(span, words)
            for clause_index, clause in enumerate(clauses):
                window_start = max(0, min(clause_index - 1, len(clauses) - 3))
                visible = clauses[window_start : window_start + 3]
                absolute_events.append(
                    {
                        "kind": "clause",
                        "start": float(clause["start"]),
                        "source_end": float(clause["end"]),
                        "active_ref": str(span["ref"]),
                        "clause_index": clause_index + 1,
                        "clause_count": len(clauses),
                        "caption": str(clause["text"]),
                        "lines": [
                            {
                                "ref": str(span["ref"]) if index == 0 else "",
                                "text": str(item["text"]),
                                "active": item is clause,
                            }
                            for index, item in enumerate(visible)
                        ],
                    }
                )
        else:
            for active in unit:
                absolute_events.append(
                    {
                        "kind": "ayah",
                        "start": float(active["start"]),
                        "source_end": float(active["end"]),
                        "active_ref": str(active["ref"]),
                        "caption": str(active["text"]),
                        "lines": [
                            {
                                "ref": str(item["ref"]),
                                "text": str(item["text"]),
                                "active": item is active,
                            }
                            for item in unit
                        ],
                    }
                )

    absolute_events.sort(key=lambda item: float(item["start"]))
    clip_start = 0.0 if include_announcement else max(0.0, float(ayahs[0]["start"]) - 0.12)
    clip_end = float(ayahs[-1]["end"]) + 0.25
    for index, event in enumerate(absolute_events):
        next_start = (
            float(absolute_events[index + 1]["start"])
            if index + 1 < len(absolute_events)
            else clip_end
        )
        event["end"] = max(float(event["source_end"]), next_start)
        event["start"] = round(max(0.0, float(event["start"]) - clip_start), 3)
        event["end"] = round(max(0.0, float(event["end"]) - clip_start), 3)
    return {
        "version": "quran-video-display-events-v1",
        "engine": alignment.get("engine"),
        "audio_sha256": alignment.get("audio_sha256"),
        "transcript_sha256": alignment.get("transcript_sha256"),
        "selection": {
            "start_ref": str(ayahs[0]["ref"]),
            "end_ref": str(ayahs[-1]["ref"]),
            "clip_start": round(clip_start, 3),
            "clip_end": round(clip_end, 3),
            "duration": round(clip_end - clip_start, 3),
        },
        "events": absolute_events,
    }


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def _line_layout(
    draw: ImageDraw.ImageDraw,
    lines: list[dict[str, Any]],
    max_width: int,
    max_height: int,
) -> tuple[ImageFont.FreeTypeFont, list[tuple[dict[str, Any], list[str]]], int]:
    for size in range(132, 79, -4):
        font = _font(size)
        wrapped = [(line, _wrap(draw, str(line["text"]), font, max_width)) for line in lines]
        line_height = int(size * 1.30)
        total = sum(len(parts) * line_height + 60 for _, parts in wrapped) - 60
        if total <= max_height:
            return font, wrapped, total
    raise AlignmentError("Text cannot fit the 16:9 panel at the minimum readable size")


def render_frames(
    *,
    display: dict[str, Any],
    output_dir: Path,
    surah_name: str,
    surah_meaning: str,
    juz_number: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = display["events"]
    frame_paths: list[Path] = []
    for index, event in enumerate(events, start=1):
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        if event["kind"] == "title":
            title_font = _font(184)
            meaning_font = _font(82)
            label_font = _font(42)
            draw.text((WIDTH // 2, 610), surah_name, font=title_font, fill=PRIMARY, anchor="mm")
            draw.text(
                (WIDTH // 2, 820),
                surah_meaning,
                font=meaning_font,
                fill=SECONDARY,
                anchor="mm",
            )
            draw.line((WIDTH // 2 - 180, 940, WIDTH // 2 + 180, 940), fill=ACCENT, width=4)
            draw.text(
                (WIDTH // 2, 1040),
                str(event["caption"]),
                font=label_font,
                fill=SECONDARY,
                anchor="mm",
            )
        else:
            eyebrow = _font(38)
            heading = _font(64)
            ref_font = _font(38)
            draw.text((250, 165), "THE QURAN  •  ENGLISH LISTENING EDITION", font=eyebrow, fill=MUTED)
            draw.text((250, 245), surah_name, font=heading, fill=PRIMARY)
            draw.text((250, 330), surah_meaning, font=eyebrow, fill=SECONDARY)
            if event["kind"] == "clause":
                draw.text(
                    (WIDTH - 250, 260),
                    f"AYAH {event['active_ref']}  •  {event['clause_index']}/{event['clause_count']}",
                    font=eyebrow,
                    fill=SECONDARY,
                    anchor="ra",
                )

            body_top = 510
            body_height = 1220
            body_width = 3040
            body_font, wrapped, total_height = _line_layout(
                draw, event["lines"], body_width, body_height
            )
            line_height = int(body_font.size * 1.30)
            y = body_top + max(0, (body_height - total_height) // 2)
            for line, parts in wrapped:
                active = bool(line["active"])
                color = PRIMARY if active else SECONDARY
                ref_color = ACCENT if active else MUTED
                if active:
                    draw.rectangle((178, y + 8, 190, y + line_height * len(parts) - 8), fill=ACCENT)
                if line.get("ref"):
                    draw.text((250, y + 10), str(line["ref"]), font=ref_font, fill=ref_color)
                text_y = y
                for part in parts:
                    draw.text((560, text_y), part, font=body_font, fill=color)
                    text_y += line_height
                y += len(parts) * line_height + 60

        progress = index / max(1, len(events))
        draw.line((250, 1988, WIDTH - 250, 1988), fill=(45, 48, 52), width=3)
        draw.line((250, 1988, 250 + int((WIDTH - 500) * progress), 1988), fill=ACCENT, width=5)
        footer_font = _font(34)
        draw.text((250, 2035), f"JUZ {juz_number}", font=footer_font, fill=MUTED)
        draw.text(
            (WIDTH - 250, 2035),
            str(event.get("active_ref") or ""),
            font=footer_font,
            fill=MUTED,
            anchor="ra",
        )
        path = output_dir / f"frame-{index:04d}.png"
        image.save(path, format="PNG", optimize=True)
        frame_paths.append(path)
    return frame_paths


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(display: dict[str, Any], output_path: Path) -> Path:
    blocks: list[str] = []
    for index, event in enumerate(display["events"], start=1):
        blocks.append(
            f"{index}\n{_srt_time(float(event['start']))} --> {_srt_time(float(event['end']))}\n"
            f"{event['caption']}"
        )
    atomic_text(output_path, "\n\n".join(blocks) + "\n")
    return output_path


def validate_srt_identity(display: dict[str, Any], srt_path: Path) -> dict[str, Any]:
    """Require captions to be an exact serialization of the display timeline."""

    content = srt_path.read_text(encoding="utf-8").strip()
    blocks = re.split(r"\n\s*\n", content) if content else []
    events = display.get("events", [])
    if len(blocks) != len(events):
        raise AlignmentError("SRT cue count differs from display event count")
    for expected_index, (block, event) in enumerate(zip(blocks, events), start=1):
        lines = block.splitlines()
        if len(lines) < 3 or lines[0] != str(expected_index):
            raise AlignmentError(f"SRT cue {expected_index} has a malformed index")
        expected_timing = f"{_srt_time(float(event['start']))} --> {_srt_time(float(event['end']))}"
        if lines[1] != expected_timing:
            raise AlignmentError(f"SRT cue {expected_index} has drifted timestamps")
        if "\n".join(lines[2:]) != str(event["caption"]):
            raise AlignmentError(f"SRT cue {expected_index} has drifted text")
    return {
        "cue_count": len(blocks),
        "timeline_identity": True,
        "srt_sha256": file_sha256(srt_path),
    }


def write_mobile_review(
    *,
    frames: list[Path],
    output_dir: Path,
    width: int = 360,
    max_samples: int = 12,
) -> dict[str, Any]:
    """Write deterministic phone-width previews and a contact sheet for visual QA."""

    if not frames:
        raise AlignmentError("Mobile review requires at least one rendered frame")
    if width < 240 or max_samples < 1:
        raise AlignmentError("Mobile review dimensions are invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(frames) <= max_samples:
        indices = list(range(len(frames)))
    else:
        indices = sorted(
            {
                round(index * (len(frames) - 1) / (max_samples - 1))
                for index in range(max_samples)
            }
        )

    previews: list[dict[str, Any]] = []
    resized_images: list[Image.Image] = []
    for index in indices:
        source = frames[index]
        with Image.open(source) as raw:
            image = raw.convert("RGB")
            height = round(image.height * width / image.width)
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
        path = output_dir / source.name
        resized.save(path, format="PNG", optimize=True)
        resized_images.append(resized)
        previews.append(
            {
                "frame_index": index + 1,
                "source": str(source),
                "source_sha256": file_sha256(source),
                "preview": str(path),
                "preview_sha256": file_sha256(path),
                "width": resized.width,
                "height": resized.height,
            }
        )

    columns = min(2, len(resized_images))
    rows = math.ceil(len(resized_images) / columns)
    label_height = 28
    cell_height = resized_images[0].height + label_height
    sheet = Image.new("RGB", (width * columns, cell_height * rows), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    label_font = _font(18)
    for position, (preview, image) in enumerate(zip(previews, resized_images)):
        x = (position % columns) * width
        y = (position // columns) * cell_height
        draw.text(
            (x + 8, y + 5),
            f"FRAME {preview['frame_index']:04d}",
            font=label_font,
            fill=PRIMARY,
        )
        sheet.paste(image, (x, y + label_height))
    contact_sheet = output_dir / "contact-sheet.png"
    sheet.save(contact_sheet, format="PNG", optimize=True)
    payload = {
        "version": "quran-video-mobile-review-v1",
        "sample_strategy": "evenly-spaced-including-first-and-last",
        "source_frame_count": len(frames),
        "sample_count": len(previews),
        "target_width": width,
        "contact_sheet": str(contact_sheet),
        "contact_sheet_sha256": file_sha256(contact_sheet),
        "previews": previews,
        "human_review_required": True,
    }
    atomic_json(output_dir / "MOBILE_REVIEW.json", payload)
    return payload


def _chapter_time(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def build_youtube_chapters(
    *,
    alignment: dict[str, Any],
    start_ref: str,
    end_ref: str,
    catalog: str,
    target_interval_seconds: float = 600.0,
    surah_names: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Build ayah-snapped chapter points that satisfy YouTube chapter rules."""

    if catalog not in {"juz", "surah"}:
        raise AlignmentError("Chapter catalog must be 'juz' or 'surah'")
    spans = [
        span
        for span in alignment.get("spans", [])
        if span.get("kind") == "ayah"
        and isinstance(span.get("ref"), str)
        and _is_selected(str(span["ref"]), start_ref, end_ref)
    ]
    if not spans:
        raise AlignmentError("Chapter range contains no ayahs")
    spans.sort(key=lambda span: float(span["start"]))
    clip_start = float(spans[0]["start"])
    clip_end = float(spans[-1]["end"])
    candidates: list[dict[str, Any]] = [spans[0]]

    if catalog == "juz":
        previous_surah = int(spans[0]["surah"])
        for span in spans[1:]:
            surah = int(span["surah"])
            if surah != previous_surah:
                candidates.append(span)
                previous_surah = surah

    next_target = clip_start + target_interval_seconds
    for span in spans[1:]:
        if float(span["start"]) >= next_target:
            candidates.append(span)
            next_target = float(span["start"]) + target_interval_seconds

    candidates.sort(key=lambda span: float(span["start"]))
    deduplicated: list[dict[str, Any]] = []
    for span in candidates:
        relative = float(span["start"]) - clip_start
        if deduplicated and relative - float(deduplicated[-1]["relative_start"]) < 10.0:
            continue
        deduplicated.append({"span": span, "relative_start": relative})

    # YouTube requires at least three chapter timestamps, each at least ten seconds long.
    if len(deduplicated) < 3 or clip_end - clip_start < 30.0:
        return []

    chapters: list[dict[str, Any]] = []
    for index, item in enumerate(deduplicated):
        span = item["span"]
        ref = str(span["ref"])
        surah = int(span["surah"])
        if catalog == "juz" and (index == 0 or surah != int(deduplicated[index - 1]["span"]["surah"])):
            label = (surah_names or {}).get(surah, f"Surah {surah}")
        else:
            label = f"Ayah {ref}"
        chapters.append(
            {
                "timestamp": "0:00" if index == 0 else _chapter_time(float(item["relative_start"])),
                "seconds": round(float(item["relative_start"]), 3),
                "ref": ref,
                "label": label,
            }
        )
    return chapters


def format_youtube_chapters(chapters: list[dict[str, Any]]) -> str:
    return "\n".join(f"{row['timestamp']} {row['label']}" for row in chapters)


def _loudness_measure(audio_path: Path, start: float, duration: float) -> dict[str, float]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(audio_path),
            "-af",
            "loudnorm=I=-18:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = re.findall(r"\{\s*\"input_i\".*?\}", result.stderr, flags=re.DOTALL)
    if not matches:
        raise AlignmentError("ffmpeg did not report loudness measurements")
    raw = json.loads(matches[-1])
    required = (
        "input_i",
        "input_tp",
        "input_lra",
        "input_thresh",
        "output_i",
        "output_tp",
        "output_lra",
        "output_thresh",
        "target_offset",
    )
    try:
        return {key: float(raw[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise AlignmentError("ffmpeg returned incomplete numeric loudness measurements") from exc


def _linear_loudnorm(measurement: dict[str, float]) -> str:
    return (
        "loudnorm=I=-18:TP=-1.5:LRA=11:linear=true:"
        f"measured_I={measurement['input_i']}:"
        f"measured_TP={measurement['input_tp']}:"
        f"measured_LRA={measurement['input_lra']}:"
        f"measured_thresh={measurement['input_thresh']}:"
        f"offset={measurement['target_offset']}:print_format=summary"
    )


def render_video(
    *,
    display: dict[str, Any],
    frames: list[Path],
    audio_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if len(frames) != len(display["events"]):
        raise AlignmentError("Frame count does not match display events")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_path = output_path.with_suffix(".frames.txt")
    concat_lines: list[str] = []
    for frame, event in zip(frames, display["events"]):
        duration = float(event["end"]) - float(event["start"])
        if duration <= 0:
            raise AlignmentError("Display event has a non-positive duration")
        concat_lines.extend((f"file '{frame.resolve()}'", f"duration {duration:.6f}"))
    concat_lines.append(f"file '{frames[-1].resolve()}'")
    atomic_text(concat_path, "\n".join(concat_lines) + "\n")

    selection = display["selection"]
    start = float(selection["clip_start"])
    duration = float(selection["duration"])
    measurement = _loudness_measure(audio_path, start, duration)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(audio_path),
            "-vf",
            f"fps=30,scale={FINAL_WIDTH}:{FINAL_HEIGHT}:flags=lanczos,format=yuv420p",
            "-af",
            _linear_loudnorm(measurement),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-g",
            "60",
            "-keyint_min",
            "60",
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-movflags",
            "+faststart",
            "-t",
            f"{duration:.3f}",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )
    return {
        "output": str(output_path),
        "sha256": file_sha256(output_path),
        "source_audio_sha256": file_sha256(audio_path),
        "input_loudness": measurement,
    }


def validate_video(
    path: Path,
    expected_duration: float,
    *,
    minimum_content_duration: float | None = None,
) -> dict[str, Any]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    video = next((stream for stream in payload["streams"] if stream["codec_type"] == "video"), None)
    audio = next((stream for stream in payload["streams"] if stream["codec_type"] == "audio"), None)
    if video is None or audio is None:
        raise AlignmentError("Rendered video is missing audio or video")
    duration = float(payload["format"]["duration"])
    checks = {
        "duration_seconds": round(duration, 3),
        "duration_delta_seconds": round(abs(duration - expected_duration), 3),
        "duration_tolerance_seconds": MAX_DURATION_DELTA_SECONDS,
        "width": int(video["width"]),
        "height": int(video["height"]),
        "pixel_format": video.get("pix_fmt"),
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": int(audio.get("sample_rate", 0)),
        "audio_channels": int(audio.get("channels", 0)),
    }
    if checks["duration_delta_seconds"] > MAX_DURATION_DELTA_SECONDS:
        raise AlignmentError("Rendered video duration differs from selected audio")
    if minimum_content_duration is not None:
        margin = duration - minimum_content_duration
        checks["content_end_margin_seconds"] = round(margin, 3)
        if margin < MIN_CONTENT_END_MARGIN_SECONDS:
            raise AlignmentError("Rendered video cuts off aligned speech")
    if (checks["width"], checks["height"]) != (FINAL_WIDTH, FINAL_HEIGHT):
        raise AlignmentError("Rendered video has an unexpected resolution")
    if checks["pixel_format"] != "yuv420p" or checks["video_codec"] != "h264":
        raise AlignmentError("Rendered video is not YouTube-compatible H.264/yuv420p")
    if checks["audio_codec"] != "aac" or checks["audio_sample_rate"] != 44100:
        raise AlignmentError("Rendered video audio is not 44.1 kHz AAC")
    output_loudness = _loudness_measure(path, 0.0, duration)
    checks["output_loudness"] = output_loudness
    checks["loudness_target_lufs"] = TARGET_LOUDNESS_LUFS
    checks["loudness_tolerance_lufs"] = MAX_LOUDNESS_DELTA_LUFS
    if abs(output_loudness["input_i"] - TARGET_LOUDNESS_LUFS) > MAX_LOUDNESS_DELTA_LUFS:
        raise AlignmentError("Rendered video is outside the -18 LUFS listening target")
    if output_loudness["input_tp"] > -1.0:
        raise AlignmentError("Rendered video exceeds the true-peak protection ceiling")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True
    )
    checks["decode_passed"] = True
    return checks


def validate_encoded_timeline(
    *,
    video_path: Path,
    display: dict[str, Any],
    frames: list[Path],
    max_samples: int = 12,
    mean_difference_ceiling: float = 8.0,
) -> dict[str, Any]:
    """Sample the encoded video and prove that event panels appear at their intended times."""

    events = display.get("events", [])
    if len(events) != len(frames) or not events:
        raise AlignmentError("Encoded timeline validation requires one frame per event")
    sample_count = min(max_samples, len(events))
    if sample_count == 1:
        indices = [0]
    else:
        indices = sorted(
            {
                round(index * (len(events) - 1) / (sample_count - 1))
                for index in range(sample_count)
            }
        )
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="quran-video-timeline-") as directory:
        root = Path(directory)
        for index in indices:
            event = events[index]
            midpoint = (float(event["start"]) + float(event["end"])) / 2
            extracted = root / f"frame-{index + 1:04d}.png"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{midpoint:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(extracted),
                ],
                check=True,
            )
            with Image.open(frames[index]) as raw_expected:
                expected = raw_expected.convert("RGB").resize(
                    (FINAL_WIDTH, FINAL_HEIGHT), Image.Resampling.LANCZOS
                )
            with Image.open(extracted) as raw_actual:
                actual = raw_actual.convert("RGB")
            difference = ImageChops.difference(expected, actual)
            channel_means = ImageStat.Stat(difference).mean
            mean_difference = sum(channel_means) / len(channel_means)
            if mean_difference > mean_difference_ceiling:
                raise AlignmentError(
                    f"Encoded panel mismatch at event {index + 1}: {mean_difference:.3f}"
                )
            rows.append(
                {
                    "event_index": index + 1,
                    "timestamp": round(midpoint, 3),
                    "mean_pixel_difference": round(mean_difference, 3),
                }
            )
    return {
        "sample_strategy": "evenly-spaced-event-midpoints",
        "sample_count": len(rows),
        "mean_difference_ceiling": mean_difference_ceiling,
        "max_mean_pixel_difference": max(row["mean_pixel_difference"] for row in rows),
        "passed": True,
        "samples": rows,
    }


def write_metadata_kit(
    *,
    output_path: Path,
    surah_number: int,
    surah_name: str,
    surah_meaning: str,
    start_ref: str,
    end_ref: str,
    chapters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    range_label = f"{start_ref}–{end_ref}" if start_ref != end_ref else start_ref
    title = f"{surah_name} ({surah_number}) — {surah_meaning} | English Quran Audiobook"
    chapter_text = format_youtube_chapters(chapters or [])
    description = (
        f"{surah_name}, ayahs {range_label}, from The Quran: An English Listening Edition.\n\n"
        "English translation produced through a documented AI-assisted, evidence-audited workflow. "
        "Narration uses a selected synthetic voice; timings are aligned to the exact published text.\n\n"
        + (chapter_text + "\n" if chapter_text else "")
    )
    payload = {
        "version": "quran-youtube-metadata-v1",
        "title": title,
        "description": description,
        "playlist": "The Quran — English Listening Edition",
        "synthetic_content_disclosure": True,
        "language": "English",
        "caption_language": "English",
        "chapters": chapters or [],
        "thumbnail_label": f"{surah_number:03d}  {surah_name}",
    }
    atomic_json(output_path, payload)
    return payload
