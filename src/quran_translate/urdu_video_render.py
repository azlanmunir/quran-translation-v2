"""RTL-aware Urdu Quran video frames and display timelines."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .video_alignment import AlignmentError


WIDTH = 3840
HEIGHT = 2160
BACKGROUND = (15, 17, 19)
PRIMARY = (240, 236, 226)
SECONDARY = (148, 151, 154)
MUTED = (88, 92, 98)
ACCENT = (211, 178, 91)
LATIN_FONT_PATH = Path("/System/Library/Fonts/Avenir Next.ttc")
URDU_FONT_NAME = "Noto Nastaliq Urdu"
URDU_FONT_PATH = Path("/System/Library/Fonts/NotoNastaliq.ttc")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[۔؟؛:!])\s+|(?<=،)\s+")


def _ref_key(ref: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{1,3}):(\d{1,3})", ref)
    if not match:
        raise AlignmentError(f"Malformed Quran reference: {ref}")
    return int(match.group(1)), int(match.group(2))


def _split_clauses(text: str) -> list[str]:
    pieces = [piece.strip() for piece in CLAUSE_SPLIT_RE.split(text) if piece.strip()]
    if len(pieces) == 1 and len(text) > 190:
        words = text.split()
        pieces = []
        pending: list[str] = []
        for word in words:
            candidate = " ".join(pending + [word])
            if pending and len(candidate) > 145:
                pieces.append(" ".join(pending))
                pending = []
            pending.append(word)
        if pending:
            pieces.append(" ".join(pending))
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) + 1 <= 145:
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    return merged or [text]


def build_urdu_display_events(alignment: dict[str, Any]) -> dict[str, Any]:
    """Build stable ayah panels with proportional clause timing for long ayahs."""

    spans = alignment.get("spans")
    if not isinstance(spans, list):
        raise AlignmentError("Normalized alignment is missing spans")
    ayahs = [
        span
        for span in spans
        if span.get("kind") == "ayah" and isinstance(span.get("ref"), str)
    ]
    if not ayahs:
        raise AlignmentError("Alignment has no ayah spans")

    groups: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            groups.append(list(pending))
            pending.clear()

    for span in ayahs:
        text = str(span["text"])
        duration = float(span["end"]) - float(span["start"])
        is_short = len(text) <= 105 and duration <= 12
        candidate_chars = sum(len(str(item["text"])) for item in pending) + len(text)
        if not is_short:
            flush()
            groups.append([span])
        else:
            if pending and (len(pending) >= 3 or candidate_chars > 225):
                flush()
            pending.append(span)
    flush()

    events: list[dict[str, Any]] = []
    for group in groups:
        span = group[0]
        text = str(span["text"])
        duration = float(span["end"]) - float(span["start"])
        if len(group) == 1 and (len(text) > 220 or duration > 20):
            clauses = _split_clauses(text)
            total_chars = sum(max(1, len(item)) for item in clauses)
            elapsed_chars = 0
            for clause_index, clause in enumerate(clauses):
                start = float(span["start"]) + duration * elapsed_chars / total_chars
                elapsed_chars += max(1, len(clause))
                end = float(span["start"]) + duration * elapsed_chars / total_chars
                window_start = max(0, min(clause_index - 1, len(clauses) - 3))
                visible = clauses[window_start : window_start + 3]
                events.append(
                    {
                        "kind": "clause",
                        "start": start,
                        "source_end": end,
                        "active_ref": str(span["ref"]),
                        "clause_index": clause_index + 1,
                        "clause_count": len(clauses),
                        "caption": clause,
                        "lines": [
                            {
                                "ref": str(span["ref"]) if index == 0 else "",
                                "text": item,
                                "active": index == clause_index - window_start,
                            }
                            for index, item in enumerate(visible)
                        ],
                    }
                )
        else:
            for active in group:
                events.append(
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
                            for item in group
                        ],
                    }
                )

    events.sort(key=lambda item: float(item["start"]))
    clip_start = max(0.0, float(ayahs[0]["start"]) - 0.12)
    clip_end = float(ayahs[-1]["end"]) + 0.25
    for index, event in enumerate(events):
        next_start = (
            float(events[index + 1]["start"])
            if index + 1 < len(events)
            else clip_end
        )
        event["end"] = max(float(event["source_end"]), next_start)
        event["start"] = round(max(0.0, float(event["start"]) - clip_start), 3)
        event["end"] = round(max(0.0, float(event["end"]) - clip_start), 3)
    return {
        "version": "quran-urdu-video-display-events-v1",
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
        "events": events,
    }


def _latin_font(size: int, *, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(LATIN_FONT_PATH), size=size, index=index)


class PangoUrduRenderer:
    """Render shaped Nastaliq text through Pango and cache identical layers."""

    def __init__(self, cache_dir: Path) -> None:
        if not URDU_FONT_PATH.is_file():
            raise AlignmentError(f"Urdu font is missing: {URDU_FONT_PATH}")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        text: str,
        *,
        size: int,
        width: int,
        color: tuple[int, int, int],
        line_spacing: float = 1.12,
    ) -> Image.Image:
        digest = hashlib.sha256(
            f"{text}\0{size}\0{width}\0{color}\0{line_spacing}".encode("utf-8")
        ).hexdigest()
        path = self.cache_dir / f"{digest}.png"
        if not path.is_file():
            subprocess.run(
                [
                    "pango-view",
                    "--no-display",
                    "--backend=cairo",
                    "--pixels",
                    "--rtl",
                    "--align=right",
                    "--wrap=word-char",
                    f"--width={width}",
                    f"--font={URDU_FONT_NAME} {size}",
                    f"--foreground=#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    "--background=transparent",
                    "--margin=0",
                    f"--line-spacing={line_spacing}",
                    f"--text={text}",
                    f"--output={path}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        with Image.open(path) as image:
            return image.convert("RGBA")


def _line_layers(
    renderer: PangoUrduRenderer,
    lines: list[dict[str, Any]],
    *,
    width: int,
    max_height: int,
) -> tuple[int, list[tuple[dict[str, Any], Image.Image]], int]:
    for size in range(124, 75, -4):
        layers = [
            (
                line,
                renderer.render(
                    str(line["text"]),
                    size=size,
                    width=width,
                    color=PRIMARY if line["active"] else SECONDARY,
                ),
            )
            for line in lines
        ]
        total = sum(layer.height + 42 for _, layer in layers) - 42
        if total <= max_height:
            return size, layers, total
    raise AlignmentError("Urdu text cannot fit the 16:9 panel at a readable size")


def render_urdu_frames(
    *,
    display: dict[str, Any],
    output_dir: Path,
    surah_name_ar: str,
    surah_name_en: str,
    para_number: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = PangoUrduRenderer(output_dir.parent / "pango-cache")
    header_urdu = renderer.render(
        "قرآن اردو ترجمہ",
        size=46,
        width=900,
        color=MUTED,
    )
    surah_ar = renderer.render(
        f"سورۃ {surah_name_ar}",
        size=68,
        width=1500,
        color=PRIMARY,
    )
    events = display["events"]
    frame_paths: list[Path] = []
    for index, event in enumerate(events, start=1):
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        image.paste(header_urdu, (WIDTH - 250 - header_urdu.width, 112), header_urdu)
        image.paste(surah_ar, (WIDTH - 250 - surah_ar.width, 205), surah_ar)
        draw.text((250, 220), surah_name_en, font=_latin_font(62), fill=PRIMARY)
        draw.text(
            (250, 305),
            f"PARA {para_number}/30",
            font=_latin_font(36),
            fill=SECONDARY,
        )
        if event["kind"] == "clause":
            draw.text(
                (250, 365),
                f"AYAH {event['active_ref']}  {event['clause_index']}/{event['clause_count']}",
                font=_latin_font(32),
                fill=MUTED,
            )

        body_top = 500
        body_height = 1260
        body_width = 2920
        _, layers, total_height = _line_layers(
            renderer,
            event["lines"],
            width=body_width,
            max_height=body_height,
        )
        y = body_top + max(0, (body_height - total_height) // 2)
        for line, layer in layers:
            active = bool(line["active"])
            if active:
                draw.rectangle(
                    (WIDTH - 190, y + 18, WIDTH - 178, y + layer.height - 18),
                    fill=ACCENT,
                )
            image.paste(layer, (WIDTH - 250 - body_width, y), layer)
            if line.get("ref"):
                draw.text(
                    (250, y + max(10, layer.height // 2 - 28)),
                    str(line["ref"]),
                    font=_latin_font(38),
                    fill=ACCENT if active else MUTED,
                )
            y += layer.height + 42

        progress = index / max(1, len(events))
        draw.line((250, 1988, WIDTH - 250, 1988), fill=(45, 48, 52), width=3)
        draw.line(
            (250, 1988, 250 + int((WIDTH - 500) * progress), 1988),
            fill=ACCENT,
            width=5,
        )
        draw.text(
            (250, 2035),
            f"PARA {para_number}/30",
            font=_latin_font(34),
            fill=MUTED,
        )
        draw.text(
            (WIDTH - 250, 2035),
            str(event.get("active_ref") or ""),
            font=_latin_font(34),
            fill=MUTED,
            anchor="ra",
        )
        path = output_dir / f"frame-{index:04d}.png"
        image.save(path, format="PNG", optimize=True)
        frame_paths.append(path)
    return frame_paths
