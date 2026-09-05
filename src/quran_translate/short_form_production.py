"""Render and audit context-approved vertical Quran episodes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .production_packets import atomic_json
from .publication_receipts import preserve_publication_receipt
from .state_safety import exclusive_lock


class ShortFormProductionError(RuntimeError):
    """Raised when an episode is not safe or ready to render."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShortFormProductionError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShortFormProductionError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        detail = result.stderr.strip()[-4000:]
        raise ShortFormProductionError(f"Command failed: {' '.join(command)}\n{detail}")


def _ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
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
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ShortFormProductionError(result.stderr.strip())
    return json.loads(result.stdout)


def _ref_key(ref: str) -> tuple[int, int]:
    try:
        surah, ayah = ref.split(":", 1)
        return int(surah), int(ayah)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ShortFormProductionError(f"Invalid Quran reference: {ref}") from exc


def _validate_segment_contract(
    alignment_path: Path,
    video_path: Path,
    source: dict[str, Any],
    start_ref: str,
    end_ref: str,
) -> Path:
    try:
        segment_index = int(source["segment_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ShortFormProductionError("Episode source lacks a valid segment index") from exc

    segment_label = f"{segment_index:04d}"
    expected_prefix = f"segment-{segment_label}-"
    expected_suffix = f"-{segment_label}.mp4"
    if alignment_path.parent.name != segment_label:
        raise ShortFormProductionError("Alignment does not belong to the declared segment")
    if not (
        video_path.name.startswith(expected_prefix)
        and video_path.name.endswith(expected_suffix)
    ):
        raise ShortFormProductionError(
            "Source video does not belong to the alignment segment"
        )

    run_root = alignment_path.parents[2]
    segment_qa_path = run_root / "segment-work" / segment_label / "QA.json"
    if not segment_qa_path.is_file():
        raise ShortFormProductionError(f"Missing segment QA receipt: {segment_qa_path}")
    segment_qa = _read_json(segment_qa_path)
    if int(segment_qa.get("chunk_index", -1)) != segment_index:
        raise ShortFormProductionError("Segment QA index differs from the episode source")
    if segment_qa.get("alignment", {}).get("sha256") != _sha256(alignment_path):
        raise ShortFormProductionError("Segment QA does not verify the selected alignment")
    if segment_qa.get("render", {}).get("sha256") != _sha256(video_path):
        raise ShortFormProductionError("Segment QA does not verify the selected video")
    if segment_qa.get("checks", {}).get("decode_passed") is not True:
        raise ShortFormProductionError("Selected source segment did not pass decode QA")

    selection = segment_qa.get("selection", {})
    selection_start = _ref_key(str(selection.get("start_ref", "")))
    selection_end = _ref_key(str(selection.get("end_ref", "")))
    target_start = _ref_key(start_ref)
    target_end = _ref_key(end_ref)
    if not selection_start <= target_start <= target_end <= selection_end:
        raise ShortFormProductionError("Target verse range is outside the source segment")
    return segment_qa_path


def validate_episode_spec(project_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Verify the episode against the canonical catalog and forced alignment."""

    if spec.get("production_status") != "approved_for_production":
        raise ShortFormProductionError("Episode lacks explicit production approval")

    source = spec.get("source")
    if not isinstance(source, dict):
        raise ShortFormProductionError("Episode has no source contract")
    catalog_path = project_root / str(source.get("catalog_path", ""))
    alignment_path = project_root / str(source.get("alignment_path", ""))
    video_path = project_root / str(source.get("video_path", ""))
    for path in (catalog_path, alignment_path, video_path):
        if not path.is_file():
            raise ShortFormProductionError(f"Missing source artifact: {path}")

    catalog = _read_json(catalog_path)
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list):
        raise ShortFormProductionError("Candidate catalog is malformed")
    candidate = next(
        (row for row in candidates if row.get("candidate_id") == spec.get("candidate_id")),
        None,
    )
    if not isinstance(candidate, dict):
        raise ShortFormProductionError("Episode candidate is not in the reviewed catalog")

    exact_translation = str(source.get("exact_translation", ""))
    ref = str(source.get("ref", ""))
    candidate_start_ref = str(candidate.get("start_ref", ""))
    candidate_end_ref = str(candidate.get("end_ref", ""))
    start_key = _ref_key(candidate_start_ref)
    end_key = _ref_key(candidate_end_ref)
    if start_key[0] != end_key[0] or start_key > end_key:
        raise ShortFormProductionError("Candidate range must be ordered within one surah")
    expected_ref = (
        candidate_start_ref
        if candidate_start_ref == candidate_end_ref
        else f"{start_key[0]}:{start_key[1]}-{end_key[1]}"
    )
    if ref != expected_ref:
        raise ShortFormProductionError("Episode reference differs from the catalog range")
    if candidate.get("exact_translation") != exact_translation:
        raise ShortFormProductionError("Episode wording differs from the canonical catalog")
    segment_qa_path = _validate_segment_contract(
        alignment_path,
        video_path,
        source,
        candidate_start_ref,
        candidate_end_ref,
    )

    alignment = _read_json(alignment_path)
    spans = alignment.get("spans")
    if not isinstance(spans, list):
        raise ShortFormProductionError("Alignment has no spans")
    candidate_refs = candidate.get("refs")
    if not isinstance(candidate_refs, list) or not candidate_refs:
        raise ShortFormProductionError("Candidate has no ordered verse references")
    selected_spans = [row for ref_value in candidate_refs for row in spans if row.get("ref") == ref_value]
    if len(selected_spans) != len(candidate_refs):
        raise ShortFormProductionError("Alignment is missing part of the episode range")
    combined_text = " ".join(str(row.get("text", "")) for row in selected_spans)
    if combined_text != exact_translation:
        raise ShortFormProductionError("Alignment wording differs from the episode")
    span = {
        "start": selected_spans[0]["start"],
        "end": selected_spans[-1]["end"],
        "text": combined_text,
        "refs": candidate_refs,
    }
    if abs(float(span["start"]) - float(source["clip_start_seconds"])) > 0.001:
        raise ShortFormProductionError("Episode clip start differs from forced alignment")
    if abs(float(span["end"]) - float(source["clip_end_seconds"])) > 0.001:
        raise ShortFormProductionError("Episode clip end differs from forced alignment")

    segments = spec.get("caption_segments")
    if not isinstance(segments, list) or not segments:
        raise ShortFormProductionError("Episode has no exact-text caption segments")
    caption_text = " ".join(str(row.get("text", "")).strip() for row in segments)
    if caption_text != exact_translation:
        raise ShortFormProductionError("Caption segments do not reconstruct the exact quote")
    previous_end = -1.0
    clip_duration = float(source["clip_end_seconds"]) - float(source["clip_start_seconds"])
    for row in segments:
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        if start < previous_end or end <= start or end > clip_duration + 0.001:
            raise ShortFormProductionError("Caption timings are invalid or overlapping")
        previous_end = end

    post_roll = float(source.get("post_roll_seconds", 0.0))
    if post_roll < 0.2:
        raise ShortFormProductionError("Source post-roll must protect the final spoken word")
    next_word_starts = [
        float(row["start"])
        for row in alignment.get("words", [])
        if float(row.get("start", -1.0)) > float(span["end"]) + 0.001
    ]
    if (
        next_word_starts
        and float(span["end"]) + post_roll >= min(next_word_starts) - 0.1
    ):
        raise ShortFormProductionError("Source post-roll crosses into the next spoken verse")
    if clip_duration + post_roll >= float(spec["render"]["duration_seconds"]):
        raise ShortFormProductionError("Episode has no room for a clean closing frame")

    return {
        "candidate": candidate,
        "span": span,
        "spans": selected_spans,
        "video_path": video_path,
        "segment_qa_path": segment_qa_path,
    }


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return "\n".join(lines)


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    max_size: int,
    min_size: int,
    max_width: int,
    max_height: int,
    spacing: int,
) -> tuple[ImageFont.FreeTypeFont, str]:
    for size in range(max_size, min_size - 1, -2):
        font = _font(font_path, size)
        wrapped = _wrap_text(draw, text, font, max_width)
        box = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing)
        if box[2] - box[0] <= max_width and box[3] - box[1] <= max_height:
            return font, wrapped
    raise ShortFormProductionError(f"Text does not fit the vertical safe area: {text}")


def _save_layer(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _render_layers(spec: dict[str, Any], layers_dir: Path) -> dict[str, Path]:
    render = spec["render"]
    width = int(render["width"])
    height = int(render["height"])
    regular = str(render["font_regular"])
    bold = str(render["font_bold"])
    white = (247, 246, 242, 255)
    muted = (212, 216, 220, 255)
    accent = (244, 77, 75, 255)
    content_center_x = 520
    branding = spec.get("branding", {})
    creative = spec["creative"]
    episode_number = str(branding.get("episode_number", "001"))
    reference_label = str(
        branding.get("reference_label", f"QURAN {spec['source']['ref']}")
    )
    surah_label = str(branding.get("surah_label", "AL-MA'IDAH  |  THE TABLE"))
    hook_label = str(creative.get("hook_label", "A HARDER TEST OF FAIRNESS"))
    outro_label = str(creative.get("outro_label", "READ IT AGAIN"))

    paths: dict[str, Path] = {}
    common = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(common)
    draw.rounded_rectangle((68, 138, 972, 248), radius=8, fill=(4, 10, 16, 150))
    draw.text((94, 168), "READ THAT AGAIN", font=_font(bold, 34), fill=white)
    draw.text(
        (943, 174), episode_number, font=_font(bold, 27), fill=accent, anchor="ra"
    )
    draw.rounded_rectangle((68, 1604, 972, 1738), radius=8, fill=(4, 10, 16, 178))
    draw.text((94, 1637), reference_label, font=_font(bold, 36), fill=white)
    draw.text(
        (94, 1684),
        surah_label,
        font=_font(regular, 23),
        fill=muted,
    )
    draw.rounded_rectangle((68, 1761, 972, 1773), radius=6, fill=(255, 255, 255, 75))
    draw.rounded_rectangle((68, 1761, 732, 1773), radius=6, fill=accent)
    paths["common"] = layers_dir / "common.png"
    _save_layer(paths["common"], common)

    hook = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hook)
    draw.rounded_rectangle((68, 318, 972, 762), radius=8, fill=(4, 10, 16, 156))
    label_font = _font(bold, 25)
    draw.text((94, 360), hook_label, font=label_font, fill=accent)
    hook_font, wrapped_hook = _fit_text(
        draw,
        creative["hook"],
        bold,
        78,
        56,
        824,
        310,
        18,
    )
    draw.multiline_text(
        (94, 422),
        wrapped_hook,
        font=hook_font,
        fill=white,
        spacing=18,
    )
    paths["hook"] = layers_dir / "hook.png"
    _save_layer(paths["hook"], hook)

    for index, segment in enumerate(spec["caption_segments"], start=1):
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.rounded_rectangle((68, 1052, 972, 1516), radius=8, fill=(4, 10, 16, 182))
        draw.text((94, 1092), "THE VERSE", font=_font(bold, 23), fill=accent)
        caption_font, wrapped = _fit_text(
            draw,
            str(segment["text"]),
            regular,
            56,
            42,
            796,
            324,
            14,
        )
        box = draw.multiline_textbbox((0, 0), wrapped, font=caption_font, spacing=14)
        text_height = box[3] - box[1]
        draw.multiline_text(
            (content_center_x, 1300 - text_height / 2),
            wrapped,
            font=caption_font,
            fill=white,
            spacing=14,
            anchor="ma",
            align="center",
        )
        path = layers_dir / f"caption-{index:02d}.png"
        paths[f"caption_{index}"] = path
        _save_layer(path, layer)

    outro = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(outro)
    draw.rounded_rectangle((68, 430, 972, 1010), radius=8, fill=(4, 10, 16, 178))
    draw.text((94, 478), outro_label, font=_font(bold, 25), fill=accent)
    outro_font, wrapped_outro = _fit_text(
        draw,
        creative["closing"],
        bold,
        70,
        48,
        824,
        360,
        16,
    )
    draw.multiline_text(
        (94, 555),
        wrapped_outro,
        font=outro_font,
        fill=white,
        spacing=16,
    )
    draw.text(
        (94, 918),
        creative["closing_question"],
        font=_font(regular, 28),
        fill=muted,
    )
    paths["outro"] = layers_dir / "outro.png"
    _save_layer(paths["outro"], outro)
    return paths


def _extract_audio(spec: dict[str, Any], video_path: Path, audio_path: Path) -> None:
    source = spec["source"]
    duration = (
        float(source["clip_end_seconds"])
        - float(source["clip_start_seconds"])
        + float(source.get("post_roll_seconds", 0.0))
    )
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-ss",
            f"{float(source['clip_start_seconds']):.3f}",
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s24le",
            str(audio_path),
        ]
    )


def _normalize_spoken_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("’", "'").replace("—", " "))


def _audit_audio_semantics(spec: dict[str, Any], audio_path: Path) -> dict[str, Any]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise ShortFormProductionError(
            "mlx-whisper is required for spoken-audio semantic QA"
        ) from exc

    model = "mlx-community/whisper-tiny.en-mlx"
    payload = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        verbose=False,
    )
    transcript = str(payload.get("text", "")).strip()
    expected = _normalize_spoken_words(spec["source"]["exact_translation"])
    observed = _normalize_spoken_words(transcript)
    similarity = SequenceMatcher(None, expected, observed, autojunk=False).ratio()
    terminal_word_count = min(3, len(expected))
    expected_terminal = expected[-terminal_word_count:]
    observed_terminal = observed[-terminal_word_count:]
    if not expected or observed != expected:
        raise ShortFormProductionError(
            "Spoken audio does not match the canonical verse through its final words: "
            f"similarity={similarity:.3f}, transcript={transcript!r}"
        )
    return {
        "passed": True,
        "policy": "exact-normalized-words-v2",
        "engine": f"mlx-whisper:{model}",
        "transcript": transcript,
        "word_sequence_similarity": round(similarity, 6),
        "expected_terminal_words": expected_terminal,
        "observed_terminal_words": observed_terminal,
    }


def _render_video(
    spec: dict[str, Any],
    project_root: Path,
    layers: dict[str, Path],
    audio_path: Path,
    output_path: Path,
) -> None:
    render = spec["render"]
    duration = float(render["duration_seconds"])
    fps = int(render["fps"])
    background = project_root / str(spec["visual"]["background_path"])
    layer_order = [
        layers["common"],
        layers["hook"],
        *(layers[f"caption_{i}"] for i in range(1, len(spec["caption_segments"]) + 1)),
        layers["outro"],
    ]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(background),
    ]
    for layer in layer_order:
        command.extend(["-loop", "1", "-framerate", str(fps), "-i", str(layer)])
    audio_index = 1 + len(layer_order)
    command.extend(["-i", str(audio_path)])

    quote_end = float(spec["source"]["clip_end_seconds"]) - float(
        spec["source"]["clip_start_seconds"]
    )
    audio_end = quote_end + float(spec["source"].get("post_roll_seconds", 0.0))
    filter_parts = [
        (
            f"[0:v]scale={int(render['width'])}:{int(render['height'])}:"
            "force_original_aspect_ratio=increase,"
            f"crop={int(render['width'])}:{int(render['height'])},"
            "zoompan=z='min(zoom+0.00012,1.045)':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:"
            f"s={int(render['width'])}x{int(render['height'])}:fps={fps},"
            "eq=brightness=-0.055:saturation=0.88,format=yuv420p[base]"
        ),
        "[1:v]format=rgba[common]",
        "[base][common]overlay=0:0:format=auto[v1]",
    ]
    current = "v1"
    input_index = 2
    hook_end = float(spec["creative"].get("hook_end_seconds", 4.8))
    windows = [(0.0, hook_end)]
    windows.extend(
        (float(row["start_seconds"]), float(row["end_seconds"]))
        for row in spec["caption_segments"]
    )
    windows.append((quote_end, duration))
    for layer_number, (start, end) in enumerate(windows, start=2):
        label = f"layer{layer_number}"
        next_video = f"v{layer_number}"
        filter_parts.append(f"[{input_index}:v]format=rgba[{label}]")
        filter_parts.append(
            f"[{current}][{label}]overlay=0:0:format=auto:"
            f"enable='between(t,{start:.3f},{end:.3f})'[{next_video}]"
        )
        current = next_video
        input_index += 1
    filter_parts.append(
        f"[{audio_index}:a]apad=pad_dur={max(0.0, duration - audio_end):.3f},"
        f"afade=t=out:st={max(0.0, audio_end - 0.12):.3f}:d=0.12[aout]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            f"[{current}]",
            "-map",
            "[aout]",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    _run(command)


def _write_platform_copy(spec: dict[str, Any], output_dir: Path) -> None:
    for platform in ("instagram", "tiktok"):
        copy = str(spec["platform_copy"][platform]).strip() + "\n"
        (output_dir / f"caption-{platform}.txt").write_text(copy, encoding="utf-8")


def _audit_video(
    spec: dict[str, Any],
    output_dir: Path,
    master: Path,
    exports: dict[str, Path],
    audio_semantic_qa: dict[str, Any],
) -> dict[str, Any]:
    probe = _ffprobe(master)
    streams = probe.get("streams", [])
    video = next((row for row in streams if row.get("codec_type") == "video"), None)
    audio = next((row for row in streams if row.get("codec_type") == "audio"), None)
    if not isinstance(video, dict) or not isinstance(audio, dict):
        raise ShortFormProductionError("Rendered master lacks video or audio")
    render = spec["render"]
    actual_duration = float(probe["format"]["duration"])
    if int(video.get("width", 0)) != int(render["width"]) or int(video.get("height", 0)) != int(
        render["height"]
    ):
        raise ShortFormProductionError("Rendered dimensions differ from the 9:16 contract")
    if abs(actual_duration - float(render["duration_seconds"])) > 0.12:
        raise ShortFormProductionError("Rendered duration differs from the episode contract")
    if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
        raise ShortFormProductionError("Rendered codecs differ from H.264/AAC")

    _run(["ffmpeg", "-v", "error", "-i", str(master), "-f", "null", "-"])
    encoded_audio_qa = _audit_audio_semantics(spec, master)
    master_hash = _sha256(master)
    for export in exports.values():
        if _sha256(export) != master_hash:
            raise ShortFormProductionError("Platform export differs from the audited master")
    frames_dir = output_dir / "qa-frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    sample_seconds = tuple(
        min(max(0.05, actual_duration * fraction), actual_duration - 0.05)
        for fraction in (0.06, 0.28, 0.5, 0.72, 0.92)
    )
    for index, second in enumerate(sample_seconds, start=1):
        frame_path = frames_dir / f"frame-{index:02d}.png"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(second),
                "-i",
                str(master),
                "-frames:v",
                "1",
                str(frame_path),
            ]
        )
        frame_paths.append(frame_path)

    thumbs = []
    for path in frame_paths:
        with Image.open(path) as frame:
            thumb = frame.convert("RGB").resize((270, 480), Image.Resampling.LANCZOS)
            thumbs.append(thumb.copy())
    sheet = Image.new("RGB", (270 * len(thumbs), 480), (10, 10, 10))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, (index * 270, 0))
    contact_sheet = output_dir / "CONTACT_SHEET.jpg"
    sheet.save(contact_sheet, quality=92)

    qa = {
        "version": "quran-short-form-qa-v1",
        "episode_id": spec["episode_id"],
        "status": "technical_qa_passed_visual_review_pending",
        "video": {
            "width": video["width"],
            "height": video["height"],
            "duration_seconds": actual_duration,
            "video_codec": video["codec_name"],
            "audio_codec": audio["codec_name"],
            "decode_passed": True,
        },
        "spoken_audio": audio_semantic_qa,
        "encoded_spoken_audio": encoded_audio_qa,
        "artifacts": {
            "master": {"path": str(master), "sha256": _sha256(master)},
            "instagram": {
                "path": str(exports["instagram"]),
                "sha256": _sha256(exports["instagram"]),
            },
            "tiktok": {"path": str(exports["tiktok"]), "sha256": _sha256(exports["tiktok"])},
            "contact_sheet": str(contact_sheet),
        },
    }
    (output_dir / "QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return qa


def _render_short_form_episode(
    project_root: Path, spec_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Render one episode, platform exports, captions, QA frames, and receipts."""

    spec = _read_json(spec_path)
    validated = validate_episode_spec(project_root, spec)
    layers_dir = output_dir / "layers"
    output_dir.mkdir(parents=True, exist_ok=True)
    background = project_root / str(spec["visual"]["background_path"])
    if not background.is_file():
        raise ShortFormProductionError(f"Missing visual asset: {background}")

    layers = _render_layers(spec, layers_dir)
    audio_path = output_dir / "source-quote.wav"
    _extract_audio(spec, validated["video_path"], audio_path)
    audio_semantic_qa = _audit_audio_semantics(spec, audio_path)
    master = output_dir / f"{spec['episode_id']}-master.mp4"
    _render_video(spec, project_root, layers, audio_path, master)

    exports = {
        "instagram": output_dir / f"{spec['episode_id']}-instagram.mp4",
        "tiktok": output_dir / f"{spec['episode_id']}-tiktok.mp4",
    }
    for export in exports.values():
        shutil.copy2(master, export)
    _write_platform_copy(spec, output_dir)
    qa = _audit_video(spec, output_dir, master, exports, audio_semantic_qa)

    source_receipt = {
        "version": "quran-short-form-source-receipt-v1",
        "episode_id": spec["episode_id"],
        "candidate_id": spec["candidate_id"],
        "ref": spec["source"]["ref"],
        "exact_translation": spec["source"]["exact_translation"],
        "catalog_sha256": _sha256(project_root / spec["source"]["catalog_path"]),
        "alignment_sha256": _sha256(project_root / spec["source"]["alignment_path"]),
        "segment_index": spec["source"]["segment_index"],
        "segment_qa_path": str(validated["segment_qa_path"]),
        "segment_qa_sha256": _sha256(validated["segment_qa_path"]),
        "source_video_sha256": _sha256(project_root / spec["source"]["video_path"]),
        "spoken_audio_qa": audio_semantic_qa,
        "background_sha256": _sha256(background),
        "visual_disclosure": spec["visual"]["disclosure"],
        "replaces_episode_id": spec.get("replaces_episode_id"),
    }
    (output_dir / "SOURCE_RECEIPT.json").write_text(
        json.dumps(source_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    publication = {
        "version": "quran-short-form-publication-state-v1",
        "episode_id": spec["episode_id"],
        "status": "visual_review_pending",
        "accounts": {
            "instagram": {"username": "leftonread_1", "state": "not_uploaded"},
            "tiktok": {"username": "read.that.again0", "state": "not_uploaded"},
        },
        "qa_path": str(output_dir / "QA.json"),
        "source_receipt_path": str(output_dir / "SOURCE_RECEIPT.json"),
    }
    (output_dir / "PUBLICATION_STATE.json").write_text(
        json.dumps(publication, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"output_dir": str(output_dir), "qa": qa, "publication": publication}

def _rebase_paths(value: Any, old: Path, new: Path) -> Any:
    if isinstance(value, str) and value == str(old):
        return str(new)
    if isinstance(value, dict):
        return {key: _rebase_paths(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase_paths(item, old, new) for item in value]
    if isinstance(value, str) and value.startswith(str(old) + "/"):
        return str(new) + value[len(str(old)):]
    return value


def render_short_form_episode(project_root: Path, spec_path: Path) -> dict[str, Any]:
    """Publish an immutable local render; never reset an existing upload ledger."""
    project_root = project_root.resolve()
    spec_path = spec_path.resolve()
    spec = _read_json(spec_path)
    episode_id = str(spec.get("episode_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", episode_id):
        raise ShortFormProductionError("Invalid episode ID")
    parent = project_root / "output" / "short-form" / "episodes"
    destination = parent / episode_id
    with exclusive_lock(parent / ".locks" / (episode_id + ".lock")):
        validated = validate_episode_spec(project_root, spec)
        paths = [project_root / spec["source"][key]
                 for key in ("catalog_path", "alignment_path", "video_path")]
        paths += [validated["segment_qa_path"],
                  project_root / spec["visual"]["background_path"], Path(__file__)]
        fingerprint = hashlib.sha256(json.dumps(
            {"spec": spec, "inputs": {str(path): _sha256(path) for path in paths}},
            sort_keys=True, ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        if destination.exists():
            preserve_publication_receipt(destination)
            marker_path = destination / "RENDER_COMPLETE.json"
            if not marker_path.is_file():
                raise ShortFormProductionError(
                    "Existing episode has no immutable render receipt; preserve it and "
                    "use a new episode ID with replaces_episode_id for a replacement"
                )
            marker = _read_json(marker_path)
            if marker.get("input_fingerprint") != fingerprint:
                raise ShortFormProductionError(
                    "Episode inputs changed; use a new episode ID with replaces_episode_id"
                )
            artifacts = marker.get("artifacts")
            if not isinstance(artifacts, dict) or not artifacts:
                raise ShortFormProductionError("Missing render artifact manifest")
            for relative, digest in artifacts.items():
                artifact = destination / relative
                if not artifact.is_file() or _sha256(artifact) != digest:
                    raise ShortFormProductionError(f"Immutable render artifact changed: {artifact}")
            return {"output_dir": str(destination),
                    "qa": _read_json(destination / "QA.json"),
                    "publication": _read_json(destination / "PUBLICATION_STATE.json"),
                    "reused": True}
        replaces = spec.get("replaces_episode_id")
        if replaces:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(replaces)):
                raise ShortFormProductionError("Invalid replacement episode ID")
            if not (parent / replaces / "PUBLICATION_STATE.json").is_file():
                raise ShortFormProductionError("Replacement must reference an existing episode")
        staging = parent / ".render-attempts" / f"{episode_id}-{uuid.uuid4().hex}"
        result = _render_short_form_episode(project_root, spec_path, staging)
        for path in staging.rglob("*.json"):
            atomic_json(path, _rebase_paths(_read_json(path), staging, destination))
        atomic_json(staging / "RENDER_COMPLETE.json", {
            "version": "quran-short-form-immutable-render-v1",
            "input_fingerprint": fingerprint,
            "replaces_episode_id": replaces,
            "artifacts": {
                str(path.relative_to(staging)): _sha256(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file() and path.name != "PUBLICATION_STATE.json"
            },
        })
        os.rename(staging, destination)
        preserve_publication_receipt(destination)
        return _rebase_paths(result, staging, destination)
