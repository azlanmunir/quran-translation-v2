"""Normalize independent aligners into one hashed ayah-timeline contract."""

from __future__ import annotations

import re
import shutil
import statistics
import subprocess
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Any

import requests

from .config import file_sha256, text_sha256
from .production_packets import atomic_json
from .video_pipeline import VideoPipelineError


WORD_RE = re.compile(r"[\w]+(?:[’'-][\w]+)*", re.UNICODE)


class AlignmentError(VideoPipelineError):
    """Raised when alignment output is missing, malformed, or untraceable."""


def _normalized_word(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def transcript_words(text: str) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for match in WORD_RE.finditer(text):
        normalized = _normalized_word(match.group(0))
        if normalized:
            words.append(
                {
                    "text": match.group(0),
                    "normalized": normalized,
                    "start_char": match.start(),
                    "end_char": match.end(),
                }
            )
    return words


def _flatten_whisper_words(payload: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise AlignmentError("Whisper output has no segments")
    for segment in segments:
        if not isinstance(segment, dict):
            raise AlignmentError("Whisper output contains a malformed segment")
        for raw in segment.get("words", []):
            if not isinstance(raw, dict):
                raise AlignmentError("Whisper output contains a malformed word")
            text = raw.get("word")
            start = raw.get("start")
            end = raw.get("end")
            if not isinstance(text, str) or not isinstance(start, (int, float)) or not isinstance(
                end, (int, float)
            ):
                raise AlignmentError("Whisper word is missing text or timestamps")
            normalized = _normalized_word(text)
            if normalized and float(end) >= float(start):
                words.append(
                    {
                        "text": text.strip(),
                        "normalized": normalized,
                        "start": float(start),
                        "end": float(end),
                        "probability": raw.get("probability"),
                    }
                )
    if not words:
        raise AlignmentError("Whisper output has no timestamped words")
    return words


def _match_expected_words(
    expected: list[dict[str, Any]],
    observed: list[dict[str, Any]],
) -> tuple[dict[int, tuple[float, float]], int]:
    expected_values = [word["normalized"] for word in expected]
    observed_values = [word["normalized"] for word in observed]
    matcher = SequenceMatcher(None, expected_values, observed_values, autojunk=False)
    mapping: dict[int, tuple[float, float]] = {}
    exact_matches = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for expected_index, observed_index in zip(range(i1, i2), range(j1, j2)):
                mapping[expected_index] = (
                    float(observed[observed_index]["start"]),
                    float(observed[observed_index]["end"]),
                )
                exact_matches += 1
        elif tag == "replace" and i2 - i1 == j2 - j1:
            for expected_index, observed_index in zip(range(i1, i2), range(j1, j2)):
                ratio = SequenceMatcher(
                    None,
                    expected_values[expected_index],
                    observed_values[observed_index],
                    autojunk=False,
                ).ratio()
                if ratio >= 0.60:
                    mapping[expected_index] = (
                        float(observed[observed_index]["start"]),
                        float(observed[observed_index]["end"]),
                    )

    if not mapping:
        raise AlignmentError("Whisper transcript has no usable overlap with narration script")
    return mapping, exact_matches


def _interpolated_word_times(
    expected: list[dict[str, Any]],
    mapping: dict[int, tuple[float, float]],
) -> list[tuple[float, float]]:
    result: list[tuple[float, float] | None] = [mapping.get(i) for i in range(len(expected))]
    known = sorted(mapping)
    for index, timing in enumerate(result):
        if timing is not None:
            continue
        previous = max((item for item in known if item < index), default=None)
        following = min((item for item in known if item > index), default=None)
        if previous is None and following is None:
            raise AlignmentError("Cannot interpolate Whisper timestamps")
        if previous is None:
            end = mapping[following][0]
            duration = max(0.08, (end / max(1, following)) * 0.8)
            start = max(0.0, end - duration * (following - index))
            result[index] = (start, min(end, start + duration))
        elif following is None:
            start = mapping[previous][1]
            duration = max(0.08, (mapping[previous][1] - mapping[previous][0]) * 0.8)
            offset = index - previous - 1
            result[index] = (start + duration * offset, start + duration * (offset + 1))
        else:
            window_start = mapping[previous][1]
            window_end = mapping[following][0]
            count = following - previous - 1
            slot = max(0.0, window_end - window_start) / max(1, count)
            offset = index - previous - 1
            result[index] = (
                window_start + slot * offset,
                window_start + slot * (offset + 1),
            )
    return [timing for timing in result if timing is not None]


def _span_word_range(
    words: list[dict[str, Any]], start_char: int, end_char: int
) -> tuple[int, int]:
    indices = [
        index
        for index, word in enumerate(words)
        if int(word["start_char"]) < end_char and int(word["end_char"]) > start_char
    ]
    if not indices:
        raise AlignmentError(f"Transcript span {start_char}:{end_char} has no words")
    return indices[0], indices[-1]


def normalize_whisper_alignment(
    *,
    raw_payload: dict[str, Any],
    transcript: str,
    spans: list[dict[str, Any]],
    audio_path: Path,
    raw_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    expected = transcript_words(transcript)
    observed = _flatten_whisper_words(raw_payload)
    mapping, exact_matches = _match_expected_words(expected, observed)
    word_times = _interpolated_word_times(expected, mapping)
    aligned_words = [
        {
            "text": word["text"],
            "start_char": int(word["start_char"]),
            "end_char": int(word["end_char"]),
            "start": round(float(word_times[index][0]), 3),
            "end": round(float(word_times[index][1]), 3),
            "timing_source": "exact_word" if index in mapping else "interpolated_word",
        }
        for index, word in enumerate(expected)
    ]
    aligned_spans: list[dict[str, Any]] = []
    for span in spans:
        first, last = _span_word_range(
            expected, int(span["start_char"]), int(span["end_char"])
        )
        aligned_spans.append(
            {
                key: span[key]
                for key in (
                    "kind",
                    "ref",
                    "surah",
                    "ayah",
                    "start_char",
                    "end_char",
                    "text",
                )
                if key in span
            }
            | {
                "start": round(float(word_times[first][0]), 3),
                "end": round(float(word_times[last][1]), 3),
                "timing_source": "exact_words"
                if all(index in mapping for index in range(first, last + 1))
                else "interpolated_words",
            }
        )
    payload = {
        "version": "quran-video-alignment-v1",
        "engine": "openai-whisper-word-timestamps",
        "audio_sha256": file_sha256(audio_path),
        "transcript_sha256": text_sha256(transcript),
        "raw_alignment": {"path": raw_path.name, "sha256": file_sha256(raw_path)},
        "metrics": {
            "expected_words": len(expected),
            "observed_words": len(observed),
            "exact_word_coverage": round(exact_matches / len(expected), 6),
            "mapped_word_coverage": round(len(mapping) / len(expected), 6),
        },
        "words": aligned_words,
        "spans": aligned_spans,
    }
    atomic_json(output_path, payload)
    return payload


def normalize_forced_alignment(
    *,
    raw_payload: dict[str, Any],
    transcript: str,
    spans: list[dict[str, Any]],
    audio_path: Path,
    raw_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    characters = raw_payload.get("characters")
    if not isinstance(characters, list) or not characters:
        raise AlignmentError("Forced-alignment output has no characters")
    response_text = "".join(str(item.get("text", "")) for item in characters if isinstance(item, dict))
    if response_text != transcript:
        raise AlignmentError("Forced-alignment characters do not reproduce the narration script")

    aligned_spans: list[dict[str, Any]] = []
    for span in spans:
        start_char = int(span["start_char"])
        end_char = int(span["end_char"])
        relevant = [
            item
            for item in characters[start_char:end_char]
            if isinstance(item, dict)
            and isinstance(item.get("start"), (int, float))
            and isinstance(item.get("end"), (int, float))
        ]
        if not relevant:
            raise AlignmentError(f"Forced alignment has no timing for span {start_char}:{end_char}")
        aligned_spans.append(
            {
                key: span[key]
                for key in (
                    "kind",
                    "ref",
                    "surah",
                    "ayah",
                    "start_char",
                    "end_char",
                    "text",
                )
                if key in span
            }
            | {
                "start": round(float(relevant[0]["start"]), 3),
                "end": round(float(relevant[-1]["end"]), 3),
                "timing_source": "forced_characters",
            }
        )
    expected_words = transcript_words(transcript)
    aligned_words: list[dict[str, Any]] = []
    for word in expected_words:
        relevant = [
            item
            for item in characters[int(word["start_char"]) : int(word["end_char"])]
            if isinstance(item, dict)
            and isinstance(item.get("start"), (int, float))
            and isinstance(item.get("end"), (int, float))
        ]
        if not relevant:
            raise AlignmentError(
                f"Forced alignment has no timing for word {word['start_char']}:{word['end_char']}"
            )
        aligned_words.append(
            {
                "text": word["text"],
                "start_char": int(word["start_char"]),
                "end_char": int(word["end_char"]),
                "start": round(float(relevant[0]["start"]), 3),
                "end": round(float(relevant[-1]["end"]), 3),
                "timing_source": "forced_characters",
            }
        )
    payload = {
        "version": "quran-video-alignment-v1",
        "engine": "elevenlabs-forced-alignment",
        "audio_sha256": file_sha256(audio_path),
        "transcript_sha256": text_sha256(transcript),
        "raw_alignment": {"path": raw_path.name, "sha256": file_sha256(raw_path)},
        "metrics": {
            "characters": len(characters),
            "words": len(raw_payload.get("words", [])),
            "loss": raw_payload.get("loss"),
        },
        "words": aligned_words,
        "spans": aligned_spans,
    }
    atomic_json(output_path, payload)
    return payload


def request_forced_alignment(
    *,
    audio_path: Path,
    transcript: str,
    api_key: str,
    output_path: Path,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    if not api_key.strip():
        raise AlignmentError("ELEVENLABS_API_KEY is required")
    try:
        with audio_path.open("rb") as handle:
            response = requests.post(
                "https://api.elevenlabs.io/v1/forced-alignment",
                headers={"xi-api-key": api_key},
                files={"file": (audio_path.name, handle, "audio/mpeg")},
                data={"text": transcript},
                timeout=timeout_seconds,
            )
    except requests.RequestException as exc:
        raise AlignmentError(f"ElevenLabs forced alignment network failure: {exc}") from exc
    if response.status_code != 200:
        body = response.text[:1000]
        raise AlignmentError(f"ElevenLabs forced alignment failed ({response.status_code}): {body}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise AlignmentError("ElevenLabs forced alignment returned non-JSON output") from exc
    if not isinstance(payload, dict):
        raise AlignmentError("ElevenLabs forced alignment returned a malformed object")
    atomic_json(output_path, payload)
    return payload


def run_whisper(
    *,
    audio_path: Path,
    transcript: str,
    output_dir: Path,
    whisper_command: str = "whisper",
    model: str = "turbo",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{audio_path.stem}.json"
    subprocess.run(
        [
            whisper_command,
            str(audio_path),
            "--model",
            model,
            "--device",
            "cpu",
            "--language",
            "en",
            "--word_timestamps",
            "True",
            "--output_format",
            "json",
            "--output_dir",
            str(output_dir),
            "--verbose",
            "False",
            "--fp16",
            "False",
            "--condition_on_previous_text",
            "True",
            "--initial_prompt",
            transcript,
        ],
        check=True,
    )
    if not output_path.exists():
        raise AlignmentError(f"Whisper did not write expected output: {output_path}")
    return output_path


def compare_alignments(
    *, first: dict[str, Any], second: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    if first.get("audio_sha256") != second.get("audio_sha256"):
        raise AlignmentError("Cannot compare alignments for different audio")
    if first.get("transcript_sha256") != second.get("transcript_sha256"):
        raise AlignmentError("Cannot compare alignments for different transcripts")
    first_by_key = {
        (span.get("kind"), span.get("ref"), span.get("surah")): span
        for span in first.get("spans", [])
    }
    second_by_key = {
        (span.get("kind"), span.get("ref"), span.get("surah")): span
        for span in second.get("spans", [])
    }
    if first_by_key.keys() != second_by_key.keys():
        raise AlignmentError("Alignment span coverage differs")
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for key, first_span in first_by_key.items():
        second_span = second_by_key[key]
        start_delta = abs(float(first_span["start"]) - float(second_span["start"]))
        end_delta = abs(float(first_span["end"]) - float(second_span["end"]))
        deltas.extend((start_delta, end_delta))
        rows.append(
            {
                "kind": key[0],
                "ref": key[1],
                "surah": key[2],
                "first_start": first_span["start"],
                "second_start": second_span["start"],
                "start_delta": round(start_delta, 3),
                "first_end": first_span["end"],
                "second_end": second_span["end"],
                "end_delta": round(end_delta, 3),
            }
        )
    sorted_deltas = sorted(deltas)
    p95_index = min(len(sorted_deltas) - 1, round(0.95 * (len(sorted_deltas) - 1)))
    payload = {
        "version": "quran-video-alignment-comparison-v1",
        "first_engine": first.get("engine"),
        "second_engine": second.get("engine"),
        "audio_sha256": first.get("audio_sha256"),
        "transcript_sha256": first.get("transcript_sha256"),
        "metrics": {
            "spans": len(rows),
            "median_boundary_delta_seconds": round(statistics.median(deltas), 3),
            "p95_boundary_delta_seconds": round(sorted_deltas[p95_index], 3),
            "max_boundary_delta_seconds": round(max(deltas), 3),
            "boundaries_over_0_5_seconds": sum(delta > 0.5 for delta in deltas),
        },
        "largest_disagreements": sorted(
            rows, key=lambda row: max(row["start_delta"], row["end_delta"]), reverse=True
        )[:20],
        "spans": rows,
    }
    atomic_json(output_path, payload)
    return payload


def write_alignment_review(
    *,
    first: dict[str, Any],
    second: dict[str, Any],
    comparison: dict[str, Any],
    audio_path: Path,
    output_dir: Path,
    limit: int = 20,
) -> dict[str, Any]:
    """Package the largest boundary disagreements into a portable listening console."""

    audio_sha = file_sha256(audio_path)
    transcript_sha = first.get("transcript_sha256")
    if first.get("audio_sha256") != audio_sha or second.get("audio_sha256") != audio_sha:
        raise AlignmentError("Alignment review audio does not match both alignments")
    if second.get("transcript_sha256") != transcript_sha:
        raise AlignmentError("Alignment review transcripts differ")
    if comparison.get("audio_sha256") != audio_sha or comparison.get(
        "transcript_sha256"
    ) != transcript_sha:
        raise AlignmentError("Alignment comparison provenance differs from its inputs")

    output_dir.mkdir(parents=True, exist_ok=True)
    portable_audio = output_dir / audio_path.name
    if not portable_audio.exists() or file_sha256(portable_audio) != audio_sha:
        shutil.copy2(audio_path, portable_audio)

    first_spans = {
        (span.get("kind"), span.get("ref"), span.get("surah")): span
        for span in first.get("spans", [])
    }
    rows: list[dict[str, Any]] = []
    for raw in comparison.get("largest_disagreements", [])[:limit]:
        key = (raw.get("kind"), raw.get("ref"), raw.get("surah"))
        span = first_spans.get(key, {})
        rows.append(
            {
                **raw,
                "text": span.get("text", ""),
                "review": {
                    "start_boundary": None,
                    "end_boundary": None,
                    "notes": "",
                },
            }
        )

    row_html: list[str] = []
    for row in rows:
        ref = escape(str(row.get("ref") or f"Surah {row.get('surah')} announcement"))
        text = escape(str(row.get("text", "")))
        first_start = float(row["first_start"])
        second_start = float(row["second_start"])
        first_end = float(row["first_end"])
        second_end = float(row["second_end"])
        row_html.append(
            "<article><h2>"
            + ref
            + "</h2><p>"
            + text
            + "</p><div class='grid'>"
            + f"<button onclick='playAt({first_start})'>A start · {first_start:.3f}s</button>"
            + f"<button onclick='playAt({second_start})'>B start · {second_start:.3f}s</button>"
            + f"<button onclick='playAt({first_end})'>A end · {first_end:.3f}s</button>"
            + f"<button onclick='playAt({second_end})'>B end · {second_end:.3f}s</button>"
            + "</div><small>Start disagreement "
            + f"{float(row['start_delta']):.3f}s · end disagreement {float(row['end_delta']):.3f}s"
            + "</small></article>"
        )

    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quran alignment boundary review</title><style>
body{margin:0;background:#0f1113;color:#f0ece2;font:16px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:880px;margin:auto;padding:28px 20px 80px}h1{font-size:30px;margin:0 0 10px}p{line-height:1.55}
.sticky{position:sticky;top:0;background:#0f1113;padding:12px 0 18px;border-bottom:1px solid #303238;z-index:2}
audio{width:100%}article{padding:24px 0;border-bottom:1px solid #303238}h2{color:#d3b25b;font-size:18px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:14px 0}button{padding:12px;border:1px solid #555861;
background:#202327;color:#f0ece2;border-radius:6px;text-align:left}small{color:#8d9299}@media(max-width:520px){.grid{grid-template-columns:1fr}}
</style></head><body><main><div class="sticky"><h1>Boundary review</h1>
<p>Listen around each candidate boundary. A and B are the two engines; choose the timestamp that begins or ends on the spoken text, not the one that merely looks cleaner numerically.</p>
<audio id="audio" controls preload="metadata" src="__AUDIO__"></audio></div>__ROWS__
</main><script>const audio=document.getElementById('audio');let timer;function playAt(t){clearTimeout(timer);audio.currentTime=Math.max(0,t-1.25);audio.play();timer=setTimeout(()=>audio.pause(),3500)}</script></body></html>
"""
    html = html.replace("__AUDIO__", escape(portable_audio.name)).replace(
        "__ROWS__", "".join(row_html)
    )
    html_path = output_dir / "review.html"
    html_path.write_text(html, encoding="utf-8")
    payload = {
        "version": "quran-video-alignment-review-v1",
        "audio": portable_audio.name,
        "audio_sha256": audio_sha,
        "transcript_sha256": transcript_sha,
        "aligner_a": first.get("engine"),
        "aligner_b": second.get("engine"),
        "comparison_metrics": comparison.get("metrics"),
        "review_row_count": len(rows),
        "rows": rows,
        "html": html_path.name,
        "html_sha256": file_sha256(html_path),
        "human_review_required": True,
    }
    atomic_json(output_dir / "REVIEW.json", payload)
    return payload
