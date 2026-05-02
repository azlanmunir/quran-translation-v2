"""Audio QA and release packaging for completed ElevenLabs runs."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_pipeline import audio_run, audio_status, slugify, write_audio_manifest
from .config import OUTPUT_DIR, file_sha256
from .metadata import SURAHS, SURAH_BY_NUMBER


DEFAULT_RELEASE_ROOT = OUTPUT_DIR / "release" / "quran-translation-v2"
VOICE_LABEL = "Nathan - natural narrator"
RELEASE_YEAR = "2026"
SURAH_ALBUM = "Quran Translation - English by Surah"
PART_ALBUM = "Quran Translation - English in 30 Parts"
FULL_BOOK_ALBUM = "Quran Translation - English Complete Book"
MIN_REASONABLE_BITRATE = 96_000


@dataclass(frozen=True)
class ReleaseOutput:
    output_type: str
    output_id: str
    label: str
    start_ref: str
    end_ref: str
    source_path: Path
    release_path: Path
    expected_duration_seconds: float
    expected_bytes: int


def create_audio_release(
    conn: sqlite3.Connection,
    *,
    audio_run_id: str,
    release_root: Path = DEFAULT_RELEASE_ROOT,
    force: bool = False,
    decode_check: bool = True,
) -> dict[str, Any]:
    """Create clean release copies, checksums, and QA reports for a generated audio run."""
    manifest_path = write_audio_manifest(conn, audio_run_id)
    release_root.mkdir(parents=True, exist_ok=True)

    qa_before = qa_source_audio(conn, audio_run_id)
    if qa_before["issue_count"]:
        write_json(release_root / "qa-source-report.json", qa_before)
        raise SystemExit(f"Source audio QA failed with {qa_before['issue_count']} issue(s).")

    release_outputs = build_release_outputs(conn, audio_run_id, release_root)
    copy_publication_artifacts(release_root)
    for output in release_outputs:
        if output.release_path.exists() and not force:
            continue
        remux_with_metadata(conn, audio_run_id, output)
    full_book_output = build_full_book_output(conn, audio_run_id, release_root)
    if not full_book_output.release_path.exists() or force:
        concat_full_book(conn, audio_run_id, full_book_output)
    release_outputs.append(full_book_output)

    release_manifest = build_release_manifest(conn, audio_run_id, release_root, release_outputs)
    write_json(release_root / "manifest" / "release-manifest.json", release_manifest)

    qa_release = qa_release_audio(release_outputs, decode_check=decode_check)
    write_json(release_root / "manifest" / "qa-report.json", qa_release)
    write_qa_summary(release_root / "manifest" / "qa-summary.md", release_manifest, qa_release)
    write_release_readme(release_root / "README.md", release_manifest, qa_release)
    write_checksums(release_root)

    return {
        "audio_run_id": audio_run_id,
        "release_root": str(release_root),
        "source_manifest": str(manifest_path),
        "release_manifest": str(release_root / "manifest" / "release-manifest.json"),
        "qa_report": str(release_root / "manifest" / "qa-report.json"),
        "checksums": str(release_root / "SHA256SUMS.txt"),
        "source_issues": qa_before["issue_count"],
        "release_issues": qa_release["issue_count"],
        "audio": {
            "by_surah": str(release_root / "audio" / "by-surah"),
            "parts_30": str(release_root / "audio" / "30-parts"),
            "full_book": str(release_root / "audio" / "full-book"),
            "surah_files": sum(1 for item in release_outputs if item.output_type == "surah"),
            "part_files": sum(1 for item in release_outputs if item.output_type == "part"),
            "full_book_files": sum(1 for item in release_outputs if item.output_type == "full"),
        },
        "pdf": {
            "print_book": str(release_root / "pdf" / "print-book"),
            "reader_book": str(release_root / "pdf" / "reader-book"),
        },
        "decode_check": decode_check,
    }


def qa_source_audio(conn: sqlite3.Connection, audio_run_id: str) -> dict[str, Any]:
    status = audio_status(conn, audio_run_id)
    issues: list[dict[str, Any]] = []
    if status["chunk_status"] != {"complete": 383}:
        issues.append(
            {
                "scope": "chunks",
                "message": f"Expected exactly 383 complete chunks; saw {status['chunk_status']}.",
            }
        )
    if status["outputs"].get("surah") != 114:
        issues.append(
            {
                "scope": "outputs",
                "message": f"Expected 114 surah outputs; saw {status['outputs'].get('surah', 0)}.",
            }
        )
    if status["outputs"].get("part") != 30:
        issues.append(
            {
                "scope": "outputs",
                "message": f"Expected 30 part outputs; saw {status['outputs'].get('part', 0)}.",
            }
        )

    chunk_rows = list(
        conn.execute(
            """
            SELECT chunk_id, output_path, duration_seconds, bytes
            FROM audio_chunks
            WHERE audio_run_id = ?
            ORDER BY chunk_index
            """,
            (audio_run_id,),
        )
    )
    for row in chunk_rows:
        check_audio_file(
            path=Path(row["output_path"]),
            expected_duration_seconds=float(row["duration_seconds"] or 0),
            expected_bytes=int(row["bytes"] or 0),
            scope=f"chunk:{row['chunk_id']}",
            issues=issues,
            ffprobe_check=True,
        )

    output_rows = list(
        conn.execute(
            """
            SELECT output_type, output_id, output_path, duration_seconds, bytes
            FROM audio_outputs
            WHERE audio_run_id = ?
            ORDER BY output_type, output_id
            """,
            (audio_run_id,),
        )
    )
    for row in output_rows:
        check_audio_file(
            path=Path(row["output_path"]),
            expected_duration_seconds=float(row["duration_seconds"] or 0),
            expected_bytes=int(row["bytes"] or 0),
            scope=f"{row['output_type']}:{row['output_id']}",
            issues=issues,
            ffprobe_check=True,
        )

    return {
        "audio_run_id": audio_run_id,
        "status": status,
        "checked": {
            "chunks": len(chunk_rows),
            "outputs": len(output_rows),
        },
        "issue_count": len(issues),
        "issues": issues,
    }


def build_release_outputs(
    conn: sqlite3.Connection,
    audio_run_id: str,
    release_root: Path,
) -> list[ReleaseOutput]:
    rows = list(
        conn.execute(
            """
            SELECT output_type, output_id, label, start_ref, end_ref, output_path,
                   duration_seconds, bytes
            FROM audio_outputs
            WHERE audio_run_id = ?
            ORDER BY output_type, output_id
            """,
            (audio_run_id,),
        )
    )
    outputs: list[ReleaseOutput] = []
    for row in rows:
        output_type = str(row["output_type"])
        output_id = str(row["output_id"])
        source_path = Path(row["output_path"])
        if output_type == "surah":
            info = SURAH_BY_NUMBER[int(output_id)]
            dest_dir = release_root / "audio" / "by-surah"
            release_name = f"{info.number:03d}-{slugify(info.transliteration)}.mp3"
        elif output_type == "part":
            dest_dir = release_root / "audio" / "30-parts"
            release_name = f"part-{int(output_id):02d}.mp3"
        else:
            continue
        outputs.append(
            ReleaseOutput(
                output_type=output_type,
                output_id=output_id,
                label=str(row["label"]),
                start_ref=str(row["start_ref"]),
                end_ref=str(row["end_ref"]),
                source_path=source_path,
                release_path=dest_dir / release_name,
                expected_duration_seconds=float(row["duration_seconds"] or 0),
                expected_bytes=int(row["bytes"] or 0),
            )
        )
    return outputs


def build_full_book_output(
    conn: sqlite3.Connection,
    audio_run_id: str,
    release_root: Path,
) -> ReleaseOutput:
    totals = conn.execute(
        """
        SELECT
            COALESCE(SUM(duration_seconds), 0) AS duration_seconds,
            COALESCE(SUM(bytes), 0) AS bytes
        FROM audio_chunks
        WHERE audio_run_id = ? AND status = 'complete'
        """,
        (audio_run_id,),
    ).fetchone()
    first = conn.execute(
        """
        SELECT start_ref
        FROM audio_chunks
        WHERE audio_run_id = ? AND status = 'complete'
        ORDER BY chunk_index
        LIMIT 1
        """,
        (audio_run_id,),
    ).fetchone()
    last = conn.execute(
        """
        SELECT end_ref
        FROM audio_chunks
        WHERE audio_run_id = ? AND status = 'complete'
        ORDER BY chunk_index DESC
        LIMIT 1
        """,
        (audio_run_id,),
    ).fetchone()
    return ReleaseOutput(
        output_type="full",
        output_id="complete",
        label="Complete Book",
        start_ref=str(first["start_ref"] if first else "1:1"),
        end_ref=str(last["end_ref"] if last else "114:6"),
        source_path=release_root / "audio" / "full-book" / "quran-translation-complete.mp3",
        release_path=release_root / "audio" / "full-book" / "quran-translation-complete.mp3",
        expected_duration_seconds=float(totals["duration_seconds"] or 0),
        expected_bytes=int(totals["bytes"] or 0),
    )


def concat_full_book(
    conn: sqlite3.Connection,
    audio_run_id: str,
    output: ReleaseOutput,
) -> None:
    run = audio_run(conn, audio_run_id)
    chunks = list(
        conn.execute(
            """
            SELECT output_path
            FROM audio_chunks
            WHERE audio_run_id = ? AND status = 'complete'
            ORDER BY chunk_index
            """,
            (audio_run_id,),
        )
    )
    if len(chunks) != 383:
        raise SystemExit(f"Cannot build full-book MP3: expected 383 chunks, found {len(chunks)}.")

    output.release_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output.release_path.with_suffix(".concat.txt")
    list_path.write_text(
        "\n".join(ffconcat_path(Path(row["output_path"])) for row in chunks) + "\n",
        encoding="utf-8",
    )
    metadata = metadata_for_output(run, output)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-fflags",
        "+genpts",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-map",
        "0:a:0",
        "-c",
        "copy",
        "-map_metadata",
        "-1",
        "-id3v2_version",
        "3",
        "-write_id3v1",
        "1",
    ]
    for key, value in metadata.items():
        command.extend(["-metadata", f"{key}={value}"])
    command.append(str(output.release_path))
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    finally:
        list_path.unlink(missing_ok=True)


def remux_with_metadata(
    conn: sqlite3.Connection,
    audio_run_id: str,
    output: ReleaseOutput,
) -> None:
    run = audio_run(conn, audio_run_id)
    output.release_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata_for_output(run, output)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-fflags",
        "+genpts",
        "-i",
        str(output.source_path),
        "-map",
        "0:a:0",
        "-c",
        "copy",
        "-map_metadata",
        "-1",
        "-id3v2_version",
        "3",
        "-write_id3v1",
        "1",
    ]
    for key, value in metadata.items():
        command.extend(["-metadata", f"{key}={value}"])
    command.append(str(output.release_path))
    subprocess.run(command, check=True, capture_output=True, text=True)


def metadata_for_output(run: sqlite3.Row, output: ReleaseOutput) -> dict[str, str]:
    comment = (
        f"Audio run: {run['audio_run_id']}; voice: {VOICE_LABEL}; "
        f"voice_id: {run['voice_id']}; model: {run['model_id']}; "
        f"range: {output.start_ref}-{output.end_ref}"
    )
    if output.output_type == "surah":
        info = SURAH_BY_NUMBER[int(output.output_id)]
        return {
            "title": f"{info.number:03d}. {info.transliteration}",
            "album": SURAH_ALBUM,
            "artist": VOICE_LABEL,
            "album_artist": "Quran Translation v2",
            "track": f"{info.number}/114",
            "date": RELEASE_YEAR,
            "genre": "Spoken Word",
            "comment": comment,
        }
    if output.output_type == "full":
        return {
            "title": "Quran Translation - Complete Book",
            "album": FULL_BOOK_ALBUM,
            "artist": VOICE_LABEL,
            "album_artist": "Quran Translation v2",
            "track": "1/1",
            "date": RELEASE_YEAR,
            "genre": "Spoken Word",
            "comment": comment,
        }
    return {
        "title": f"Part {int(output.output_id):02d}",
        "album": PART_ALBUM,
        "artist": VOICE_LABEL,
        "album_artist": "Quran Translation v2",
        "track": f"{int(output.output_id)}/30",
        "date": RELEASE_YEAR,
        "genre": "Spoken Word",
        "comment": comment,
    }


def copy_publication_artifacts(release_root: Path) -> None:
    copies = [
        (OUTPUT_DIR / "book" / "quran-translation-book.pdf", release_root / "pdf" / "print-book"),
        (
            OUTPUT_DIR / "book" / "quran-translation-book.inspection.json",
            release_root / "pdf" / "print-book",
        ),
        (
            OUTPUT_DIR / "book" / "quran-translation-reader-edition.pdf",
            release_root / "pdf" / "reader-book",
        ),
        (
            OUTPUT_DIR / "book" / "quran-translation-reader-edition.inspection.json",
            release_root / "pdf" / "reader-book",
        ),
        (
            OUTPUT_DIR / "publication" / "quran-publication.md",
            release_root / "text" / "publication",
        ),
        (
            OUTPUT_DIR / "publication" / "quran-publication.json",
            release_root / "text" / "publication",
        ),
    ]
    for source, dest_dir in copies:
        if not source.exists():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_dir / source.name)


def build_release_manifest(
    conn: sqlite3.Connection,
    audio_run_id: str,
    release_root: Path,
    release_outputs: list[ReleaseOutput],
) -> dict[str, Any]:
    run = dict(audio_run(conn, audio_run_id))
    status = audio_status(conn, audio_run_id)
    outputs = []
    for output in release_outputs:
        probe = ffprobe_json(output.release_path)
        outputs.append(
            {
                "type": output.output_type,
                "id": output.output_id,
                "label": output.label,
                "range": f"{output.start_ref}-{output.end_ref}",
                "path": relative_to(output.release_path, release_root),
                "sha256": file_sha256(output.release_path),
                "bytes": output.release_path.stat().st_size,
                "duration_seconds": float(probe["format"]["duration"]),
                "bit_rate": int(probe["format"].get("bit_rate") or 0),
            }
        )
    pdf_outputs = [
        {
            "path": relative_to(path, release_root),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted((release_root / "pdf").rglob("*"))
        if path.is_file()
    ]
    text_outputs = [
        {
            "path": relative_to(path, release_root),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted((release_root / "text").rglob("*"))
        if path.is_file()
    ]
    return {
        "release": {
            "name": "quran-translation-v2",
            "root": str(release_root),
            "year": RELEASE_YEAR,
            "voice_label": VOICE_LABEL,
        },
        "audio_run": run,
        "audio_status": status,
        "counts": {
            "surah_mp3": sum(1 for item in outputs if item["type"] == "surah"),
            "part_mp3": sum(1 for item in outputs if item["type"] == "part"),
            "full_book_mp3": sum(1 for item in outputs if item["type"] == "full"),
            "pdf": len(pdf_outputs),
            "text": len(text_outputs),
        },
        "outputs": outputs,
        "pdf": pdf_outputs,
        "text": text_outputs,
    }


def qa_release_audio(
    release_outputs: list[ReleaseOutput],
    *,
    decode_check: bool,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    checked_files: list[dict[str, Any]] = []
    for output in release_outputs:
        check_audio_file(
            path=output.release_path,
            expected_duration_seconds=output.expected_duration_seconds,
            expected_bytes=0,
            scope=f"release:{output.output_type}:{output.output_id}",
            issues=issues,
            ffprobe_check=True,
        )
        probe = ffprobe_json(output.release_path)
        stream = first_audio_stream(probe)
        file_record: dict[str, Any] = {
            "type": output.output_type,
            "id": output.output_id,
            "path": str(output.release_path),
            "duration_seconds": float(probe["format"]["duration"]),
            "bit_rate": int(probe["format"].get("bit_rate") or 0),
            "codec": stream.get("codec_name"),
            "sample_rate": int(stream.get("sample_rate") or 0),
        }
        if decode_check:
            decode_error = decode_audio(output.release_path)
            file_record["decode_ok"] = decode_error is None
            if decode_error:
                issues.append(
                    {
                        "scope": f"decode:{output.output_type}:{output.output_id}",
                        "message": decode_error,
                    }
                )
        checked_files.append(file_record)

    return {
        "checked_files": len(checked_files),
        "decode_check": decode_check,
        "issue_count": len(issues),
        "issues": issues,
        "files": checked_files,
    }


def check_audio_file(
    *,
    path: Path,
    expected_duration_seconds: float,
    expected_bytes: int,
    scope: str,
    issues: list[dict[str, Any]],
    ffprobe_check: bool,
) -> None:
    if not path.exists():
        issues.append({"scope": scope, "message": f"Missing file: {path}"})
        return
    if path.stat().st_size <= 0:
        issues.append({"scope": scope, "message": f"Empty file: {path}"})
        return
    if expected_bytes and abs(path.stat().st_size - expected_bytes) > max(4096, expected_bytes * 0.02):
        issues.append(
            {
                "scope": scope,
                "message": (
                    f"Unexpected byte size for {path}: {path.stat().st_size} "
                    f"vs database {expected_bytes}."
                ),
            }
        )
    if not ffprobe_check:
        return
    try:
        probe = ffprobe_json(path)
    except subprocess.CalledProcessError as exc:
        issues.append({"scope": scope, "message": exc.stderr.strip() or str(exc)})
        return
    stream = first_audio_stream(probe)
    duration = float(probe["format"].get("duration") or 0)
    bit_rate = int(probe["format"].get("bit_rate") or 0)
    sample_rate = int(stream.get("sample_rate") or 0)
    if stream.get("codec_name") != "mp3":
        issues.append({"scope": scope, "message": f"Expected mp3 codec for {path}."})
    if duration <= 0:
        issues.append({"scope": scope, "message": f"Non-positive duration for {path}."})
    duration_tolerance = max(1.5, min(30.0, expected_duration_seconds * 0.0005))
    if expected_duration_seconds and abs(duration - expected_duration_seconds) > duration_tolerance:
        issues.append(
            {
                "scope": scope,
                "message": (
                    f"Duration drift for {path}: {duration:.3f}s "
                    f"vs expected {expected_duration_seconds:.3f}s."
                ),
            }
        )
    if bit_rate and bit_rate < MIN_REASONABLE_BITRATE:
        issues.append({"scope": scope, "message": f"Low bitrate for {path}: {bit_rate}."})
    if sample_rate != 44100:
        issues.append({"scope": scope, "message": f"Expected 44100 Hz for {path}."})


def ffprobe_json(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-print_format",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def first_audio_stream(probe: dict[str, Any]) -> dict[str, Any]:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    return {}


def decode_audio(path: Path) -> str | None:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or f"ffmpeg decode failed for {path}").strip()


def ffconcat_path(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


def write_checksums(release_root: Path) -> Path:
    checksum_path = release_root / "SHA256SUMS.txt"
    paths = [
        path
        for path in sorted(release_root.rglob("*"))
        if path.is_file() and path.name != checksum_path.name
    ]
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative_to(path, release_root)}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def write_qa_summary(path: Path, release_manifest: dict[str, Any], qa: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = release_manifest["audio_status"]
    lines = [
        "# Audio QA Summary",
        "",
        f"- Audio run: `{status['audio_run_id']}`",
        f"- Voice: {VOICE_LABEL}",
        f"- Model: `{status['model_id']}`",
        f"- Chunks: {status['chunk_status'].get('complete', 0)}/{status['chunks']} complete",
        f"- Surah MP3s: {release_manifest['counts']['surah_mp3']}",
        f"- 30-part MP3s: {release_manifest['counts']['part_mp3']}",
        f"- Full-book MP3s: {release_manifest['counts']['full_book_mp3']}",
        f"- Generated duration: {format_hours(status['generated_duration_seconds'])}",
        f"- Release decode check: {'passed' if qa['issue_count'] == 0 else 'failed'}",
        f"- Issue count: {qa['issue_count']}",
        "",
        "Automated checks include file existence, positive duration, MP3 codec, 44.1 kHz sample rate, bitrate floor, duration drift against the database, and full ffmpeg decode of release MP3s.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_release_readme(path: Path, release_manifest: dict[str, Any], qa: dict[str, Any]) -> None:
    status = release_manifest["audio_status"]
    lines = [
        "# Quran Translation v2 Release",
        "",
        "This folder packages the completed English translation artifacts and the ElevenLabs audio release.",
        "",
        "## Audio",
        "",
        f"- Voice: {VOICE_LABEL}",
        f"- ElevenLabs model: `{status['model_id']}`",
        f"- Output format: `{status['output_format']}`",
        f"- Total generated text: {status['characters']:,} characters",
        f"- Estimated Multilingual v2 synthesis cost: ${status['estimated_multilingual_v2_cost_usd']:.2f}",
        f"- Total listening duration: {format_hours(status['generated_duration_seconds'])}",
        f"- By-surah files: {release_manifest['counts']['surah_mp3']}",
        f"- 30-part files: {release_manifest['counts']['part_mp3']}",
        f"- Full-book files: {release_manifest['counts']['full_book_mp3']}",
        "",
        "Release MP3s are clean tagged remuxes of the generated audio. They were not re-synthesized and not lossy re-encoded.",
        "",
        "## Folders",
        "",
        "- `audio/by-surah`: one MP3 per surah.",
        "- `audio/30-parts`: thirty longer MP3s for broad listening sessions.",
        "- `audio/full-book`: one complete-book MP3.",
        "- `pdf/print-book`: printable ayah-numbered book PDF.",
        "- `pdf/reader-book`: paragraph-style reader PDF.",
        "- `text/publication`: publication-layer Markdown and JSON.",
        "- `manifest`: release manifest, QA report, and QA summary.",
        "- `SHA256SUMS.txt`: checksums for every release file.",
        "",
        "## QA",
        "",
        f"- Automated release issues: {qa['issue_count']}",
        f"- Full decode check: {'enabled' if qa['decode_check'] else 'not run'}",
        "",
        "A human listening pass is still recommended before public distribution, especially for the longest surahs and part transitions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative_to(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def format_hours(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = int(seconds % 60)
    return f"{hours}h {minutes}m {remaining}s"
