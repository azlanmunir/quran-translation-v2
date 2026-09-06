"""Resumable ElevenLabs audio generation and assembly."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import time
import warnings
from pathlib import Path

from .config import OUTPUT_DIR, file_sha256, text_sha256
from .production_packets import atomic_json
from .db import utc_now
from .elevenlabs_tts import (
    ElevenLabsError,
    synthesize_text,
)
from .metadata import SURAHS, SURAH_BY_NUMBER
from .publication import publication_rows


DEFAULT_AUDIO_RUN_ID = "nathan_multilingual_v2"
DEFAULT_CHUNK_TARGET_CHARS = 3000
DEFAULT_CONTEXT_CHARS = 450
DEFAULT_PART_COUNT = 30
DEFAULT_VOICE_ID = "lWDDHwXsJXJM7nv2YgHY"
MP3_EXTENSIONS = {
    "mp3_44100_128": ".mp3",
    "mp3_44100_192": ".mp3",
    "mp3_44100_96": ".mp3",
}


def slugify(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def ref_sort_key(ref: str) -> tuple[int, int]:
    surah, ayah = ref.split(":", 1)
    return int(surah), int(ayah)


def chunk_filename(chunk_index: int, start_ref: str, end_ref: str, output_format: str) -> str:
    extension = MP3_EXTENSIONS.get(output_format, ".mp3")
    safe_start = start_ref.replace(":", "_")
    safe_end = end_ref.replace(":", "_")
    return f"{chunk_index:04d}-{safe_start}-{safe_end}{extension}"


def surah_title(number: int) -> str:
    info = SURAH_BY_NUMBER[number]
    return f"Surah {info.number}. {info.transliteration}.\n\n"


def clean_tts_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prepare_audio_chunks(
    conn: sqlite3.Connection,
    *,
    audio_run_id: str,
    translation_run_id: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    chunk_target_chars: int = DEFAULT_CHUNK_TARGET_CHARS,
    force: bool = False,
) -> dict[str, object]:
    if force:
        warnings.warn(
            "audio-prepare --force is deprecated and does not overwrite audio. "
            "Matching inputs are reused; changed inputs require a new audio_run_id.",
            FutureWarning,
            stacklevel=2,
        )
    existing = conn.execute(
        "SELECT audio_run_id FROM audio_runs WHERE audio_run_id = ?",
        (audio_run_id,),
    ).fetchone()
    existing_chunks = int(
        conn.execute(
            "SELECT COUNT(*) AS count FROM audio_chunks WHERE audio_run_id = ?",
            (audio_run_id,),
        ).fetchone()["count"]
    )

    rows = publication_rows(conn, translation_run_id)
    if not rows:
        raise SystemExit(
            f"No publication layer found for {translation_run_id}; run publication-build first."
        )

    output_root = OUTPUT_DIR / "audio" / "runs" / audio_run_id / "chunks"
    by_surah: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_surah.setdefault(int(row["surah_number"]), []).append(row)

    now = utc_now()
    chunks: list[dict[str, object]] = []
    chunk_index = 1
    for info in SURAHS:
        surah_rows = by_surah.get(info.number, [])
        current_parts: list[str] = []
        current_start_ref = ""
        current_end_ref = ""

        for ayah_index, row in enumerate(surah_rows):
            ayah_text = clean_tts_text(str(row["translation"]))
            prefix = surah_title(info.number) if ayah_index == 0 else ""
            next_part = prefix + ayah_text
            candidate = (" ".join(current_parts + [next_part])).strip()
            if current_parts and len(candidate) > chunk_target_chars:
                text = " ".join(current_parts).strip()
                chunks.append(
                    build_chunk_record(
                        audio_run_id=audio_run_id,
                        chunk_index=chunk_index,
                        surah_number=info.number,
                        start_ref=current_start_ref,
                        end_ref=current_end_ref,
                        text=text,
                        output_root=output_root,
                        output_format=output_format,
                    )
                )
                chunk_index += 1
                current_parts = [next_part]
                current_start_ref = str(row["verse_key"])
            else:
                current_parts.append(next_part)
                if not current_start_ref:
                    current_start_ref = str(row["verse_key"])
            current_end_ref = str(row["verse_key"])

        if current_parts:
            text = " ".join(current_parts).strip()
            chunks.append(
                build_chunk_record(
                    audio_run_id=audio_run_id,
                    chunk_index=chunk_index,
                    surah_number=info.number,
                    start_ref=current_start_ref,
                    end_ref=current_end_ref,
                    text=text,
                    output_root=output_root,
                    output_format=output_format,
                )
            )
            chunk_index += 1

    if existing and existing_chunks:
        prior = audio_run(conn, audio_run_id)
        same_settings = all(prior[key] == value for key, value in {
            "translation_run_id": translation_run_id, "voice_id": voice_id,
            "model_id": model_id, "output_format": output_format,
            "chunk_target_chars": chunk_target_chars,
        }.items())
        old_chunks = list(conn.execute(
            "SELECT chunk_id, text_sha256 FROM audio_chunks WHERE audio_run_id = ? ORDER BY chunk_index",
            (audio_run_id,),
        ))
        if not same_settings or [(row["chunk_id"], row["text_sha256"]) for row in old_chunks] != [
            (row["chunk_id"], row["text_sha256"]) for row in chunks
        ]:
            raise ValueError("Audio inputs changed; preserve the existing run and use a new audio_run_id")
        for completed in complete_chunks(conn, audio_run_id):
            _validate_existing_audio(conn, completed)
        return audio_status(conn, audio_run_id)

    with conn:
        if existing:
            conn.execute("DELETE FROM audio_outputs WHERE audio_run_id = ?", (audio_run_id,))
            conn.execute("DELETE FROM audio_chunks WHERE audio_run_id = ?", (audio_run_id,))
            conn.execute("DELETE FROM audio_runs WHERE audio_run_id = ?", (audio_run_id,))
        conn.execute(
            """
            INSERT INTO audio_runs (
                audio_run_id,
                translation_run_id,
                voice_id,
                model_id,
                output_format,
                chunk_target_chars,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'prepared', ?, ?)
            """,
            (
                audio_run_id,
                translation_run_id,
                voice_id,
                model_id,
                output_format,
                chunk_target_chars,
                now,
                now,
            ),
        )
        for chunk in chunks:
            conn.execute(
                """
                INSERT INTO audio_chunks (
                    audio_run_id,
                    chunk_id,
                    chunk_index,
                    surah_number,
                    start_ref,
                    end_ref,
                    text,
                    text_sha256,
                    char_count,
                    output_path,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    audio_run_id,
                    chunk["chunk_id"],
                    chunk["chunk_index"],
                    chunk["surah_number"],
                    chunk["start_ref"],
                    chunk["end_ref"],
                    chunk["text"],
                    chunk["text_sha256"],
                    chunk["char_count"],
                    chunk["output_path"],
                    now,
                    now,
                ),
            )
    write_audio_manifest(conn, audio_run_id)
    return audio_status(conn, audio_run_id)


def build_chunk_record(
    *,
    audio_run_id: str,
    chunk_index: int,
    surah_number: int,
    start_ref: str,
    end_ref: str,
    text: str,
    output_root: Path,
    output_format: str,
) -> dict[str, object]:
    info = SURAH_BY_NUMBER[surah_number]
    surah_dir = output_root / f"{surah_number:03d}-{slugify(info.transliteration)}"
    chunk_id = f"{audio_run_id}-{chunk_index:04d}"
    output_path = surah_dir / chunk_filename(chunk_index, start_ref, end_ref, output_format)
    return {
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "surah_number": surah_number,
        "start_ref": start_ref,
        "end_ref": end_ref,
        "text": text,
        "text_sha256": text_sha256(text),
        "char_count": len(text),
        "output_path": str(output_path),
    }


def audio_run(conn: sqlite3.Connection, audio_run_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM audio_runs WHERE audio_run_id = ?",
        (audio_run_id,),
    ).fetchone()
    if not row:
        raise SystemExit(f"No audio run exists with id {audio_run_id}. Run audio-prepare first.")
    return row


def audio_status(conn: sqlite3.Connection, audio_run_id: str) -> dict[str, object]:
    run = audio_run(conn, audio_run_id)
    status_rows = list(
        conn.execute(
            """
            SELECT status, COUNT(*) AS count, COALESCE(SUM(char_count), 0) AS chars
            FROM audio_chunks
            WHERE audio_run_id = ?
            GROUP BY status
            ORDER BY status
            """,
            (audio_run_id,),
        )
    )
    output_rows = list(
        conn.execute(
            """
            SELECT output_type, COUNT(*) AS count, COALESCE(SUM(duration_seconds), 0) AS duration
            FROM audio_outputs
            WHERE audio_run_id = ? AND status = 'complete'
            GROUP BY output_type
            ORDER BY output_type
            """,
            (audio_run_id,),
        )
    )
    totals = conn.execute(
        """
        SELECT
            COUNT(*) AS chunks,
            COALESCE(SUM(char_count), 0) AS chars,
            COALESCE(SUM(duration_seconds), 0) AS duration,
            COALESCE(SUM(bytes), 0) AS bytes
        FROM audio_chunks
        WHERE audio_run_id = ?
        """,
        (audio_run_id,),
    ).fetchone()
    return {
        "audio_run_id": audio_run_id,
        "translation_run_id": run["translation_run_id"],
        "voice_id": run["voice_id"],
        "model_id": run["model_id"],
        "output_format": run["output_format"],
        "status": run["status"],
        "chunks": int(totals["chunks"]),
        "characters": int(totals["chars"]),
        "estimated_multilingual_v2_cost_usd": round(int(totals["chars"]) / 1000 * 0.10, 2),
        "generated_duration_seconds": float(totals["duration"]),
        "generated_bytes": int(totals["bytes"]),
        "chunk_status": {row["status"]: int(row["count"]) for row in status_rows},
        "chunk_chars_by_status": {row["status"]: int(row["chars"]) for row in status_rows},
        "outputs": {row["output_type"]: int(row["count"]) for row in output_rows},
        "output_duration_seconds": {
            row["output_type"]: float(row["duration"]) for row in output_rows
        },
    }


def pending_chunks(
    conn: sqlite3.Connection,
    audio_run_id: str,
    *,
    retry_failed: bool = False,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    statuses = ("pending", "running") + (("failed",) if retry_failed else ())
    placeholders = ", ".join("?" for _ in statuses)
    sql = f"""
        SELECT *
        FROM audio_chunks
        WHERE audio_run_id = ?
          AND status IN ({placeholders})
        ORDER BY chunk_index
    """
    params: list[object] = [audio_run_id, *statuses]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return list(conn.execute(sql, params))


def synthesize_audio_chunks(
    conn: sqlite3.Connection,
    *,
    audio_run_id: str,
    retry_failed: bool = False,
    limit: int | None = None,
    max_attempts: int = 3,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
    request_timeout_seconds: int = 240,
    sleep_seconds: float = 0.0,
    stop_on_error: bool = False,
) -> dict[str, object]:
    run = audio_run(conn, audio_run_id)
    chunks = pending_chunks(conn, audio_run_id, retry_failed=retry_failed, limit=limit)
    for completed in complete_chunks(conn, audio_run_id):
        _validate_existing_audio(conn, completed, context_chars=context_chars)
    processed = 0
    failed = 0
    quota_paused = False

    with conn:
        conn.execute(
            "UPDATE audio_runs SET status = 'generating', updated_at = ? WHERE audio_run_id = ?",
            (utc_now(), audio_run_id),
        )

    for chunk in chunks:
        try:
            if maybe_mark_existing_complete(conn, chunk, context_chars=context_chars):
                processed += 1
                print(progress_line(conn, audio_run_id, chunk, reused=True), flush=True)
                continue
            synthesize_one_chunk(
                conn,
                run=run,
                chunk=chunk,
                max_attempts=max_attempts,
                context_chars=context_chars,
                request_timeout_seconds=request_timeout_seconds,
            )
            processed += 1
            print(progress_line(conn, audio_run_id, chunk), flush=True)
            if sleep_seconds:
                time.sleep(sleep_seconds)
        except ElevenLabsError as exc:
            failed += 1
            mark_chunk_failed(conn, chunk, str(exc))
            if is_quota_error(str(exc)):
                quota_paused = True
                break
            if stop_on_error:
                break
        except Exception as exc:
            failed += 1
            mark_chunk_failed(conn, chunk, repr(exc))
            if stop_on_error:
                break

    update_audio_run_status(conn, audio_run_id, quota_paused=quota_paused)
    write_audio_manifest(conn, audio_run_id)
    status = audio_status(conn, audio_run_id)
    status["processed_this_call"] = processed
    status["failed_this_call"] = failed
    return status


def synthesize_one_chunk(
    conn: sqlite3.Connection,
    *,
    run: sqlite3.Row,
    chunk: sqlite3.Row,
    max_attempts: int,
    context_chars: int,
    request_timeout_seconds: int,
) -> None:
    output_path = Path(chunk["output_path"])
    if maybe_mark_existing_complete(conn, chunk, context_chars=context_chars):
        return
    last_error = ""
    for _ in range(max_attempts):
        now = utc_now()
        with conn:
            conn.execute(
                """
                UPDATE audio_chunks
                SET status = 'running',
                    attempts = attempts + 1,
                    last_error = NULL,
                    updated_at = ?
                WHERE audio_run_id = ? AND chunk_id = ?
                """,
                (now, chunk["audio_run_id"], chunk["chunk_id"]),
            )
        previous_text, next_text = chunk_context(conn, chunk, context_chars=context_chars)
        try:
            synthesize_text(
                text=chunk["text"],
                voice_id=run["voice_id"],
                output_path=output_path,
                model_id=run["model_id"],
                output_format=run["output_format"],
                previous_text=previous_text,
                next_text=next_text,
                seed=seed_for_chunk(chunk["text_sha256"]),
                apply_text_normalization="auto",
                request_timeout_seconds=request_timeout_seconds,
            )
            mark_chunk_complete(conn, chunk, output_path, context_chars=context_chars)
            return
        except ElevenLabsError as exc:
            last_error = str(exc)
            if is_quota_error(last_error):
                raise
            time.sleep(2)
    raise ElevenLabsError(last_error or "ElevenLabs generation failed")


def _audio_provenance(conn, chunk, context_chars: int) -> dict:
    run = audio_run(conn, chunk["audio_run_id"])
    previous, following = chunk_context(conn, chunk, context_chars=context_chars)
    return {
        "text_sha256": text_sha256(chunk["text"]),
        "voice_id": run["voice_id"], "model_id": run["model_id"],
        "output_format": run["output_format"], "seed": seed_for_chunk(chunk["text_sha256"]),
        "previous_text": previous, "next_text": following, "normalization": "auto",
        "context_chars": context_chars,
    }


def _validate_existing_audio(
    conn: sqlite3.Connection, chunk: sqlite3.Row, *, context_chars: int | None = None
) -> bool:
    output_path = Path(chunk["output_path"])
    receipt_path = output_path.with_suffix(".receipt.json")
    if output_path.exists() and output_path.stat().st_size > 0:
        if not receipt_path.is_file():
            raise ElevenLabsError("Existing audio lacks provenance; preserve it for explicit review")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if context_chars is None:
            context_chars = receipt.get("inputs", {}).get("context_chars", DEFAULT_CONTEXT_CHARS)
        if receipt.get("inputs") != _audio_provenance(conn, chunk, context_chars) or (
            receipt.get("sha256") != file_sha256(output_path)
        ):
            raise ElevenLabsError("Existing audio provenance or checksum changed")
        return True
    if chunk["status"] == "complete" or receipt_path.exists():
        raise ElevenLabsError("Completed audio is missing or empty; explicit recovery required")
    return False


def maybe_mark_existing_complete(
    conn: sqlite3.Connection, chunk: sqlite3.Row, *, context_chars: int = DEFAULT_CONTEXT_CHARS
) -> bool:
    if not _validate_existing_audio(conn, chunk, context_chars=context_chars):
        return False
    mark_chunk_complete(conn, chunk, Path(chunk["output_path"]), context_chars=context_chars)
    return True


def mark_chunk_complete(
    conn: sqlite3.Connection, chunk: sqlite3.Row, output_path: Path,
    *, context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> None:
    duration = probe_duration(output_path)
    atomic_json(output_path.with_suffix(".receipt.json"), {
        "version": "audio-chunk-provenance-v1",
        "inputs": _audio_provenance(conn, chunk, context_chars),
        "sha256": file_sha256(output_path),
    })
    now = utc_now()
    with conn:
        conn.execute(
            """
            UPDATE audio_chunks
            SET status = 'complete',
                duration_seconds = ?,
                bytes = ?,
                last_error = NULL,
                updated_at = ?
            WHERE audio_run_id = ? AND chunk_id = ?
            """,
            (
                duration,
                output_path.stat().st_size,
                now,
                chunk["audio_run_id"],
                chunk["chunk_id"],
            ),
        )


def mark_chunk_failed(conn: sqlite3.Connection, chunk: sqlite3.Row, error: str) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            """
            UPDATE audio_chunks
            SET status = 'failed',
                last_error = ?,
                updated_at = ?
            WHERE audio_run_id = ? AND chunk_id = ?
            """,
            (error[:2000], now, chunk["audio_run_id"], chunk["chunk_id"]),
        )


def update_audio_run_status(
    conn: sqlite3.Connection,
    audio_run_id: str,
    *,
    quota_paused: bool = False,
) -> None:
    counts = {
        row["status"]: int(row["count"])
        for row in conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM audio_chunks
            WHERE audio_run_id = ?
            GROUP BY status
            """,
            (audio_run_id,),
        )
    }
    if quota_paused:
        status = "quota_paused"
    elif counts.get("failed", 0):
        status = "failed"
    elif counts.get("pending", 0) or counts.get("running", 0):
        status = "generating"
    else:
        status = "generated"
    with conn:
        conn.execute(
            "UPDATE audio_runs SET status = ?, updated_at = ? WHERE audio_run_id = ?",
            (status, utc_now(), audio_run_id),
        )


def progress_line(
    conn: sqlite3.Connection,
    audio_run_id: str,
    chunk: sqlite3.Row,
    *,
    reused: bool = False,
) -> str:
    status = audio_status(conn, audio_run_id)
    complete = status["chunk_status"].get("complete", 0)
    total = status["chunks"]
    prefix = "reused" if reused else "complete"
    return f"{prefix} {complete}/{total} {chunk['start_ref']}-{chunk['end_ref']} chars={chunk['char_count']}"


def chunk_context(
    conn: sqlite3.Connection,
    chunk: sqlite3.Row,
    *,
    context_chars: int,
) -> tuple[str | None, str | None]:
    previous_row = conn.execute(
        """
        SELECT text
        FROM audio_chunks
        WHERE audio_run_id = ? AND chunk_index = ?
        """,
        (chunk["audio_run_id"], int(chunk["chunk_index"]) - 1),
    ).fetchone()
    next_row = conn.execute(
        """
        SELECT text
        FROM audio_chunks
        WHERE audio_run_id = ? AND chunk_index = ?
        """,
        (chunk["audio_run_id"], int(chunk["chunk_index"]) + 1),
    ).fetchone()
    previous_text = previous_row["text"][-context_chars:] if previous_row else None
    next_text = next_row["text"][:context_chars] if next_row else None
    return previous_text, next_text


def seed_for_chunk(text_hash: str) -> int:
    return int(text_hash[:8], 16)


def is_quota_error(error: str) -> bool:
    lowered = error.lower()
    return any(token in lowered for token in ("quota", "credit", "billing", "limit exceeded"))


def complete_chunks(conn: sqlite3.Connection, audio_run_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT *
            FROM audio_chunks
            WHERE audio_run_id = ? AND status = 'complete'
            ORDER BY chunk_index
            """,
            (audio_run_id,),
        )
    )


def require_all_chunks_complete(conn: sqlite3.Connection, audio_run_id: str) -> None:
    for chunk in complete_chunks(conn, audio_run_id):
        _validate_existing_audio(conn, chunk)
    incomplete = list(
        conn.execute(
            """
            SELECT chunk_id, status, start_ref, end_ref
            FROM audio_chunks
            WHERE audio_run_id = ? AND status != 'complete'
            ORDER BY chunk_index
            LIMIT 20
            """,
            (audio_run_id,),
        )
    )
    if incomplete:
        sample = ", ".join(
            f"{row['chunk_id']}:{row['status']}:{row['start_ref']}-{row['end_ref']}"
            for row in incomplete[:5]
        )
        raise SystemExit(f"Cannot assemble: incomplete chunks remain ({sample}).")


def assemble_surah_outputs(
    conn: sqlite3.Connection,
    audio_run_id: str,
    *,
    allow_partial: bool = False,
) -> dict[str, object]:
    if not allow_partial:
        require_all_chunks_complete(conn, audio_run_id)
    else:
        for chunk in complete_chunks(conn, audio_run_id):
            _validate_existing_audio(conn, chunk)
    run = audio_run(conn, audio_run_id)
    output_dir = OUTPUT_DIR / "audio" / "surahs" / audio_run_id
    chunks = list(
        conn.execute(
            """
            SELECT *
            FROM audio_chunks
            WHERE audio_run_id = ?
            ORDER BY chunk_index
            """,
            (audio_run_id,),
        )
    )
    by_surah: dict[int, list[sqlite3.Row]] = {}
    for chunk in chunks:
        by_surah.setdefault(int(chunk["surah_number"]), []).append(chunk)

    created = 0
    with conn:
        conn.execute(
            "DELETE FROM audio_outputs WHERE audio_run_id = ? AND output_type = 'surah'",
            (audio_run_id,),
        )
    for info in SURAHS:
        surah_chunks = by_surah.get(info.number, [])
        if not surah_chunks:
            continue
        if allow_partial and any(row["status"] != "complete" for row in surah_chunks):
            continue
        output_path = output_dir / f"{info.number:03d}-{slugify(info.transliteration)}.mp3"
        concat_mp3([Path(row["output_path"]) for row in surah_chunks], output_path)
        insert_audio_output(
            conn,
            audio_run_id=audio_run_id,
            output_type="surah",
            output_id=f"{info.number:03d}",
            label=info.transliteration,
            start_ref=surah_chunks[0]["start_ref"],
            end_ref=surah_chunks[-1]["end_ref"],
            output_path=output_path,
            source_chunks=[row["chunk_id"] for row in surah_chunks],
        )
        created += 1
    write_audio_manifest(conn, audio_run_id)
    return {"audio_run_id": audio_run_id, "model_id": run["model_id"], "surahs": created}


def assemble_part_outputs(
    conn: sqlite3.Connection,
    audio_run_id: str,
    *,
    part_count: int = DEFAULT_PART_COUNT,
) -> dict[str, object]:
    require_all_chunks_complete(conn, audio_run_id)
    chunks = complete_chunks(conn, audio_run_id)
    groups = split_chunks_by_char_count(chunks, part_count)
    output_dir = OUTPUT_DIR / "audio" / "parts" / audio_run_id

    with conn:
        conn.execute(
            "DELETE FROM audio_outputs WHERE audio_run_id = ? AND output_type = 'part'",
            (audio_run_id,),
        )
    for index, group in enumerate(groups, start=1):
        output_path = output_dir / f"part-{index:02d}.mp3"
        concat_mp3([Path(row["output_path"]) for row in group], output_path)
        insert_audio_output(
            conn,
            audio_run_id=audio_run_id,
            output_type="part",
            output_id=f"{index:02d}",
            label=f"Part {index:02d}",
            start_ref=group[0]["start_ref"],
            end_ref=group[-1]["end_ref"],
            output_path=output_path,
            source_chunks=[row["chunk_id"] for row in group],
        )
    write_audio_manifest(conn, audio_run_id)
    return {"audio_run_id": audio_run_id, "parts": len(groups)}


def split_chunks_by_char_count(chunks: list[sqlite3.Row], part_count: int) -> list[list[sqlite3.Row]]:
    total_chars = sum(int(row["char_count"]) for row in chunks)
    groups: list[list[sqlite3.Row]] = []
    current: list[sqlite3.Row] = []
    cumulative = 0
    for index, chunk in enumerate(chunks):
        current.append(chunk)
        cumulative += int(chunk["char_count"])
        next_boundary = total_chars * (len(groups) + 1) / part_count
        remaining_chunks = len(chunks) - index - 1
        remaining_parts = part_count - len(groups) - 1
        if (
            len(groups) < part_count - 1
            and cumulative >= next_boundary
            and remaining_chunks >= remaining_parts
        ):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def insert_audio_output(
    conn: sqlite3.Connection,
    *,
    audio_run_id: str,
    output_type: str,
    output_id: str,
    label: str,
    start_ref: str,
    end_ref: str,
    output_path: Path,
    source_chunks: list[str],
) -> None:
    duration = probe_duration(output_path)
    now = utc_now()
    with conn:
        conn.execute(
            """
            INSERT INTO audio_outputs (
                audio_run_id,
                output_type,
                output_id,
                label,
                start_ref,
                end_ref,
                output_path,
                source_chunks_json,
                duration_seconds,
                bytes,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'complete', ?, ?)
            """,
            (
                audio_run_id,
                output_type,
                output_id,
                label,
                start_ref,
                end_ref,
                str(output_path),
                json.dumps(source_chunks),
                duration,
                output_path.stat().st_size,
                now,
                now,
            ),
        )


def ffconcat_path(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


def concat_mp3(input_paths: list[Path], output_path: Path) -> None:
    if not input_paths:
        raise ValueError("No input files to concatenate.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.with_suffix(".concat.txt")
    list_path.write_text("\n".join(ffconcat_path(path) for path in input_paths) + "\n")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def write_audio_manifest(conn: sqlite3.Connection, audio_run_id: str) -> Path:
    manifest_path = OUTPUT_DIR / "audio" / "runs" / audio_run_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run = audio_run(conn, audio_run_id)
    chunks = list(
        conn.execute(
            """
            SELECT chunk_id, chunk_index, surah_number, start_ref, end_ref, text_sha256,
                   char_count, output_path, status, attempts, duration_seconds, bytes, last_error
            FROM audio_chunks
            WHERE audio_run_id = ?
            ORDER BY chunk_index
            """,
            (audio_run_id,),
        )
    )
    outputs = list(
        conn.execute(
            """
            SELECT output_type, output_id, label, start_ref, end_ref, output_path,
                   source_chunks_json, duration_seconds, bytes, status
            FROM audio_outputs
            WHERE audio_run_id = ?
            ORDER BY output_type, output_id
            """,
            (audio_run_id,),
        )
    )
    payload = {
        "audio_run": dict(run),
        "status": audio_status(conn, audio_run_id),
        "chunks": [dict(row) for row in chunks],
        "outputs": [dict(row) for row in outputs],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return manifest_path
