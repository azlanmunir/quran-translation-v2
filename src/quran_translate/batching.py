"""Translation run and batch preparation."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_AFTER,
    DEFAULT_CONTEXT_BEFORE,
    DEFAULT_MAX_TARGET_CHARS,
    DEFAULT_MODEL,
    read_prompt_files,
)
from .db import utc_now


def make_run_id() -> str:
    return "run_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def source_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT *
            FROM source_ayahs
            ORDER BY surah_number, ayah_number
            """
        )
    )


def _split_surah_rows(
    rows: list[sqlite3.Row],
    batch_size: int,
    max_target_chars: int,
) -> list[list[sqlite3.Row]]:
    batches: list[list[sqlite3.Row]] = []
    current: list[sqlite3.Row] = []
    current_chars = 0

    for row in rows:
        row_chars = len(row["arabic_uthmani_min"])
        would_exceed_count = len(current) >= batch_size
        would_exceed_chars = current and current_chars + row_chars > max_target_chars
        if would_exceed_count or would_exceed_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += row_chars

    if current:
        batches.append(current)
    return batches


def build_batches(
    rows: list[sqlite3.Row],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_target_chars: int = DEFAULT_MAX_TARGET_CHARS,
) -> list[list[sqlite3.Row]]:
    batches: list[list[sqlite3.Row]] = []
    current_surah: int | None = None
    surah_rows: list[sqlite3.Row] = []

    for row in rows:
        surah_number = int(row["surah_number"])
        if current_surah is None:
            current_surah = surah_number
        if surah_number != current_surah:
            batches.extend(_split_surah_rows(surah_rows, batch_size, max_target_chars))
            current_surah = surah_number
            surah_rows = []
        surah_rows.append(row)

    if surah_rows:
        batches.extend(_split_surah_rows(surah_rows, batch_size, max_target_chars))

    return batches


def prepare_run(
    conn: sqlite3.Connection,
    run_id: str | None = None,
    model: str = DEFAULT_MODEL,
    prompt_version: str = "philological-v3+output-contract-v2",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_target_chars: int = DEFAULT_MAX_TARGET_CHARS,
    context_before: int = DEFAULT_CONTEXT_BEFORE,
    context_after: int = DEFAULT_CONTEXT_AFTER,
) -> str:
    rows = source_rows(conn)
    if not rows:
        raise SystemExit("No source ayahs found. Run import-source first.")

    run_id = run_id or make_run_id()
    _, _, prompt_hash = read_prompt_files()
    now = utc_now()
    batches = build_batches(rows, batch_size=batch_size, max_target_chars=max_target_chars)

    with conn:
        conn.execute(
            """
            INSERT INTO translation_runs (
                run_id,
                model,
                prompt_version,
                prompt_hash,
                batch_size,
                max_target_chars,
                context_before,
                context_after,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)
            """,
            (
                run_id,
                model,
                prompt_version,
                prompt_hash,
                batch_size,
                max_target_chars,
                context_before,
                context_after,
                now,
                now,
            ),
        )

        for index, batch in enumerate(batches, start=1):
            target_refs = [row["verse_key"] for row in batch]
            batch_id = f"{run_id}_b{index:04d}"
            conn.execute(
                """
                INSERT INTO translation_batches (
                    batch_id,
                    run_id,
                    batch_index,
                    surah_number,
                    start_ref,
                    end_ref,
                    target_refs_json,
                    status,
                    attempts,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
                """,
                (
                    batch_id,
                    run_id,
                    index,
                    int(batch[0]["surah_number"]),
                    target_refs[0],
                    target_refs[-1],
                    json.dumps(target_refs, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    return run_id


def run_status(conn: sqlite3.Connection, run_id: str) -> dict[str, int | str]:
    row = conn.execute(
        "SELECT * FROM translation_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if not row:
        raise SystemExit(f"Unknown run_id: {run_id}")

    counts = {
        item["status"]: int(item["count"])
        for item in conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM translation_batches
            WHERE run_id = ?
            GROUP BY status
            """,
            (run_id,),
        )
    }
    translated = conn.execute(
        "SELECT COUNT(*) AS count FROM translations WHERE run_id = ? AND status = 'complete'",
        (run_id,),
    ).fetchone()["count"]
    total_ayahs = conn.execute("SELECT COUNT(*) AS count FROM source_ayahs").fetchone()["count"]
    return {
        "run_id": run_id,
        "model": row["model"],
        "status": row["status"],
        "batches_pending": counts.get("pending", 0),
        "batches_running": counts.get("running", 0),
        "batches_complete": counts.get("complete", 0),
        "batches_failed": counts.get("failed", 0),
        "translated_ayahs": int(translated),
        "total_ayahs": int(total_ayahs),
    }
