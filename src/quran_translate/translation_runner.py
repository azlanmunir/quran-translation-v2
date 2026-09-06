"""Run Gemini translation batches and persist validated responses."""

from __future__ import annotations

import json
import sqlite3
import time

from .db import utc_now
from .gemini_client import GeminiGenerator, RetryConfig, TextGenerator, generate_with_retry
from .prompt_builder import build_batch_payload, build_prompt, parse_translation_response


def reset_running_batches(conn: sqlite3.Connection, run_id: str) -> int:
    now = utc_now()
    with conn:
        cursor = conn.execute(
            """
            UPDATE translation_batches
            SET status = 'pending',
                last_error = 'Reset from running state before resume',
                updated_at = ?
            WHERE run_id = ? AND status = 'running'
            """,
            (now, run_id),
        )
    return cursor.rowcount


def _batch_query(retry_failed: bool) -> str:
    statuses = "('pending', 'failed')" if retry_failed else "('pending')"
    return f"""
        SELECT *
        FROM translation_batches
        WHERE run_id = ?
          AND status IN {statuses}
        ORDER BY
          CASE status WHEN 'pending' THEN 0 ELSE 1 END,
          batch_index
    """


def pending_batches(
    conn: sqlite3.Connection,
    run_id: str,
    retry_failed: bool = False,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    rows = list(conn.execute(_batch_query(retry_failed), (run_id,)))
    if limit is not None:
        return rows[:limit]
    return rows


def save_batch_success(
    conn: sqlite3.Connection,
    batch: sqlite3.Row,
    raw_response: str,
    parsed: list[dict],
    added_attempts: int,
) -> None:
    now = utc_now()
    refs = [item["ref"] for item in parsed]
    placeholders = ",".join("?" for _ in refs)
    with conn:
        conn.execute(
            f"""
            DELETE FROM word_bank_entries
            WHERE run_id = ?
              AND verse_key IN ({placeholders})
            """,
            [batch["run_id"], *refs],
        )

        for item in parsed:
            raw_item = json.dumps(item, ensure_ascii=False, sort_keys=True)
            conn.execute(
                """
                INSERT INTO translations (
                    run_id,
                    verse_key,
                    translation,
                    status,
                    raw_translation_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, 'complete', ?, ?, ?)
                ON CONFLICT(run_id, verse_key) DO UPDATE SET
                    translation = excluded.translation,
                    status = excluded.status,
                    raw_translation_json = excluded.raw_translation_json,
                    updated_at = excluded.updated_at
                """,
                (
                    batch["run_id"],
                    item["ref"],
                    item["translation"],
                    raw_item,
                    now,
                    now,
                ),
            )

            for entry in item.get("word_bank", []):
                if not isinstance(entry, dict):
                    continue
                term = str(entry.get("term") or "").strip()
                root = str(entry.get("root") or "").strip() or None
                rendering = str(entry.get("rendering") or "").strip() or None
                physical_reality = (
                    str(
                        entry.get("physical_reality")
                        or entry.get("630_ce_physical_reality")
                        or entry.get("definition")
                        or ""
                    ).strip()
                    or None
                )
                definition = str(entry.get("definition") or physical_reality or "").strip()
                if not term or not definition:
                    continue
                conn.execute(
                    """
                    INSERT INTO word_bank_entries (
                        run_id,
                        verse_key,
                        term,
                        root,
                        rendering,
                        physical_reality,
                        definition,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch["run_id"],
                        item["ref"],
                        term,
                        root,
                        rendering,
                        physical_reality,
                        definition,
                        now,
                    ),
                )

        conn.execute(
            """
            UPDATE translation_batches
            SET status = 'complete',
                attempts = attempts + ?,
                last_error = NULL,
                raw_response = ?,
                updated_at = ?
            WHERE batch_id = ?
            """,
            (added_attempts, raw_response, now, batch["batch_id"]),
        )


def save_batch_failure(
    conn: sqlite3.Connection,
    batch: sqlite3.Row,
    error: BaseException | str,
    added_attempts: int,
    raw_response: str | None = None,
) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            """
            UPDATE translation_batches
            SET status = 'failed',
                attempts = attempts + ?,
                last_error = ?,
                raw_response = COALESCE(?, raw_response),
                updated_at = ?
            WHERE batch_id = ?
            """,
            (added_attempts, str(error), raw_response, now, batch["batch_id"]),
        )


def mark_batch_running(conn: sqlite3.Connection, batch_id: str) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            """
            UPDATE translation_batches
            SET status = 'running',
                updated_at = ?
            WHERE batch_id = ?
            """,
            (now, batch_id),
        )


def update_run_status(conn: sqlite3.Connection, run_id: str) -> None:
    counts = {
        row["status"]: int(row["count"])
        for row in conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM translation_batches
            WHERE run_id = ?
            GROUP BY status
            """,
            (run_id,),
        )
    }
    if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0 and counts.get("failed", 0) == 0:
        status = "complete"
    elif counts.get("failed", 0):
        status = "has_failures"
    else:
        status = "in_progress"

    with conn:
        conn.execute(
            "UPDATE translation_runs SET status = ?, updated_at = ? WHERE run_id = ?",
            (status, utc_now(), run_id),
        )


def translate_batches(
    conn: sqlite3.Connection,
    run_id: str,
    generator: TextGenerator | None = None,
    limit: int | None = None,
    retry_failed: bool = False,
    stop_on_error: bool = False,
    retry_config: RetryConfig | None = None,
) -> dict[str, int]:
    generator = generator or GeminiGenerator(
        model=conn.execute(
            "SELECT model FROM translation_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()["model"]
    )
    retry_config = retry_config or RetryConfig()
    reset_running_batches(conn, run_id)

    completed = 0
    failed = 0
    skipped = 0
    rows = pending_batches(conn, run_id, retry_failed=retry_failed, limit=limit)

    for batch in rows:
        target_refs = json.loads(batch["target_refs_json"])
        already_done = conn.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM translations
            WHERE run_id = ?
              AND status = 'complete'
              AND verse_key IN ({','.join('?' for _ in target_refs)})
            """,
            [run_id, *target_refs],
        ).fetchone()["count"]
        if int(already_done) == len(target_refs):
            with conn:
                conn.execute(
                    """
                    UPDATE translation_batches
                    SET status = 'complete', updated_at = ?
                    WHERE batch_id = ?
                    """,
                    (utc_now(), batch["batch_id"]),
                )
            skipped += 1
            continue

        print(
            f"{utc_now()} starting batch {batch['batch_index']} "
            f"({batch['start_ref']}..{batch['end_ref']}, {len(target_refs)} ayahs)",
            flush=True,
        )
        mark_batch_running(conn, batch["batch_id"])
        prompt = build_prompt(build_batch_payload(conn, batch))
        raw_response: str | None = None
        added_attempts = 0

        try:
            for parse_attempt in range(1, retry_config.max_attempts + 1):
                raw_response, sdk_attempts = generate_with_retry(generator, prompt, retry_config)
                added_attempts += sdk_attempts
                try:
                    parsed = parse_translation_response(raw_response, target_refs)
                    save_batch_success(conn, batch, raw_response, parsed, added_attempts)
                    print(
                        f"{utc_now()} completed batch {batch['batch_index']} "
                        f"({batch['start_ref']}..{batch['end_ref']})",
                        flush=True,
                    )
                    completed += 1
                    break
                except Exception as parse_error:  # noqa: BLE001 - parse/shape validation.
                    if parse_attempt >= retry_config.max_attempts:
                        raise parse_error
                    time.sleep(min(2 * parse_attempt, 10))
            else:
                raise RuntimeError("No response parsed")
        except Exception as exc:  # noqa: BLE001 - SDKs raise varied exception types.
            save_batch_failure(conn, batch, exc, added_attempts or 1, raw_response)
            print(
                f"{utc_now()} failed batch {batch['batch_index']} "
                f"({batch['start_ref']}..{batch['end_ref']}): {exc}",
                flush=True,
            )
            failed += 1
            if stop_on_error:
                update_run_status(conn, run_id)
                raise

    update_run_status(conn, run_id)
    return {"completed": completed, "failed": failed, "skipped": skipped}
