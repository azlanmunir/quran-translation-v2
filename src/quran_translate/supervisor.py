"""Supervise a translation run one child process at a time.

The Gemini SDK can occasionally block inside a network call in a way that an
in-process alarm does not interrupt. This supervisor keeps the long run moving
by spawning a fresh child process for one batch, killing that child if it
exceeds a wall-clock timeout, resetting stale running batches, and continuing.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import DEFAULT_DB_PATH, PROJECT_ROOT
from .db import connect, init_db, utc_now


def counts(conn, run_id: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM translation_batches
        WHERE run_id = ?
        GROUP BY status
        """,
        (run_id,),
    )
    return {row["status"]: int(row["count"]) for row in rows}


def translated_count(conn, run_id: str) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM translations
            WHERE run_id = ? AND status = 'complete'
            """,
            (run_id,),
        ).fetchone()["count"]
    )


def reset_running(conn, run_id: str, reason: str, increment_attempts: bool = False) -> int:
    now = utc_now()
    with conn:
        cursor = conn.execute(
            f"""
            UPDATE translation_batches
            SET status = 'pending',
                attempts = attempts + ?,
                last_error = ?,
                updated_at = ?
            WHERE run_id = ? AND status = 'running'
            """,
            (1 if increment_attempts else 0, reason, now, run_id),
        )
    return int(cursor.rowcount)


def latest_failed_error(conn, run_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT last_error
        FROM translation_batches
        WHERE run_id = ?
          AND status = 'failed'
          AND last_error IS NOT NULL
        ORDER BY updated_at DESC, batch_index DESC
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return str(row["last_error"])


def is_quota_exhausted(error: str | None) -> bool:
    if not error:
        return False
    return "RESOURCE_EXHAUSTED" in error or "GenerateRequestsPerDayPerProjectPerModel" in error


def pause_for_quota(conn, run_id: str) -> int:
    now = utc_now()
    with conn:
        cursor = conn.execute(
            """
            UPDATE translation_batches
            SET status = 'quota_paused',
                last_error = COALESCE(
                    last_error,
                    'Paused by supervisor: Gemini daily quota exhausted'
                ),
                updated_at = ?
            WHERE run_id = ?
              AND status IN ('pending', 'failed', 'running')
            """,
            (now, run_id),
        )
        conn.execute(
            """
            UPDATE translation_runs
            SET status = 'quota_paused',
                updated_at = ?
            WHERE run_id = ?
            """,
            (now, run_id),
        )
    return int(cursor.rowcount)


def child_command(args: argparse.Namespace) -> list[str]:
    python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    if not Path(python).exists():
        python = sys.executable
    return [
        python,
        "-m",
        "quran_translate.cli",
        "--db",
        str(args.db),
        "translate",
        "--run-id",
        args.run_id,
        "--limit",
        "1",
        "--retry-failed",
        "--stop-on-error",
        "--max-attempts",
        str(args.child_max_attempts),
        "--request-timeout",
        str(args.request_timeout),
    ]


def run_child(args: argparse.Namespace, log_handle) -> int | None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    started = utc_now()
    print(f"{started} supervisor launching child", flush=True)
    print(f"{started} supervisor launching child", file=log_handle, flush=True)
    process = subprocess.Popen(
        child_command(args),
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + args.child_timeout
    while True:
        return_code = process.poll()
        if return_code is not None:
            ended = utc_now()
            print(f"{ended} supervisor child exited {return_code}", flush=True)
            print(f"{ended} supervisor child exited {return_code}", file=log_handle, flush=True)
            return int(return_code)

        if time.monotonic() >= deadline:
            ended = utc_now()
            print(
                f"{ended} supervisor killed child after {args.child_timeout}s",
                flush=True,
            )
            print(
                f"{ended} supervisor killed child after {args.child_timeout}s",
                file=log_handle,
                flush=True,
            )
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(
                    f"{utc_now()} supervisor could not reap killed child immediately",
                    flush=True,
                )
                print(
                    f"{utc_now()} supervisor could not reap killed child immediately",
                    file=log_handle,
                    flush=True,
                )
            return None

        time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))


def supervise(args: argparse.Namespace) -> int:
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with connect(Path(args.db)) as conn:
        init_db(conn)
        reset_running(conn, args.run_id, "Supervisor reset stale running batch before resume")

    with log_path.open("a", encoding="utf-8") as log_handle:
        print(f"{utc_now()} supervisor started for {args.run_id}", flush=True)
        print(f"{utc_now()} supervisor started for {args.run_id}", file=log_handle, flush=True)

        while True:
            with connect(Path(args.db)) as conn:
                current = counts(conn, args.run_id)
                done = current.get("complete", 0)
                pending = current.get("pending", 0)
                failed = current.get("failed", 0)
                running = current.get("running", 0)
                quota_paused = current.get("quota_paused", 0)
                ayahs = translated_count(conn, args.run_id)

            print(
                f"{utc_now()} supervisor status: complete={done} pending={pending} "
                f"running={running} failed={failed} quota_paused={quota_paused} ayahs={ayahs}",
                flush=True,
            )
            print(
                f"{utc_now()} supervisor status: complete={done} pending={pending} "
                f"running={running} failed={failed} quota_paused={quota_paused} ayahs={ayahs}",
                file=log_handle,
                flush=True,
            )

            if pending == 0 and running == 0 and failed == 0:
                if quota_paused:
                    print(f"{utc_now()} supervisor quota-paused", flush=True)
                    print(f"{utc_now()} supervisor quota-paused", file=log_handle, flush=True)
                    return 75
                print(f"{utc_now()} supervisor complete", flush=True)
                print(f"{utc_now()} supervisor complete", file=log_handle, flush=True)
                return 0

            return_code = run_child(args, log_handle)

            if return_code is None:
                with connect(Path(args.db)) as conn:
                    reset_running(
                        conn,
                        args.run_id,
                        f"Supervisor killed hung child after {args.child_timeout}s",
                        increment_attempts=True,
                    )
                time.sleep(args.sleep_after_timeout)
                continue

            if return_code != 0:
                with connect(Path(args.db)) as conn:
                    error = latest_failed_error(conn, args.run_id)
                    if is_quota_exhausted(error):
                        paused = pause_for_quota(conn, args.run_id)
                        message = (
                            f"{utc_now()} supervisor paused {paused} batches "
                            "because Gemini quota is exhausted"
                        )
                        print(message, flush=True)
                        print(message, file=log_handle, flush=True)
                        return 75
                # The child records parse/API failures in the DB. Keep going so a
                # later child can retry failed batches, but pause briefly to avoid
                # tight loops during provider trouble.
                time.sleep(args.sleep_after_error)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervise Quran translation batches")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--log", default=str(PROJECT_ROOT / "data" / "work" / "supervisor.log"))
    parser.add_argument("--child-timeout", type=int, default=420)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--child-max-attempts", type=int, default=2)
    parser.add_argument("--sleep-after-timeout", type=int, default=10)
    parser.add_argument("--sleep-after-error", type=int, default=20)
    return parser


def main() -> None:
    raise SystemExit(supervise(build_parser().parse_args()))


if __name__ == "__main__":
    main()
