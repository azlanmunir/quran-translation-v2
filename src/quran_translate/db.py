"""SQLite schema and database helpers."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .config import DEFAULT_DB_PATH, ensure_dirs


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_files (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            imported_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS source_ayahs (
            verse_key TEXT PRIMARY KEY,
            global_ayah_number INTEGER NOT NULL UNIQUE,
            surah_number INTEGER NOT NULL,
            ayah_number INTEGER NOT NULL,
            surah_name_ar TEXT NOT NULL,
            surah_name_en TEXT NOT NULL,
            surah_meaning_en TEXT NOT NULL,
            arabic_uthmani_min TEXT NOT NULL,
            bismillah TEXT,
            UNIQUE (surah_number, ayah_number)
        );

        CREATE TABLE IF NOT EXISTS translation_runs (
            run_id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            batch_size INTEGER NOT NULL,
            max_target_chars INTEGER NOT NULL,
            context_before INTEGER NOT NULL,
            context_after INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS translation_batches (
            batch_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES translation_runs(run_id) ON DELETE CASCADE,
            batch_index INTEGER NOT NULL,
            surah_number INTEGER NOT NULL,
            start_ref TEXT NOT NULL,
            end_ref TEXT NOT NULL,
            target_refs_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            raw_response TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (run_id, batch_index)
        );

        CREATE TABLE IF NOT EXISTS translations (
            run_id TEXT NOT NULL REFERENCES translation_runs(run_id) ON DELETE CASCADE,
            verse_key TEXT NOT NULL REFERENCES source_ayahs(verse_key) ON DELETE CASCADE,
            translation TEXT NOT NULL,
            status TEXT NOT NULL,
            raw_translation_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, verse_key)
        );

        CREATE TABLE IF NOT EXISTS word_bank_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES translation_runs(run_id) ON DELETE CASCADE,
            verse_key TEXT NOT NULL REFERENCES source_ayahs(verse_key) ON DELETE CASCADE,
            term TEXT NOT NULL,
            root TEXT,
            rendering TEXT,
            physical_reality TEXT,
            definition TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS validation_issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            scope TEXT NOT NULL,
            ref TEXT,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS publication_translations (
            run_id TEXT NOT NULL REFERENCES translation_runs(run_id) ON DELETE CASCADE,
            verse_key TEXT NOT NULL REFERENCES source_ayahs(verse_key) ON DELETE CASCADE,
            source_translation TEXT NOT NULL,
            publication_translation TEXT NOT NULL,
            changed INTEGER NOT NULL,
            edits_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, verse_key)
        );

        CREATE TABLE IF NOT EXISTS audio_runs (
            audio_run_id TEXT PRIMARY KEY,
            translation_run_id TEXT NOT NULL REFERENCES translation_runs(run_id) ON DELETE CASCADE,
            voice_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            output_format TEXT NOT NULL,
            chunk_target_chars INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audio_chunks (
            audio_run_id TEXT NOT NULL REFERENCES audio_runs(audio_run_id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            surah_number INTEGER NOT NULL,
            start_ref TEXT NOT NULL,
            end_ref TEXT NOT NULL,
            text TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            output_path TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            duration_seconds REAL,
            bytes INTEGER,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (audio_run_id, chunk_id),
            UNIQUE (audio_run_id, chunk_index)
        );

        CREATE TABLE IF NOT EXISTS audio_outputs (
            audio_run_id TEXT NOT NULL REFERENCES audio_runs(audio_run_id) ON DELETE CASCADE,
            output_type TEXT NOT NULL,
            output_id TEXT NOT NULL,
            label TEXT NOT NULL,
            start_ref TEXT NOT NULL,
            end_ref TEXT NOT NULL,
            output_path TEXT NOT NULL,
            source_chunks_json TEXT NOT NULL,
            duration_seconds REAL,
            bytes INTEGER,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (audio_run_id, output_type, output_id)
        );

        CREATE INDEX IF NOT EXISTS idx_source_ayahs_surah
            ON source_ayahs(surah_number, ayah_number);

        CREATE INDEX IF NOT EXISTS idx_batches_run_status
            ON translation_batches(run_id, status, batch_index);

        CREATE INDEX IF NOT EXISTS idx_translations_run
            ON translations(run_id, verse_key);

        CREATE INDEX IF NOT EXISTS idx_word_bank_run_term
            ON word_bank_entries(run_id, term);

        CREATE INDEX IF NOT EXISTS idx_publication_translations_run
            ON publication_translations(run_id, verse_key);

        CREATE INDEX IF NOT EXISTS idx_audio_chunks_run_status
            ON audio_chunks(audio_run_id, status, chunk_index);

        CREATE INDEX IF NOT EXISTS idx_audio_outputs_run_type
            ON audio_outputs(audio_run_id, output_type, output_id);
        """
    )
    _migrate_word_bank_entries(conn)
    conn.commit()


def _migrate_word_bank_entries(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(word_bank_entries)")
    }
    if "root" not in columns:
        conn.execute("ALTER TABLE word_bank_entries ADD COLUMN root TEXT")
    if "physical_reality" not in columns:
        conn.execute("ALTER TABLE word_bank_entries ADD COLUMN physical_reality TEXT")


def latest_run_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT run_id FROM translation_runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return str(row["run_id"]) if row else None


def require_run_id(conn: sqlite3.Connection, run_id: str | None) -> str:
    resolved = run_id or latest_run_id(conn)
    if not resolved:
        raise SystemExit("No translation run exists yet. Run prepare-run first.")
    return resolved
