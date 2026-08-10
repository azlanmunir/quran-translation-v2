"""Command line interface for the v2 translation pipeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .batching import prepare_run, run_status
from .audio_pipeline import (
    DEFAULT_AUDIO_RUN_ID,
    DEFAULT_CHUNK_TARGET_CHARS,
    DEFAULT_CONTEXT_CHARS,
    DEFAULT_PART_COUNT,
    DEFAULT_VOICE_ID,
    assemble_part_outputs,
    assemble_surah_outputs,
    audio_status,
    prepare_audio_chunks,
    synthesize_audio_chunks,
    write_audio_manifest,
)
from .audio_release import DEFAULT_RELEASE_ROOT, create_audio_release
from .audio_bakeoff import (
    DEFAULT_BAKEOFF_ROOT,
    BakeoffError,
    bakeoff_status,
    prepare_bakeoff,
    run_bakeoff,
)
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_AFTER,
    DEFAULT_CONTEXT_BEFORE,
    DEFAULT_DB_PATH,
    DEFAULT_MAX_TARGET_CHARS,
    DEFAULT_MODEL,
    DEFAULT_SOURCE_XML,
    OUTPUT_DIR,
    ensure_dirs,
)
from .db import connect, init_db, require_run_id
from .elevenlabs_tts import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_ELEVENLABS_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    ElevenLabsError,
    env_value,
    list_voices,
    synthesize_text,
)
from .exporter import export_all
from .gemini_client import RetryConfig
from .prompt_builder import build_batch_payload, build_prompt
from .book_pdf import (
    render_annotated_pdf,
    render_book_pdf,
    render_reader_pdf,
    write_pdf_inspection,
)
from .publication import (
    build_publication_layer,
    export_publication_all,
    validate_publication,
)
from .source_import import import_tanzil_xml
from .translation_runner import translate_batches
from .validation import persist_issues, validate_run, validate_source
from .release_hardening import (
    READING_NOTES_PATH,
    RELEASE_VERSION,
    apply_release_adjudications,
    create_release_package,
    run_final_quality_gate,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def open_db(args: argparse.Namespace):
    return connect(Path(args.db))


def cmd_init_db(args: argparse.Namespace) -> None:
    ensure_dirs()
    with open_db(args) as conn:
        init_db(conn)
    print(f"Initialized database: {args.db}")


def cmd_import_source(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        result = import_tanzil_xml(conn, Path(args.xml))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_validate_source(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        issues = validate_source(conn)
        persist_issues(conn, issues, scope_prefix="source")
    if issues:
        for issue in issues:
            ref = f" [{issue.ref}]" if issue.ref else ""
            print(f"{issue.severity.upper()} {issue.scope}{ref}: {issue.message}")
        raise SystemExit(1)
    print("Source validation passed: 114 surahs, 6236 ayahs.")


def cmd_prepare_run(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = prepare_run(
            conn,
            run_id=args.run_id,
            model=args.model,
            batch_size=args.batch_size,
            max_target_chars=args.max_target_chars,
            context_before=args.context_before,
            context_after=args.context_after,
        )
        status = run_status(conn, run_id)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        status = run_status(conn, run_id)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def cmd_show_batch(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        batch = conn.execute(
            """
            SELECT *
            FROM translation_batches
            WHERE run_id = ?
            ORDER BY batch_index
            LIMIT 1 OFFSET ?
            """,
            (run_id, args.index - 1),
        ).fetchone()
        if not batch:
            raise SystemExit(f"No batch #{args.index} in {run_id}")
        payload = build_batch_payload(conn, batch)
        if args.prompt:
            print(build_prompt(payload))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_translate(args: argparse.Namespace) -> None:
    retry_config = RetryConfig(
        max_attempts=args.max_attempts,
        request_timeout_seconds=args.request_timeout,
    )
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        result = translate_batches(
            conn,
            run_id,
            limit=args.limit,
            retry_failed=args.retry_failed,
            stop_on_error=args.stop_on_error,
            retry_config=retry_config,
        )
        status = run_status(conn, run_id)
    print(json.dumps({"result": result, "status": status}, ensure_ascii=False, indent=2))


def cmd_validate_run(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        issues = validate_run(conn, run_id)
        persist_issues(conn, issues, run_id=run_id)
    errors = [issue for issue in issues if issue.severity == "error"]
    if issues:
        max_print = args.max_print
        for issue in issues[:max_print]:
            ref = f" [{issue.ref}]" if issue.ref else ""
            print(f"{issue.severity.upper()} {issue.scope}{ref}: {issue.message}")
        remaining = len(issues) - max_print
        if remaining > 0:
            print(f"... {remaining} more issue(s)")
    if errors:
        raise SystemExit(1)
    print(f"Run validation passed: {run_id}")


def cmd_export(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        paths = export_all(conn, run_id, Path(args.output_dir))
    print(f"Exported {len(paths)} file(s) under {args.output_dir}")


def cmd_publication_build(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        stats = build_publication_layer(conn, run_id)
        paths = export_publication_all(conn, run_id, Path(args.output_dir))
        issues = validate_publication(conn, run_id)
    payload = {
        "stats": stats,
        "paths": [str(path) for path in paths],
        "issues": [
            {
                "scope": issue.scope,
                "severity": issue.severity,
                "ref": issue.ref,
                "message": issue.message,
            }
            for issue in issues
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise SystemExit(1)


def cmd_publication_validate(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        issues = validate_publication(conn, run_id)
    errors = [issue for issue in issues if issue.severity == "error"]
    if issues:
        for issue in issues[: args.max_print]:
            ref = f" [{issue.ref}]" if issue.ref else ""
            print(f"{issue.severity.upper()} {issue.scope}{ref}: {issue.message}")
        remaining = len(issues) - args.max_print
        if remaining > 0:
            print(f"... {remaining} more issue(s)")
    if errors:
        raise SystemExit(1)
    print(f"Publication validation passed: {run_id}")


def cmd_book_pdf(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        path = render_book_pdf(conn, run_id, Path(args.output))
        inspection_path = write_pdf_inspection(path)
    print(
        json.dumps(
            {
                "pdf": str(path),
                "inspection": str(inspection_path),
                "bytes": path.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_reader_pdf(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        path = render_reader_pdf(conn, run_id, Path(args.output))
        inspection_path = write_pdf_inspection(path)
    print(
        json.dumps(
            {
                "pdf": str(path),
                "inspection": str(inspection_path),
                "bytes": path.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_annotated_pdf(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        path = render_annotated_pdf(
            conn,
            run_id,
            Path(args.output),
            Path(args.notes),
        )
        inspection_path = write_pdf_inspection(path)
    print(
        json.dumps(
            {
                "pdf": str(path),
                "inspection": str(inspection_path),
                "bytes": path.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_release_harden(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        marker = apply_release_adjudications(conn, run_id)
        publication_stats = build_publication_layer(conn, run_id)
        translation_paths = export_all(conn, run_id, OUTPUT_DIR)
        publication_paths = export_publication_all(
            conn,
            run_id,
            OUTPUT_DIR / "publication",
        )
        pdfs = [
            render_book_pdf(
                conn,
                run_id,
                OUTPUT_DIR / "book" / "quran-translation-book.pdf",
            ),
            render_reader_pdf(
                conn,
                run_id,
                OUTPUT_DIR / "book" / "quran-translation-reader-edition.pdf",
            ),
            render_annotated_pdf(
                conn,
                run_id,
                OUTPUT_DIR / "book" / "quran-translation-annotated-reading-edition.pdf",
                Path(args.notes),
            ),
        ]
        inspections = [write_pdf_inspection(path) for path in pdfs]
        notes_payload = json.loads(Path(args.notes).read_text(encoding="utf-8"))
        notes_path = Path(args.notes)
        qa_report = run_final_quality_gate(
            conn,
            run_id,
            marker,
            notes_payload,
            notes_path,
        )
        release = create_release_package(
            run_id,
            qa_report,
            release_version=args.release_version,
            notes_path=notes_path,
        )
    print(
        json.dumps(
            {
                "adjudications": marker,
                "publication": publication_stats,
                "translation_exports": [str(path) for path in translation_paths],
                "publication_exports": [str(path) for path in publication_paths],
                "pdfs": [str(path) for path in pdfs],
                "inspections": [str(path) for path in inspections],
                "qa": qa_report,
                "release": release,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def safe_filename(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return slug or "elevenlabs-test"


def translation_for_ref(args: argparse.Namespace) -> str:
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        row = conn.execute(
            """
            SELECT translation
            FROM translations
            WHERE run_id = ?
              AND verse_key = ?
              AND status = 'complete'
            """,
            (run_id, args.ref),
        ).fetchone()
    if not row:
        raise SystemExit(f"No completed translation found for {args.ref} in {run_id}")
    return str(row["translation"])


def cmd_elevenlabs_voices(args: argparse.Namespace) -> None:
    try:
        voices = list_voices()
    except ElevenLabsError as exc:
        raise SystemExit(str(exc)) from exc
    payload = [
        {
            "name": voice.get("name"),
            "voice_id": voice.get("voice_id"),
            "category": voice.get("category"),
        }
        for voice in voices[: args.limit]
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_elevenlabs_tts(args: argparse.Namespace) -> None:
    text = args.text or translation_for_ref(args)
    ref_or_name = args.ref or "custom"
    output_path = Path(args.output) if args.output else (
        Path(args.output_dir) / f"{safe_filename(ref_or_name)}.mp3"
    )
    voice_id = args.voice_id or env_value("ELEVENLABS_VOICE_ID")
    model_id = args.model_id or env_value("ELEVENLABS_MODEL_ID") or DEFAULT_ELEVENLABS_MODEL
    output_format = args.output_format or env_value("ELEVENLABS_OUTPUT_FORMAT") or DEFAULT_OUTPUT_FORMAT
    try:
        path = synthesize_text(
            text=text,
            voice_id=voice_id or "",
            output_path=output_path,
            model_id=model_id,
            output_format=output_format,
            previous_text=args.previous_text,
            next_text=args.next_text,
        )
    except ElevenLabsError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output_path": str(path),
                "bytes": path.stat().st_size,
                "voice_id": voice_id,
                "model_id": model_id,
                "output_format": output_format,
                "text_preview": text[:160],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_audio_bakeoff_prepare(args: argparse.Namespace) -> None:
    try:
        state = prepare_bakeoff(Path(args.output))
    except BakeoffError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "bakeoff_id": state["bakeoff_id"],
                "status": state["status"],
                "jobs": len(state["jobs"]),
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_audio_bakeoff_run(args: argparse.Namespace) -> None:
    try:
        run_bakeoff(Path(args.output), max_attempts=args.max_attempts)
        status = bakeoff_status(Path(args.output))
    except BakeoffError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(status, ensure_ascii=False, indent=2))


def cmd_audio_bakeoff_status(args: argparse.Namespace) -> None:
    try:
        status = bakeoff_status(Path(args.output))
    except BakeoffError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(status, ensure_ascii=False, indent=2))


def audio_run_id_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio-run-id", default=DEFAULT_AUDIO_RUN_ID)


def cmd_audio_prepare(args: argparse.Namespace) -> None:
    voice_id = args.voice_id or env_value("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID
    model_id = args.model_id or env_value("ELEVENLABS_MODEL_ID") or DEFAULT_ELEVENLABS_MODEL
    output_format = args.output_format or env_value("ELEVENLABS_OUTPUT_FORMAT") or DEFAULT_OUTPUT_FORMAT
    with open_db(args) as conn:
        init_db(conn)
        run_id = require_run_id(conn, args.run_id)
        status = prepare_audio_chunks(
            conn,
            audio_run_id=args.audio_run_id,
            translation_run_id=run_id,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            chunk_target_chars=args.chunk_target_chars,
            force=args.force,
        )
    print(json.dumps(status, ensure_ascii=False, indent=2))


def cmd_audio_status(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        status = audio_status(conn, args.audio_run_id)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def cmd_audio_synthesize(args: argparse.Namespace) -> None:
    try:
        with open_db(args) as conn:
            init_db(conn)
            status = synthesize_audio_chunks(
                conn,
                audio_run_id=args.audio_run_id,
                retry_failed=args.retry_failed,
                limit=args.limit,
                max_attempts=args.max_attempts,
                context_chars=args.context_chars,
                request_timeout_seconds=args.request_timeout,
                sleep_seconds=args.sleep_seconds,
                stop_on_error=args.stop_on_error,
            )
    except ElevenLabsError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(status, ensure_ascii=False, indent=2))


def cmd_audio_assemble_surahs(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        result = assemble_surah_outputs(conn, args.audio_run_id, allow_partial=args.allow_partial)
        status = audio_status(conn, args.audio_run_id)
    print(json.dumps({"result": result, "status": status}, ensure_ascii=False, indent=2))


def cmd_audio_assemble_parts(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        result = assemble_part_outputs(conn, args.audio_run_id, part_count=args.part_count)
        status = audio_status(conn, args.audio_run_id)
    print(json.dumps({"result": result, "status": status}, ensure_ascii=False, indent=2))


def cmd_audio_manifest(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        path = write_audio_manifest(conn, args.audio_run_id)
    print(str(path))


def cmd_audio_release(args: argparse.Namespace) -> None:
    with open_db(args) as conn:
        init_db(conn)
        result = create_audio_release(
            conn,
            audio_run_id=args.audio_run_id,
            release_root=Path(args.release_root),
            force=args.force,
            decode_check=not args.skip_decode_check,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quran translation v2 pipeline")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init-db", help="Create database schema")
    init_cmd.set_defaults(func=cmd_init_db)

    import_cmd = sub.add_parser("import-source", help="Import Tanzil XML")
    import_cmd.add_argument("--xml", default=str(DEFAULT_SOURCE_XML), help="Path to Tanzil XML")
    import_cmd.set_defaults(func=cmd_import_source)

    validate_source_cmd = sub.add_parser("validate-source", help="Validate source corpus")
    validate_source_cmd.set_defaults(func=cmd_validate_source)

    prepare_cmd = sub.add_parser("prepare-run", help="Create a translation run and batches")
    prepare_cmd.add_argument("--run-id")
    prepare_cmd.add_argument("--model", default=DEFAULT_MODEL)
    prepare_cmd.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    prepare_cmd.add_argument("--max-target-chars", type=positive_int, default=DEFAULT_MAX_TARGET_CHARS)
    prepare_cmd.add_argument("--context-before", type=int, default=DEFAULT_CONTEXT_BEFORE)
    prepare_cmd.add_argument("--context-after", type=int, default=DEFAULT_CONTEXT_AFTER)
    prepare_cmd.set_defaults(func=cmd_prepare_run)

    status_cmd = sub.add_parser("status", help="Show latest or selected run status")
    status_cmd.add_argument("--run-id")
    status_cmd.set_defaults(func=cmd_status)

    show_batch_cmd = sub.add_parser("show-batch", help="Show a prepared batch payload or prompt")
    show_batch_cmd.add_argument("--run-id")
    show_batch_cmd.add_argument("--index", type=positive_int, default=1)
    show_batch_cmd.add_argument("--prompt", action="store_true")
    show_batch_cmd.set_defaults(func=cmd_show_batch)

    translate_cmd = sub.add_parser("translate", help="Translate pending batches with Gemini")
    translate_cmd.add_argument("--run-id")
    translate_cmd.add_argument("--limit", type=positive_int)
    translate_cmd.add_argument("--max-attempts", type=positive_int, default=8)
    translate_cmd.add_argument("--request-timeout", type=positive_int, default=180)
    translate_cmd.add_argument("--retry-failed", action="store_true")
    translate_cmd.add_argument("--stop-on-error", action="store_true")
    translate_cmd.set_defaults(func=cmd_translate)

    validate_run_cmd = sub.add_parser("validate-run", help="Validate a translation run")
    validate_run_cmd.add_argument("--run-id")
    validate_run_cmd.add_argument("--max-print", type=positive_int, default=80)
    validate_run_cmd.set_defaults(func=cmd_validate_run)

    export_cmd = sub.add_parser("export", help="Export current translation artifacts")
    export_cmd.add_argument("--run-id")
    export_cmd.add_argument("--output-dir", default=str(OUTPUT_DIR))
    export_cmd.set_defaults(func=cmd_export)

    publication_build_cmd = sub.add_parser("publication-build", help="Build audited publication-layer artifacts")
    publication_build_cmd.add_argument("--run-id")
    publication_build_cmd.add_argument("--output-dir", default=str(OUTPUT_DIR / "publication"))
    publication_build_cmd.set_defaults(func=cmd_publication_build)

    publication_validate_cmd = sub.add_parser("publication-validate", help="Validate publication-layer text")
    publication_validate_cmd.add_argument("--run-id")
    publication_validate_cmd.add_argument("--max-print", type=positive_int, default=80)
    publication_validate_cmd.set_defaults(func=cmd_publication_validate)

    book_pdf_cmd = sub.add_parser("book-pdf", help="Render a printable book PDF from the publication layer")
    book_pdf_cmd.add_argument("--run-id")
    book_pdf_cmd.add_argument("--output", default=str(OUTPUT_DIR / "book" / "quran-translation-book.pdf"))
    book_pdf_cmd.set_defaults(func=cmd_book_pdf)

    reader_pdf_cmd = sub.add_parser("reader-pdf", help="Render a paragraph-style reader PDF")
    reader_pdf_cmd.add_argument("--run-id")
    reader_pdf_cmd.add_argument("--output", default=str(OUTPUT_DIR / "book" / "quran-translation-reader-edition.pdf"))
    reader_pdf_cmd.set_defaults(func=cmd_reader_pdf)

    annotated_pdf_cmd = sub.add_parser(
        "annotated-pdf",
        help="Render the evidence-adjudicated annotated reading edition",
    )
    annotated_pdf_cmd.add_argument("--run-id")
    annotated_pdf_cmd.add_argument(
        "--notes",
        default=str(READING_NOTES_PATH),
    )
    annotated_pdf_cmd.add_argument(
        "--output",
        default=str(
            OUTPUT_DIR / "book" / "quran-translation-annotated-reading-edition.pdf"
        ),
    )
    annotated_pdf_cmd.set_defaults(func=cmd_annotated_pdf)

    release_harden_cmd = sub.add_parser(
        "release-harden",
        help="Apply tracked post-production adjudications and build a QA-gated release",
    )
    release_harden_cmd.add_argument("--run-id", required=True)
    release_harden_cmd.add_argument("--notes", default=str(READING_NOTES_PATH))
    release_harden_cmd.add_argument("--release-version", default=RELEASE_VERSION)
    release_harden_cmd.set_defaults(func=cmd_release_harden)

    voices_cmd = sub.add_parser("elevenlabs-voices", help="List ElevenLabs voices")
    voices_cmd.add_argument("--limit", type=positive_int, default=20)
    voices_cmd.set_defaults(func=cmd_elevenlabs_voices)

    tts_cmd = sub.add_parser("elevenlabs-tts", help="Create an ElevenLabs TTS MP3 smoke test")
    tts_cmd.add_argument("--run-id")
    text_source = tts_cmd.add_mutually_exclusive_group(required=True)
    text_source.add_argument("--text", help="Literal text to synthesize")
    text_source.add_argument("--ref", help="Completed verse ref to synthesize, e.g. 1:1")
    tts_cmd.add_argument("--voice-id", help="ElevenLabs voice_id; defaults to ELEVENLABS_VOICE_ID")
    tts_cmd.add_argument("--model-id", help=f"Defaults to {DEFAULT_ELEVENLABS_MODEL}")
    tts_cmd.add_argument("--output-format", help=f"Defaults to {DEFAULT_OUTPUT_FORMAT}")
    tts_cmd.add_argument("--output-dir", default=str(DEFAULT_AUDIO_DIR))
    tts_cmd.add_argument("--output", help="Exact output file path")
    tts_cmd.add_argument("--previous-text", help="Optional previous text for continuity")
    tts_cmd.add_argument("--next-text", help="Optional next text for continuity")
    tts_cmd.set_defaults(func=cmd_elevenlabs_tts)

    bakeoff_prepare_cmd = sub.add_parser(
        "audio-bakeoff-prepare",
        help="Prepare the release-pinned, multi-provider TTS bakeoff",
    )
    bakeoff_prepare_cmd.add_argument("--output", default=str(DEFAULT_BAKEOFF_ROOT))
    bakeoff_prepare_cmd.set_defaults(func=cmd_audio_bakeoff_prepare)

    bakeoff_run_cmd = sub.add_parser(
        "audio-bakeoff-run",
        help="Generate and blind all pending TTS bakeoff clips",
    )
    bakeoff_run_cmd.add_argument("--output", default=str(DEFAULT_BAKEOFF_ROOT))
    bakeoff_run_cmd.add_argument("--max-attempts", type=positive_int, default=2)
    bakeoff_run_cmd.set_defaults(func=cmd_audio_bakeoff_run)

    bakeoff_status_cmd = sub.add_parser(
        "audio-bakeoff-status",
        help="Show generation status for the TTS bakeoff",
    )
    bakeoff_status_cmd.add_argument("--output", default=str(DEFAULT_BAKEOFF_ROOT))
    bakeoff_status_cmd.set_defaults(func=cmd_audio_bakeoff_status)

    audio_prepare_cmd = sub.add_parser("audio-prepare", help="Prepare a resumable ElevenLabs audio chunk manifest")
    audio_prepare_cmd.add_argument("--run-id")
    audio_run_id_arg(audio_prepare_cmd)
    audio_prepare_cmd.add_argument("--voice-id")
    audio_prepare_cmd.add_argument("--model-id", default=DEFAULT_ELEVENLABS_MODEL)
    audio_prepare_cmd.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    audio_prepare_cmd.add_argument("--chunk-target-chars", type=positive_int, default=DEFAULT_CHUNK_TARGET_CHARS)
    audio_prepare_cmd.add_argument("--force", action="store_true")
    audio_prepare_cmd.set_defaults(func=cmd_audio_prepare)

    audio_status_cmd = sub.add_parser("audio-status", help="Show audio run status")
    audio_run_id_arg(audio_status_cmd)
    audio_status_cmd.set_defaults(func=cmd_audio_status)

    audio_synth_cmd = sub.add_parser("audio-synthesize", help="Generate pending ElevenLabs chunks")
    audio_run_id_arg(audio_synth_cmd)
    audio_synth_cmd.add_argument("--limit", type=positive_int)
    audio_synth_cmd.add_argument("--retry-failed", action="store_true")
    audio_synth_cmd.add_argument("--max-attempts", type=positive_int, default=3)
    audio_synth_cmd.add_argument("--context-chars", type=positive_int, default=DEFAULT_CONTEXT_CHARS)
    audio_synth_cmd.add_argument("--request-timeout", type=positive_int, default=240)
    audio_synth_cmd.add_argument("--sleep-seconds", type=float, default=0.0)
    audio_synth_cmd.add_argument("--stop-on-error", action="store_true")
    audio_synth_cmd.set_defaults(func=cmd_audio_synthesize)

    audio_surahs_cmd = sub.add_parser("audio-assemble-surahs", help="Assemble completed chunks into 114 surah MP3s")
    audio_run_id_arg(audio_surahs_cmd)
    audio_surahs_cmd.add_argument("--allow-partial", action="store_true")
    audio_surahs_cmd.set_defaults(func=cmd_audio_assemble_surahs)

    audio_parts_cmd = sub.add_parser("audio-assemble-parts", help="Assemble completed chunks into equal listening parts")
    audio_run_id_arg(audio_parts_cmd)
    audio_parts_cmd.add_argument("--part-count", type=positive_int, default=DEFAULT_PART_COUNT)
    audio_parts_cmd.set_defaults(func=cmd_audio_assemble_parts)

    audio_manifest_cmd = sub.add_parser("audio-manifest", help="Write the audio manifest JSON")
    audio_run_id_arg(audio_manifest_cmd)
    audio_manifest_cmd.set_defaults(func=cmd_audio_manifest)

    audio_release_cmd = sub.add_parser("audio-release", help="QA and package completed audio artifacts")
    audio_run_id_arg(audio_release_cmd)
    audio_release_cmd.add_argument("--release-root", default=str(DEFAULT_RELEASE_ROOT))
    audio_release_cmd.add_argument("--force", action="store_true", help="Rebuild tagged release MP3s")
    audio_release_cmd.add_argument(
        "--skip-decode-check",
        action="store_true",
        help="Skip the full ffmpeg decode pass over release MP3s",
    )
    audio_release_cmd.set_defaults(func=cmd_audio_release)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
