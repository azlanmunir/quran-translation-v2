# Quran Translation v2

A resumable English and Urdu Quran translation, audiobook, and video pipeline.

This repository is the maintained v2 codebase. The `v2` Git tag identifies
software snapshot `2.0.0`; English text release `v2.4.1` and Urdu text release
`v1.0.0` have independent, frozen provenance and are not renamed by this tag.
The [legacy public repository](https://github.com/azlanmunir/quran-translation)
is preserved separately.

## Scope and Quality

- Canonical Arabic source import, addressed English/Urdu translation, bounded
  review/recovery, publication exports, and release manifests.
- Narration production, 30 canonical Para and 114 Surah assemblies per language,
  synchronized captions, alignment, rendering, checksums, and decode gates.
- Short-form candidate selection, exact-verse episode specifications, final-audio
  semantic checks, and immutable publication receipts.

The daily series now uses [short-form strategy v2](docs/SHORT_FORM_DAILY_V2.md):
pass/fail editorial eligibility, sequential hook/verse timing, three story formats,
an audited buffer, and age-matched per-platform learning. Historical v1 episodes
and catalog scores are preserved, not retrospectively rewritten.

The repository contains source, policies, tests, and selected frozen decision
manifests. It does **not** contain API keys, browser sessions, production databases,
generated books/audio/video, or live publication receipts. A clone alone cannot
resume production: restore the matching original artifacts and verify their hashes
first. Social account sessions and daily scheduling live outside this repository;
the code does not by itself log in or schedule posts.

Passing software tests is not a scholarly certification of the translations or a
fresh audit of every published video. The September 2026 stricter Urdu alignment
audit flagged **202 of 343 historical records for review** (141 passed). These are
review flags, not proof of missing speech. Historical artifacts and the validated
unit-173 content repair remain preserved; no bulk timing fabrication or
regeneration is approved. See [v2 release notes](docs/V2_RELEASE_NOTES.md).

## Environment and Tests

Use Python 3.11 or newer. The complete media test stack is tested on Apple Silicon
macOS and uses MLX Whisper, FFmpeg/ffprobe, Rubber Band, and macOS fonts.
Urdu rendering also requires Pango and the configured Nastaliq font.
Cross-platform media rendering is not certified.
The optional Urdu PDF helper additionally needs Node.js, the `playwright`
package, and Google Chrome (or an explicit `CHROME_PATH`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-urdu-video.txt
python -m pytest -q
python -m ruff check src scripts --select F
```

Install FFmpeg and Rubber Band separately before running the media smoke tests.
The clean-checkout suite passes 250 tests and skips 25 production-asset integration
checks. After restoring the original matching releases/media locally, explicitly
run those checks with:

```bash
python -m pytest -q --production-assets
```

This option does not waive provenance validation: missing or changed assets fail.
Tests must not use paid provider calls. Translation/narration commands below can
spend credits; review their budgets and immutable run IDs before execution.
Do not use a new checkout to restart an existing paid or publishing job blindly.
For ambiguous submissions, follow [Submission Recovery](docs/SUBMISSION_RECOVERY.md).
CI runs lint and the default suite on a standard Apple Silicon macOS runner without
provider credentials or production media. It does not certify production releases.

Urdu video source artifacts default to this checkout. For an existing separate
source checkout, pass `--source-root` or set `QURAN_SOURCE_ROOT` in the process
environment before starting the command. Existing `RUN.json` contracts still
control resumed runs.

The design goal is simple: keep the Quran structure immutable and let models operate
only on already-addressed ayahs. Every generated translation is tied to a stable
verse key like `2:255`.

## Source

The canonical source file is:

```text
data/source/quran-uthmani-min.xml
```

It is Tanzil Quran Text, Uthmani Minimal, Version 1.1. Keep Tanzil attribution intact.

## Production v2.4 Pipeline

The current production path is `src/quran_translate/production_runner.py`:

1. Verify the pinned Tanzil XML, imported SQLite source, and Quranic Arabic Corpus
   morphology hashes; require 114 surahs and exactly 6,236 numbered ayahs.
2. Build 254 deterministic units, never crossing a surah boundary: at most 32 ayahs
   or 6,500 Arabic characters, with three neighboring ayahs of context.
3. Generate the reader draft with Claude Opus 4.6 through Anthropic Message Batches.
4. Audit every draft with the frozen Gemini 3.1 Pro forensic critic, then send only
   supported findings to an Opus constrained-revision pass.
5. Verify revised text, permit one bounded repair, and audit that repair once more.
6. Discover identical Arabic ayahs mechanically, resolve any English divergence,
   apply one rendering to every occurrence, and audit governed overrides.
7. Run the spoken-English/force checker as advisory review data. It never rewrites
   fidelity-reviewed text automatically.
8. Persist all 6,236 stable references, then run the whole-book QA gate. Only that
   gate may write `PRODUCTION_COMPLETE.json` and mark the run complete.

All provider jobs, stage inputs, responses, token usage, and validated artifacts are
atomically checkpointed under `data/work/production-v2.4/<run-id>/`. A restart polls
existing jobs and reuses valid unit artifacts. A contract retry contains only failed
units; completed text is never regenerated.

Active production policy:

```text
prompts/production-v2.4.md
prompts/sense-ledger-v2.4.md
data/evidence/sense-ledger-v2.4.json
```

Run the zero-spend preflight (it never constructs provider clients):

```bash
PYTHONPATH=src .venv/bin/python -m quran_translate.production_runner \
  --run-id production_v24_preflight --dry-run
```

The paid command is intentionally the same without `--dry-run`. Use a new immutable
run ID for the actual launch. See `PRODUCTION_V2_4.md` for the launch and recovery
contract.

## Basic Translation CLI

These commands exercise the original small-batch CLI. For the reviewed,
multi-stage production workflow use the Production v2.4 Pipeline above.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Import and validate the source:

```bash
PYTHONPATH=src python -m quran_translate.cli init-db
PYTHONPATH=src python -m quran_translate.cli import-source
PYTHONPATH=src python -m quran_translate.cli validate-source
```

Prepare a run:

```bash
PYTHONPATH=src python -m quran_translate.cli prepare-run
PYTHONPATH=src python -m quran_translate.cli status
```

Translate a small smoke-test batch:

```bash
PYTHONPATH=src python -m quran_translate.cli translate --limit 1
```

Export after translations exist:

```bash
PYTHONPATH=src python -m quran_translate.cli export
```

## ElevenLabs TTS Smoke Test

Add your ElevenLabs key and preferred voice id to `.env`:

```text
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
```

List available voices:

```bash
PYTHONPATH=src python -m quran_translate.cli elevenlabs-voices --limit 20
```

Create a quick MP3 from a completed translated ayah:

```bash
PYTHONPATH=src python -m quran_translate.cli elevenlabs-tts --run-id draft_v3 --ref 1:1
```

Or synthesize a custom line:

```bash
PYTHONPATH=src python -m quran_translate.cli elevenlabs-tts --text "The Owner of the System."
```

Audio files default to `output/audio/*.mp3`.

Package the completed full audio run for release without spending more ElevenLabs credits:

```bash
PYTHONPATH=src python -m quran_translate.cli audio-release --audio-run-id nathan_multilingual_v2 --force
```

That command validates the completed chunks, writes tagged release MP3s by surah, by 30 listening
parts, and as one complete-book MP3, copies the book PDFs and publication text, runs an ffmpeg
decode pass, and writes a release manifest plus `SHA256SUMS.txt` under
`output/release/quran-translation-v2`.

## v2.4.1 Narration Bakeoff

The production narration decision uses a release-pinned blind comparison rather than the legacy
`draft_v3` audio. The harness refuses to run unless the packaged listening edition reproduces the
approved v2.4.1 text hash. It compares six representative passages across Eleven v3, Eleven
Multilingual v2, Gemini 2.5 Pro TTS, and Gemini 3.1 Flash TTS while keeping provider identities out
of the listening package.

Prepare or inspect the resumable run:

```bash
PYTHONPATH=src .venv/bin/python -m quran_translate.cli audio-bakeoff-prepare
PYTHONPATH=src .venv/bin/python -m quran_translate.cli audio-bakeoff-status
```

Generate only missing clips, validate each response with ffprobe, level-match the review copies,
and build the blind listening page:

```bash
PYTHONPATH=src .venv/bin/python -m quran_translate.cli audio-bakeoff-run
```

Raw provider files remain under `output/audio/bakeoffs/<id>/raw`. The review package under
`output/audio/bakeoffs/<id>/blind` contains only randomized voice codes, checksummed 44.1 kHz
MP3s, the exact passage text, and a browser-local score exporter. The private key is created once,
permissioned `0600`, and reused on resume. Generated audio and the key remain outside Git.

The completed blind process selected Nathan on Eleven v3, lowered 1.25 semitones with Rubber
Band's R3 fine engine and formant preservation. The final comparison covered consolation,
eschatological oaths, and sustained narrative. No global time stretch is approved: the runner-up's
slower delivery varied materially by passage, while the selected voice won overall and especially
on the two more demanding samples. The machine-readable decision and its evidence hashes are in
`releases/quran-translation-v2.4.1-audio-voice.json`.

## v2.4.1 Production Audiobook

The production audio runner consumes only the frozen, QA-passed v2.4.1 listening edition. It
verifies the release manifest and all 6,236 references, reproduces the approved final text hash,
and writes an immutable 313-job manifest. Chunks never cross a surah or canonical juz boundary.
Canonical juz ranges are pinned in `data/evidence/juz-boundaries-v1.json` with Quran.Foundation
provenance.

Prepare or inspect the run without making a provider request:

```bash
PYTHONPATH=src .venv/bin/python -m quran_translate.cli audio-production-prepare
PYTHONPATH=src .venv/bin/python -m quran_translate.cli audio-production-status
```

The approved production source format is lossless 44.1 kHz PCM, which requires an ElevenLabs Pro
subscription. Every provider response is atomically checkpointed before Rubber Band R3 lowers it
1.25 semitones with formants preserved. The processed WAV master is then encoded once to a 192
kbps MP3 master. Completed chunks are reused on every resume.

```bash
PYTHONPATH=src .venv/bin/python -u -m quran_translate.cli \
  audio-production-synthesize --max-attempts 2 --request-timeout 600
```

After all chunks pass format, duration, and hash checks, one command produces the 114 surah files,
30 canonical juz files, and one full-book MP3. Approximately 40-minute tracks are optional and cut
only between master chunks; they never trigger more TTS generation.

```bash
PYTHONPATH=src .venv/bin/python -m quran_translate.cli \
  audio-production-assemble --decode-check
```

The frozen plan contains 750,096 billable characters. At the account rate recorded during the
preflight, a clean pass is about $150 on PAYG alone. One month of Pro plus a modest PAYG reserve is
both cheaper and higher quality because Pro includes 600,000 credits and unlocks lossless PCM.

## Bismillah Policy

In this Tanzil XML, Al-Fatihah includes Bismillah as ayah `1:1`. Other surahs store Bismillah as
an attribute on ayah 1. The translation pipeline treats non-Fatihah Bismillah as an opening marker,
not as a numbered target ayah.

The production prompt labels each non-Fatihah marker as unnumbered context and
explicitly forbids returning it as a target. Surah 9 correctly has no marker.

## Reading Notes

The listening edition remains note-free. Internal review flags and model suggestions
feed a separate evidence-adjudication queue. A note enters the reading edition only
after a human editor ties it to identified evidence; model-written philology is not
published as fact.
