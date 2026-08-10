# Quran Translation v2

A clean, resumable Quran translation and publication pipeline.

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

## Quick Start

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
