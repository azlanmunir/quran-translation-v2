# Quran Translation v2

A clean rebuild of the Quran historical-philological translation pipeline.

The design goal is simple: keep the Quran structure immutable and let Gemini only translate
already-addressed ayahs. Every generated translation is tied to a stable verse key like `2:255`.

## Source

The canonical source file is:

```text
data/source/quran-uthmani-min.xml
```

It is Tanzil Quran Text, Uthmani Minimal, Version 1.1. Keep Tanzil attribution intact.

## Pipeline

1. Import the Tanzil XML into SQLite.
2. Validate the source has 114 surahs and 6236 ayahs.
3. Create a translation run with deterministic ayah batches.
4. Send one batch at a time to Gemini with a strict JSON output contract.
5. Validate every model response before saving translations.
6. Export Markdown, JSON, bilingual Markdown, and a consolidated glossary.

Active prompt set:

```text
prompts/philological-v3.md
prompts/output-contract-v2.md
```

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

## Bismillah Policy

In this Tanzil XML, Al-Fatihah includes Bismillah as ayah `1:1`. Other surahs store Bismillah as
an attribute on ayah 1. The translation pipeline treats non-Fatihah Bismillah as an opening marker,
not as a numbered target ayah.

## Glossary Policy

Gemini returns a small per-ayah word bank. The exporter consolidates those entries into a global
glossary after the run.
