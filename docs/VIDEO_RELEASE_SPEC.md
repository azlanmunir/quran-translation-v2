# Quran YouTube Release Specification v1

## Product decision

The ayah is the canonical synchronization, caption, search, resume, and QA unit.
The visual presentation adapts without changing that data model:

- Two to four short ayahs share a stable panel; the spoken ayah is highlighted.
- A normal ayah receives its own panel.
- A long ayah advances by punctuation-aware clauses while retaining its ayah reference.
- Word karaoke is excluded.

One full-book timeline is aligned once. The 313 canonical narration chunks each belong
to exactly one Juz and one Surah. Each chunk is rendered once; the 30 Juz and 114 Surah
catalogs are lossless stream-copy concatenations of those same encoded segments. They
must never be independently aligned or rendered.

## Provenance gate

`NARRATION_MANIFEST.json` is the sole transcript authority for alignment. It binds:

- the approved semantic text hash;
- all 313 exact TTS input scripts, including spoken surah headings;
- every script's SHA-256 and character spans;
- the expected master-audio SHA-256 and duration for every chunk.

A renderer must refuse transcript, audio, span, or hash disagreement.

## Alignment gate

The pilot compares transcript-forced ElevenLabs timing with a local Whisper
word-timestamp pipeline on At-Takwir and 24:2–5. Both engines emit the same normalized
contract. The selected engine must be checked against audible boundaries, with special
attention to the largest inter-engine disagreements. Raw aligner output is retained.

The pilot selected ElevenLabs Forced Alignment. In At-Takwir, Whisper assigned only
0.28 seconds to 81:5 and absorbed the surrounding speech into adjacent ayahs, while
forced alignment preserved plausible, transcript-exact spans. On 24:2–5 both engines
were close, but forced alignment more consistently placed visual changes on spoken
starts after inter-ayah pauses. The full run must reuse cached raw results and may not
realign completed chunks.

## Video and audio

- 1920x1080, 30 fps, H.264 High Profile, `yuv420p`, MP4 fast-start.
- 44.1 kHz mono AAC at 192 kbps.
- Two-pass loudness processing targets -18 LUFS and -1.5 dBTP; QA rejects output above
  -1.0 dBTP or more than 0.3 LU from target.
- The selected Nathan v3 timbre and pacing are preserved; normalization is not a
  license to compress the narration into broadcast-loud speech.

## Captions and chapters

Timed English SRT is generated from the same display events used by the video.
Juz descriptions use surah-boundary chapters. Long Surah videos add ayah-range chapters
near ten-minute intervals. Chapter points snap to ayah starts and begin at `0:00`.

## Catalogs

- The 313 canonical segments are rendered first, in Juz order.
- 30 Juz videos are assembled from those segments and QA'd first.
- 114 Surah videos are then assembled from the identical segments, without re-encoding.
- Titles, descriptions, thumbnails, playlists, and disclosure language follow
  `configs/video_release_v1.json`.
- Synthetic narration and the AI-assisted, evidence-audited translation workflow are
  disclosed in every description and in YouTube's synthetic-content control.

## Pilot exit gate

The fleet may render only after both pilot passages pass:

- transcript/audio hashes and alignment coverage;
- audible boundary review of high-disagreement points;
- no clipped or unreadable 360-pixel frames;
- SRT/timeline identity;
- duration, decode, codec, pixel-format, sample-rate, LUFS, and true-peak checks;
- approved metadata and chapter output.

Shorts and a bilingual Arabic-English edition remain valuable follow-on products, not
release blockers for the 30-Juz and 114-Surah listening catalogs.
