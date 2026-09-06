# Daily Read That Again: Strategy v2

## Objective and boundaries

Help unfamiliar viewers watch, understand, share, and return. Raw views and
argument volume are not the objective. One English-first episode daily, adapted
for TikTok, Instagram and YouTube Shorts. Never infer individual religion from
profiles or comments. Keep the content suitable for teens without shame-based
hooks, clinical promises, sensationalized violence or pressure to disclose trauma.
The Quran and reference remain visible from the first frame. Translation,
context and editorial commentary must be distinguishable. Required synthetic
media disclosures remain truthful. No purchases, paid promotion or new providers.

Version 1 catalogs, scores, episodes and receipts are historical artifacts. Do
not overwrite them to adopt this policy. New renders use
`quran-short-form-episode-v2`. The original exact-text catalog remains the source
index, not an engagement ranking or publication approval. Extend it under a new
path with the existing canonical-release builder when additional passages are
needed; never mutate the catalog bound to a completed episode.

## Selection and planning

1. Name one recognizable situation and viewer need, such as forwarding a rumor,
   holding back an angry reply or treating someone fairly despite disagreement.
2. Find a complete contiguous passage and read the surrounding text. Two adjacent
   ayahs are only the starting context window. Review the entire relevant
   narrative, referents, qualifications and religious meaning where necessary.
3. Write the distinctive insight, accurate payoff and visual purpose. No generic
   self-help substitution, fabricated dialogue, or claim stronger than the verse.
4. Pass every eligibility check in `configs/short_form_strategy_v2.json`. Record
   reviewer, timestamp, actual context references read, risks and resolution,
   comprehension notes, canonical-quote SHA256 and catalog SHA256. Numeric scores
   cannot compensate for a failed check. A validator verifies the record, not
   theological truth; unresolved context is grounds to defer and select another
   passage, not to invent an approval. Routine editorial review is delegated to
   the operator under the owner's existing approval; no daily owner sign-off is
   required. Escalate genuinely unresolved interpretive questions.
5. Select a hypothesis with one principal variable, a variant and fixed controls.
   Balance situations and quote lengths across variants. Record unavoidable
   confounders. Organic publishing is not a randomized causal experiment.

The plan command creates a provisional 30-day discovery queue with three ten-day
blocks: opening, visual treatment, then format. It does not automatically approve
the candidates or select winners. Review the scheduled passage before assigning
it to an episode; swap unsuitable/high-context passages and record why in the
dated daily ledger. Missing candidates stay explicitly unfilled. Each seven-day
review can revise future slots in a new plan-revision artifact, preserving the
original. Do not mechanically force a short narrative into an unsuitable verse.

## Three formats

- **Everyday dilemma:** recognizable situation, complete verse, useful reflection.
- **Unexpected principle:** honest tension, complete verse, necessary nuance.
- **Short narrative:** factual setup, complete passage, source-faithful takeaway.

`brief` requires `format`, `situation`, `audience_need`, `insight`, `payoff`, and
`visual_purpose`. `experiment` requires `id`, `variable`, `variant`, `hypothesis`,
`fixed_controls`, and `causal_claim: false`. See the tests for a minimal structural
fixture; it is not a production-approved passage.

## Creative and media contract

Use a full-bleed relevant scene and one dominant text region. V2 removes the
decorative progress bar and duplicate floating panels. A single concise hook
appears before the quote, not simultaneously with it. `hook_end_seconds` equals
`quote_start_seconds`, from 0 to 1.8 seconds. For verse-first use an empty hook
and both times zero. A short narrative may add `creative.setup` after the hook;
its total pre-quote time may reach eight seconds when that context is necessary.
The setup is labeled CONTEXT and must pass normal-speed comprehension review.

Caption segment times remain relative to the extracted quote. The renderer shifts
both audio and burned-in captions by `quote_start_seconds`, and generates a
matching `captions.en.srt`. Never hand-shift only one. Final duration equals
intro plus the whole aligned quote plus a concise closing. Closing is at most
1.5 seconds and 20% of the video, and longer than the protected source post-roll.
Do not rush the quote to satisfy these commentary limits. Choose a longer complete
passage or another topic when necessary. A question is optional; use only one
short closing message if the normal-speed review shows that two cannot be read.

`visual.background_type` is `image` (restrained-still control) or `video`
(purposeful motion). A video background's audio is ignored; do not replace the
canonical narration. Motion must clarify the situation, not merely distract.
Never imply an illustrative scene is historical footage or an identifiable real
event. Use licensed/owned or properly disclosed generated assets. No new paid
asset service is authorized. Keep the local large-v3-turbo ASR setting and the
existing source-segment, checksums, post-roll, exact terminal words and decode
gates. A successful render is not creative or publication approval.

After rendering, inspect the full video at normal speed with sound and muted,
including its first frame, transitions, last spoken words and ending. Confirm
readability in mobile scale and platform safe areas, source visibility, no
competing text, and a comprehensible payoff. Check motion and asset provenance.
Write `CREATIVE_REVIEW.json` next to the immutable render with `status: passed`,
`master_sha256`, `reviewer`, `reviewed_at`, `notes`, and all three booleans
`visual_review`, `audio_review`, `comprehension_review` true only after inspection.
This separate receipt does not modify QA or render manifests. Run preflight
before opening any upload composer. Publication-state recording also enforces it.

The audited master exports to all three platforms. Titles and captions are native
per platform; any different video cut requires a separate render ID, checksums,
caption timing and creative QA, linked to the same intentional experiment.
Do not assume a platform-specific filename is a different creative treatment.

## Measurement without invented certainty

At each daily run, inspect native insights on all three accounts and collect due
24h, 72h and seven-day snapshots. Allowed matching tolerances are +/-6, +/-12 and
+/-24 hours. Missed windows remain missed, never backdated. When the actual
publication time is unknown, use `publication_time_basis: estimated`; that
observation remains usable descriptively but is excluded from matched cohorts.
`published_verified_at` is not necessarily the publication time.

Each observation identifies the episode, platform, post ID, observation and
publication timestamps, duration, organic/trial/paid distribution and evidence
(native insights URL plus captured screen/text artifact where practical).
For every known metric include `metric_definitions` with the native label, unit
and denominator. Unknown metrics are null with an `unavailable_reasons` entry.
Do not substitute public grid views for reach, or an absent likes count for zero.
Completion, initial-retention and chose-to-view rates use fractions from 0 to 1;
average percentage viewed uses native percentage units and may exceed 100%.
Record whether initial retention means one or three seconds. Keep metric
definitions consistent within a comparison; record definition changes as a new
experiment/cohort rather than merging them. YouTube AVD/APV may describe engaged
viewers, not all feed impressions.

The summary deduplicates repeated observations to the closest one per post/window
and separates platform, distribution, age, length band, experiment and variant.
Sharing, saving and following rates retain their views/reach denominator and
only use posts with both numerator and denominator known. It never names an
automatic winner. Five posts and 1,000 views per variant are review triggers,
not statistical significance or proof of an algorithm rule.

Weekly review: compare at matched ages and similar lengths, inspect actual clips
and feedback for misunderstanding, and record KEEP / REPEAT / REVISE / INCONCLUSIVE
with evidence and the next test. Repeat promising treatments across different
topics. Do not delete low-performing posts to improve the apparent sample.
Evaluate comprehension and sharing alongside retention, not controversy alone.
Missing analytics or low views never stop daily publishing. Keep the established
posting time as an operational anchor; test timing only after creative evidence
and audience geography justify it.

## Platform packaging

- TikTok: specific conversational opening; measure initial retention, watch time,
  completion, shares and follows. No watermark or invented trends.
- Instagram: readable cover, source-faithful insight, shares/saves and non-follower
  reach. Trial Reels are optional when the account supports them. Record `trial`
  separately and verify public distribution before treating the day's post as
  complete. A trial-only draft must not silently replace normal daily publication.
- YouTube: specific title, source, English captions, Education, not-made-for-kids,
  public Short. Attach the corresponding existing long-form translation via the
  related-video control when available and verified. Never alter the long-form
  videos or playlists. Keep display branding consistent without changing handles.

## Daily operations and three-episode buffer

1. Reconcile the local date ledger, ongoing processes, live drafts/posts and all
   platform receipts. Resume incomplete work; never upload because a receipt is
   missing without checking the platform. Preserve accepted IDs and upload intent.
2. Collect due insights and run status/summary. Reserve exactly one episode for
   the date in the existing daily ledger before production/upload.
3. Select a ready reviewed buffer episode that fits the experiment, or review and
   render a new one. Never reserve the same buffer item for two dates. Keep
   reservations and any substitutions in the dated ledgers.
4. Preflight, publish, verify public playback/captions and persist each platform's
   receipt immediately. A blocker on one platform need not block the others.
5. Replenish towards three fully reviewed, unpublished v2 episodes after today's
   obligation. Briefs and renders awaiting creative review are NOT ready buffer.
   Do not purchase assets or waive gates to fill it. Report the actual shortfall.
6. On weekly-review dates, save a dated decision note and revise future experiments
   only with supporting evidence. Report today's publication and blockers honestly.

All commands run in the active production checkout with its existing virtualenv:

```bash
.venv/bin/python scripts/short_form_daily.py status
.venv/bin/python scripts/short_form_daily.py plan --start 2026-09-06
.venv/bin/python scripts/short_form_daily.py observe /absolute/path/observation.json
.venv/bin/python scripts/short_form_daily.py summary
.venv/bin/python scripts/render_short_form_episode.py configs/episode-v2.json
.venv/bin/python scripts/short_form_daily.py preflight /absolute/path/episode-directory
```

`observe` is immutable/idempotent, `plan` preserves an existing cycle, and status
and summary are derived reports. Old analytics are preserved, not silently
converted into unavailable retention or exact-age data. Source artifacts and
live observations stay out of Git. Browser authentication and the existing daily
scheduler remain external dependencies; neither code nor a buffer guarantees
publication during an authentication, device or platform outage.
