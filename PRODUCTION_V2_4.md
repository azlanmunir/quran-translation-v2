# Production v2.4 Runbook

## Frozen Inputs

- Arabic: Tanzil Quran Text, Uthmani Minimal XML v1.1
- Arabic SHA256: `f78067cd98c51c03e450581e1e8713f4e7c352e0b62a4fe5c35811da28dd23bf`
- Morphology: Quranic Arabic Corpus data
- Morphology SHA256: `742bfac59941b2cb09736d5b7aae694af50792261fb8450cbf6afafcc340645f`
- Draft/revision model: `claude-opus-4-6`
- Critic model: `gemini-3.1-pro-preview`
- Prompt version: `production-v2.4-opus-gemini-transport-v1`

The manifest also hashes the prompt, sense ledger, refrain policy, batching code,
provider clients, critic, spoken-English checker, and QA gate. If any input changes,
resume refuses and requires a new run ID.

## Batching

- 254 deterministic units
- one surah per unit; no cross-surah boundary
- at most 32 target ayahs or 6,500 Arabic characters
- three neighboring ayahs on either side where available
- one bulk batch for a full 254-unit phase
- up to two contract attempts; attempt two contains only failed units
- 48,000 maximum Opus output/thinking tokens per unit; billing follows actual use
- grammar-constrained Anthropic JSON outputs for draft, revision, repair, and
  model-resolved refrains

The shared production prompt and ledger use Anthropic's one-hour cache breakpoint.
Every result is mapped by stable unit ID and must cover the exact expected ayah list.

## Stage Order

1. `draft`: Opus reader draft
2. `critic`: Gemini source-grounded fidelity audit
3. `revision`: Opus adjudicates every finding and makes minimal supported repairs
4. `verification`: Gemini checks revised units
5. `repair`: one bounded Opus repair where verification found defects
6. `final_verification`: Gemini audits repaired units; no unbounded rewrite loop
7. `refrains`: exact repeated Arabic is made English-invariant
8. `refrain_verification`: governed refrain overrides are source-audited
9. `spoken`: advisory clarity, force, oath, and stock-cadence review
10. persistence and whole-book QA

## Completion Gate

`PRODUCTION_COMPLETE.json` is written only when all of the following hold:

- the pinned source and morphology pass;
- all 6,236 stable references exist once and contain nonempty English;
- every required stage artifact passes its strict JSON contract;
- repaired and governed-refrain units have their final fidelity audits;
- identical Arabic ayahs have exactly identical English;
- no unresolved blocking/significant final fidelity finding remains;
- every unit has a spoken-English audit artifact;
- explicit forbidden archaic terms are absent.

Minor fidelity findings, spoken-English suggestions, and translator uncertainty flags
are preserved in `REVIEW_QUEUE.json`. They are not silently applied. A blocked run
keeps all paid artifacts, sets SQLite status `qa_blocked`, and writes
`QA_BLOCKED.json`; it does not regenerate completed work.

## Commands

Zero-spend preflight:

```bash
PYTHONPATH=src .venv/bin/python -m quran_translate.production_runner \
  --run-id production_v24_preflight --dry-run
```

Paid launch, only after explicit approval:

```bash
PYTHONPATH=src caffeinate -dimsu .venv/bin/python \
  -m quran_translate.production_runner --run-id production_v24_full
```

Status without submitting work:

```bash
PYTHONPATH=src .venv/bin/python -m quran_translate.production_runner \
  --run-id production_v24_full --status
```

The production process may be restarted with the exact paid command. Existing job
IDs and validated unit artifacts are reused. Never delete the run directory or use
the same run ID after changing a hashed input.

After a transport-only hardening change, a new run may explicitly seed compatible
drafts from an earlier run:

```bash
PYTHONPATH=src caffeinate -dimsu .venv/bin/python \
  -m quran_translate.production_runner \
  --run-id production_v24_1_full \
  --seed-drafts-from production_v24_full
```

Seeding verifies the source, morphology, prompt, ledgers, refrain policy, model, and
unit boundaries; contract-valid drafts are copied with source hashes into
`DRAFT_SEED.json`. Missing or invalid drafts are submitted normally. Later resumes
must use the same seed argument because it is frozen in the target manifest.
