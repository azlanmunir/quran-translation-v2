# Urdu Quran Production v1

## Decision

The blinded configuration probe selected Muse Spark 1.2 as the primary Urdu
translator. Medium-effort Opus 4.6 improved cadence but introduced a material
legal-sense regression in 2:187. Medium-effort Opus 4.6 direct translation lost
terminology discipline and ranked last. The production architecture is therefore:

1. Muse Spark 1.2 drafts directly from Arabic with the evidence packet and ledger.
2. A cross-model critic audits every draft against Arabic and evidence.
3. Medium-effort Opus 4.6 proposes constrained revisions only for units with a
   significant fidelity or intelligibility finding.
4. An independent verifier accepts or rejects every changed ayah. Rejection returns
   that ayah to the Muse draft and keeps the unresolved finding in the review queue.
5. Deterministic terminology, script, repetition, and completeness gates run over
   the assembled text.

There is no universal style pass. Fluency is not allowed to rewrite an ayah that has
not been independently identified as needing repair.

## Launch gates

The full run must refuse to start unless all of these are true:

- The canonical Tanzil source and QAC morphology hashes match the English v2.4.1
  production inputs.
- The Urdu critic model has passed the frozen critic regression benchmark. The
  benchmark includes real defects exposed by the blind bakeoffs and clean controls,
  and runs under the same policy and ledgers used in production. Approval is tied
  to the exact result artifact and current benchmark instructions.
- Benchmark v2 includes the transparent taxonomy-only amendment recorded in
  `URDU_CRITIC_BENCHMARK_AMENDMENT_V2.md`; the original v1 pilot is preserved.
- A pricing snapshot is frozen in the run manifest.
- The configured hard cost ceiling is at most USD 100. A provider request must not
  be submitted when completed spend plus its conservative reservation would cross
  that ceiling.
- Every stage input, prompt, schema, model, source file, unit boundary, and runner
  source is hash-pinned in the run manifest.

## Resume and failure policy

- Validated artifacts are immutable and reused on resume.
- An existing artifact with a mismatched input hash aborts the run.
- Provider transport failures are retryable; contract-invalid responses receive one
  bounded retry and are then recorded as failed.
- A failed draft or critic blocks the run. A failed optional Opus revision does not
  replace the Muse draft, but its unresolved critic finding blocks publication.
- No stage deletes or regenerates validated work.
- `--limit` is diagnostic and may be used only with one stage at a time; the
  all-stages `run` command refuses it.
- `PRODUCTION_COMPLETE.json` is written only after all 6,236 ayahs, repeated-ayah
  consistency, deterministic gates, and unresolved-finding checks pass.
- `REVIEW_QUEUE.json` preserves every translator flag and critic finding, including
  non-blocking advisories and the disposition of every severe finding.

## Commands

Preparation is local and costs nothing:

```bash
PYTHONPATH=src .venv/bin/python -m quran_translate.urdu_critic_benchmark prepare
```

Each benchmark candidate is run explicitly, then exactly one passing candidate is
approved. These are provider calls and therefore spend money:

```bash
PYTHONPATH=src .venv/bin/python -m quran_translate.urdu_critic_benchmark run --candidate auditor-gemini-37
PYTHONPATH=src .venv/bin/python -m quran_translate.urdu_critic_benchmark run --candidate auditor-gpt-56-terra
PYTHONPATH=src .venv/bin/python -m quran_translate.urdu_critic_benchmark approve --candidate CANDIDATE_ID
```

The production runner uses one immutable run ID and resumes validated artifacts:

```bash
PYTHONPATH=src .venv/bin/python -m quran_translate.urdu_production prepare --run-id RUN_ID --hard-cost-ceiling-usd 100
PYTHONPATH=src .venv/bin/python -u -m quran_translate.urdu_production run --run-id RUN_ID --hard-cost-ceiling-usd 100
PYTHONPATH=src .venv/bin/python -m quran_translate.urdu_production status --run-id RUN_ID --hard-cost-ceiling-usd 100
```

## Cost policy

Muse drafts are expected to cost about USD 25 at the measured bakeoff rate. Opus is
selective; a full-book Opus edit pass is explicitly prohibited. The engineering
budget is USD 100, with a planning range of USD 60-90 after critic and selective
revision. The ceiling is enforcement, not a target.

## Scope of the result

This pipeline produces a modern Pakistani Urdu rendering under this project's
fidelity contract. It does not establish that one model is generally superior at
Arabic, Urdu, or scripture translation.
