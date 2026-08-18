# Arabic-to-Urdu Model Bakeoff v1

Frozen before generation on 2026-08-18.

## Scope

This experiment selects a model architecture for this Quran translation project.
It does not establish general superiority in translation or Arabic and Urdu
scholarship.

## Candidates

All candidates receive identical target passages, local Arabic context, QAC
morphology, mechanically selected intra-Quranic concordance, the v2.4 semantic
ledger, and applicable adjudicated semantic records.

1. Claude Fable 5
2. Claude Opus 4.8
3. Gemini 3.1 Pro Preview
4. Gemini 3.7 Flash at high thinking
5. GPT-5.6 Sol at high reasoning
6. GPT-5.6 Terra at high reasoning
7. Meta Muse Spark 1.2 at high reasoning

Opus 4.6 and Muse Spark 1.1 are excluded because a later checkpoint is available
at the same provider price tier. The treatment includes each model's operational
reliability under the common output contract.

## Corpus

Eleven contiguous passages contain exactly 90 unique ayahs:

- 1:1-7: opening prayer, divine titles, and unnamed agency in 1:7.
- 2:177-187: piety, legal obligation, fasting, taqwa, and live ambiguity.
- 4:11-12: inheritance law and disputed gender scope.
- 4:34-35: contested legal and relational vocabulary.
- 12:23-30: narrative dialogue, agency, desire, and reported speech.
- 19:1-11: prayer, bodily imagery, rare language, and human signaling.
- 24:2-5: zina law, testimony, scope, and community language.
- 55:1-16: compact creation language, dual address, and repeated refrain.
- 81:1-14: eschatological imagery, rare nouns, and the girl buried alive.
- 93:1-11: oaths and consolation without unsupported intensification.
- 112:1-4: theological compression and the al-samad crux.

## Generation Contract

- Translation is directly from Arabic. No complete English or Urdu verse
  translation is supplied to candidates.
- English ledger glosses are semantic labels, not target wording.
- Output is one Urdu string per ayah plus controlled internal review flags.
- Devanagari, missing ayahs, duplicate ayahs, blank Urdu, extra fields, or prose
  outside the JSON contract are treatment failures after two attempts.
- Each provider receives a 16,000-token output envelope and its documented high or
  adaptive reasoning setting.
- Every request, response, usage record, input hash, and validation result is
  written atomically. A manifest mismatch blocks resume.

## Blinding

A cryptographically shuffled, one-time candidate-to-code mapping is stored in a
permission-locked private key. The reading workbook, audit assignments, and audit
summary contain only blind codes. The mapping is not opened until Azlan's blinded
scores are exported.

## Audit

Gemini 3.7 Flash and GPT-5.6 Terra independently audit all candidates while blind.
They see the Arabic, evidence, and Urdu, but not model identities or the English
edition. A consensus defect is the same blind code, passage, ayah, and finding type
flagged by both auditors. Singleton findings remain visible for human scrutiny and
do not become zero merely because the second auditor missed them.

Defect weights are blocking 3, significant 1, and minor 0.25. Automated scoring is
triage, not scholarly adjudication.

## Decision Rule

1. A candidate with an unrecovered treatment failure cannot be selected for the
   full run.
2. A consensus blocking defect requires manual review before that candidate is
   eligible.
3. Fidelity evidence is considered before Urdu preference. Among candidates that
   survive fidelity review, Azlan's blinded naturalness, clarity, force, spoken
   quality, and overall preference select the winner.
4. Cost breaks a genuinely close quality result; it may not excuse a material
   fidelity defect.
5. The winner advances to a separate approximately 300-ayah pipeline validation
   before full-book production.

## Budget

The generation and two-auditor bakeoff is expected to remain within approximately
$15 at current API prices. No account purchase, subscription change, or automatic
top-up is authorized by this protocol.

## Preflight Amendment v1.1

Frozen before treatment generation on 2026-08-18. Provider smoke testing found
that OpenRouter requires the account holder to complete its adult-attestation
gate before it will serve Muse Spark 1.2. This is an account-access condition, not
a translation-contract result. Muse therefore remains pending without a forfeit
while the other candidates may run. It may rejoin before the blind workbook is
built; if access remains unavailable, its omission must be disclosed and no Muse
result may be inferred. The corpus, prompts, scoring, and decision rule are
unchanged.

## Measurement Amendment v1.2

Frozen after all 77 treatment translations completed, but before accepting any
audit result into the comparison. An audit pilot exposed three validator defects:
it treated equivalent Tanzil and standard Uthmani orthography as different Arabic
grounding, treated wrapping quotation marks as part of an Urdu quote, and rejected
minor findings paired with a `pass` verdict. These are measurement-contract
errors, not candidate failures.

The repaired measurement run imports the 77 completed translation artifacts
byte-for-byte, generates a fresh sealed blind mapping, and reruns both auditors
uniformly. Arabic grounding remains constrained to a contiguous source span after
documented orthographic normalization; Urdu grounding remains a contiguous raw
span after removing only wrapping quotation marks. Blocking or significant
findings require `revise`, and an ayah with no findings requires `pass`;
minor-only findings may carry either verdict because verdict labels are not
scored. The interrupted pilot audits remain preserved under the v1 run and are
excluded from all scoring. No candidate translation is regenerated, edited, or
unblinded by this amendment.
