# Quran Translation v2.4: Original-Audience Rendering

## The task

Translate the Quranic passage below into English. The target is **the best-supported
rendering of what these words conveyed to the people who first heard them** in the
Hijaz, 610–632 CE, carried into English that a modern listener can follow in real
time. Where the evidence does not settle what the first hearers understood, preserve
the uncertainty internally and flag it for review; do not manufacture certainty in
either direction.

## Evidence hierarchy

Weigh evidence for a word's sense in this order. Higher tiers carry presumptive
weight, not automatic precedence. A later source that demonstrably records early
evidence counts at the tier of the evidence it preserves.

1. The passage's grammar, syntax, and immediate context.
2. Intra-Quranic usage.
3. Securely early Arabic: contemporary poetry, inscriptions, documents, and the
   material life of the audience.
4. Religious vocabulary already current in the Hijaz, including relevant Aramaic,
   Syriac, Hebrew, and older Arabian usage.
5. Later lexicons, grammarians, reading traditions, and commentary as critically
   weighted secondary witnesses. A sense supported only by later doctrine, law, or
   semantic development may not be read back into the text.

## Sense before etymology

Translate what a word meant in use, not what its root once meant. A root is not a
definition. Preserve a root image only when the passage itself makes that image
active. Where a word was already a settled term, render its contextual sense.

## Register

The first hearers received this as immediate, forceful spoken Arabic. Match that.

- Use concrete, direct English and common words. Write for the ear.
- Keep the rhetorical shape: oaths, questions, repetitions, and sudden turns.
- The vocabulary must be understandable to a modern listener without a dictionary.
  Ancient force must come from the source's imagery, not archaic English.
- Never use *verily, lo, thus, lest, chastisement, recompense,* or *ingrate*.
- Avoid medieval political and devotional register such as *sovereign, dominion,
  decree, bounty,* and *grace* when plain contextual English is available.
- Avoid church-register terms whose English connotations come mainly from later
  Christian history when a plain word covers the Arabic. Never distort the Arabic
  merely to avoid an established religious term.
- A literal source-language calque that a modern listener cannot parse is not a
  faithful communication. Repair English grammar and idiom while preserving every
  semantic relation and every live ambiguity.
- Conventional words are welcome; conventional rhythm is not the target. When two
  equally faithful choices are available, avoid falling into the stock cadence of a
  major published translation. Preserve the source's own physical turn and pressure.
- Do not soften warning, threat, legal, or judgment language into a therapeutic
  register. In particular, select taqwā-family English by context: fear or guard
  language in active warnings and boundaries; mindful language may remain when the
  passage describes a person's settled character.
- Render exclamations by sense, not inherited liturgical formula.
- Never add what the Arabic does not assert: no new imagery, intensity, emotional
  color, agency, causality, motive, specificity, sequence, or moral judgment.
- Do not pad. The Arabic is compressed; keep the English tight.

Positive register targets, where the supplied Arabic supports them:

- "cool your eye" rather than flattening the physical idiom to "take comfort";
- "my head is ablaze with white hair" rather than explaining the image;
- "We can restore even his fingertips" rather than leaving a source-shaped fragment.

These are examples of physical force and speakable English, not phrases to imitate
where the Arabic does not contain the image.

When goals conflict, the priority order is **fidelity > preserved ambiguity >
natural spoken English > literary force**.

## Sense ledger

A contextual sense ledger is supplied. Only entries explicitly marked invariant may
be copied mechanically. Select every other sense by construction, syntax, immediate
context, and concordance. If no recorded sense fits, translate the passage rather
than forcing the ledger and add a review flag. Evidence summaries constrain the work
but are not reader notes.

## Uncertainty policy

Do **not** write reader-facing philological notes. A model may identify uncertainty;
it may not publish evidence. Choose the strongest supported contextual rendering and
add one or more internal `review_flags` when a material uncertainty remains. Those
flags route the ayah to a separate editorial process where any publication note must
be written from an identified, cited evidence record.

## Output

Return only a JSON array, one object per ayah:

```json
[{"ayah": 1, "english": "...", "review_flags": ["rare_word"]}]
```

Include `review_flags` only when applicable, using: `rare_word`,
`disputed_grammar`, `disputed_sense`, `legal`, `loanword`, `ledger_gap`, or
`low_confidence`. Flags are workflow metadata and must not leak into the English.
No `note` field and no text outside the JSON array.
