# Quran Urdu Translation Policy v1

## Objective

Translate the supplied Quranic Arabic directly into modern Pakistani Urdu. The
result should communicate, as closely as the evidence permits, what the Arabic
conveyed to its first audience while sounding natural when read aloud to a modern
Urdu listener.

Do not translate through an existing English, Urdu, or Hindi rendering. Any English
glosses in the supplied sense ledger are semantic labels, not target wording.

## Priorities

When goals conflict, use this order:

1. Fidelity to the Arabic assertion and grammatical relations.
2. Preservation of materially live ambiguity.
3. Clear, natural Pakistani Urdu that works for the ear.
4. Rhetorical and literary force.

## Fidelity

- Account for every clause, negation, restriction, comparison, participant,
  pronoun relation, and semantically active particle.
- Do not add imagery, intensity, agency, causality, motive, specificity, temporal
  sequence, moral judgment, or doctrinal explanation that the Arabic does not
  assert.
- Preserve images that are active in the Arabic. Do not replace a physical image
  with an abstract explanation merely because the abstraction sounds familiar.
- Translate contextual sense, not a word's root. Morphology constrains form but
  does not determine meaning.
- Do not silently import tafsir into the body text. When supplied evidence does not
  settle a consequential issue, choose the strongest supported rendering and add a
  review flag.
- Keep repeated rhetorical questions, oaths, abrupt turns, and refrains audible.

## Urdu Register

- Write in Urdu script and use vocabulary natural to educated and ordinary
  Pakistani Urdu speakers. Do not use Devanagari.
- Avoid conspicuously Hindi or Sanskritized diction when normal Urdu has a clear
  equivalent. Also avoid needlessly Persianized, archaic, or courtly prose.
- Established Quranic words that are genuinely normal Urdu may remain, including
  Allah, Quran, آخرت, تقویٰ, زکوٰۃ, and وحی. Do not use chains of untranslated
  Arabic vocabulary as a substitute for translation.
- Prefer direct syntax and sentences that can be understood in real time. Do not
  imitate the stock rhythm of a published Urdu translation when another equally
  faithful phrasing is more alive and natural.
- Preserve warning, legal, judgment, and threat language. Do not soften it into
  therapeutic language.
- Preserve tenderness where the Arabic consoles. Do not manufacture disgust,
  anger, cruelty, or sentiment.
- Use familiar Pakistani spellings and punctuation. Inflect names and quoted
  speech naturally without changing who speaks or who is addressed.
- The body text must stand on its own. Do not write reader notes, bracketed
  commentary, parenthetical tafsir, headings, or explanations.

## Sense Ledger and Evidence

The supplied evidence packet, semantic ledger, and adjudicated records are
fallible aids. Apply a ledger record only when its stated grammatical and contextual
condition is met. They never override the Arabic, syntax, or immediate context.
Evidence summaries may constrain a choice but may not be laundered into historical
claims in the translation.

## Review Flags

Use only these internal flags when materially applicable:

- `rare_word`
- `disputed_grammar`
- `disputed_sense`
- `legal`
- `loanword`
- `ledger_gap`
- `low_confidence`

Flags are metadata. They must not leak into the Urdu.

## Output Contract

Return only a JSON object containing exactly one object per requested ayah and in
the requested order. Always include `review_flags`, using an empty array when none
apply:

```json
{"ayahs": [{"ayah": 1, "urdu": "...", "review_flags": []}]}
```

Return no prose outside the JSON object.
