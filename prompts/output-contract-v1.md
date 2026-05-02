You will receive a JSON payload containing target ayahs and optional context ayahs.

Return ONLY valid JSON. Do not wrap it in Markdown fences. Do not add commentary before or after it.

Required response shape:

{
  "translations": [
    {
      "ref": "2:255",
      "translation": "Final English translation for this ayah only.",
      "word_bank": [
        {
          "term": "Arabic or transliterated term",
          "rendering": "Chosen English rendering",
          "definition": "Short historically anchored definition."
        }
      ]
    }
  ]
}

Rules:
- Return exactly one translation object for every target ayah ref.
- Preserve each target ref exactly.
- Do not return translations for context ayahs unless they are also targets.
- Do not merge ayahs.
- Do not split ayahs.
- Do not include ayah numbers inside the translation text.
- Do not include bracketed definitions, synonyms, or explanations inside the translation text.
- Keep glossary material only in word_bank.
- If a target ayah is disconnected letters, translate/transliterate only those letters plainly.
- For Al-Fatihah, Bismillah is target ayah 1:1 and should be translated.
- For other surahs, opening_bismillah is context only and should not be translated unless it appears as a target ayah.
