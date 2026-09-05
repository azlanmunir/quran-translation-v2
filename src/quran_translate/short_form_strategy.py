"""Build a context-preserving editorial catalog for short-form Quran videos."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import file_sha256
from .production_packets import atomic_json


SCORE_KEYS = (
    "universality",
    "clarity",
    "emotional_force",
    "daily_applicability",
    "counterintuitive_value",
    "visual_potential",
    "discussion_potential",
    "standalone_integrity",
)
REF_RE = re.compile(r"^(?P<surah>[1-9][0-9]{0,2}):(?P<ayah>[1-9][0-9]{0,2})$")


class ShortFormStrategyError(ValueError):
    """Raised when a candidate could misquote or misrepresent the release."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShortFormStrategyError(f"Could not read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShortFormStrategyError(f"Expected a JSON object in {path}")
    return value


def _parse_ref(ref: str) -> tuple[int, int]:
    match = REF_RE.fullmatch(ref)
    if not match:
        raise ShortFormStrategyError(f"Invalid ayah reference: {ref}")
    return int(match.group("surah")), int(match.group("ayah"))


def _score_candidate(scores: dict[str, Any], weights: dict[str, Any]) -> float:
    if set(scores) != set(SCORE_KEYS):
        missing = sorted(set(SCORE_KEYS) - set(scores))
        extra = sorted(set(scores) - set(SCORE_KEYS))
        raise ShortFormStrategyError(f"Score keys differ; missing={missing}, extra={extra}")
    if set(weights) != set(SCORE_KEYS):
        raise ShortFormStrategyError("Scoring weights must match the score contract")

    weight_total = sum(float(weights[key]) for key in SCORE_KEYS)
    if abs(weight_total - 100.0) > 0.001:
        raise ShortFormStrategyError(f"Scoring weights must total 100, found {weight_total}")

    weighted = 0.0
    for key in SCORE_KEYS:
        value = scores[key]
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            raise ShortFormStrategyError(f"Score {key} must be an integer from 1 to 5")
        weighted += value * float(weights[key])
    return round(weighted / 5.0, 1)


def build_short_form_catalog(
    *,
    release_path: Path,
    strategy_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Join hand-selected thought units to exact release text and nearby context."""

    release = _load_json(release_path)
    strategy = _load_json(strategy_path)
    ayahs = release.get("ayahs")
    if not isinstance(ayahs, list) or not ayahs:
        raise ShortFormStrategyError("Release has no ayah catalog")

    by_ref: dict[str, dict[str, Any]] = {}
    surah_limits: dict[int, int] = {}
    for row in ayahs:
        if not isinstance(row, dict) or not isinstance(row.get("ref"), str):
            raise ShortFormStrategyError("Release contains an invalid ayah row")
        ref = row["ref"]
        if ref in by_ref:
            raise ShortFormStrategyError(f"Duplicate release reference: {ref}")
        surah, ayah = _parse_ref(ref)
        if int(row.get("surah", -1)) != surah or int(row.get("ayah", -1)) != ayah:
            raise ShortFormStrategyError(f"Release reference fields disagree at {ref}")
        translation = row.get("translation")
        if not isinstance(translation, str) or not translation.strip():
            raise ShortFormStrategyError(f"Release translation is empty at {ref}")
        by_ref[ref] = row
        surah_limits[surah] = max(surah_limits.get(surah, 0), ayah)

    context_radius = strategy.get("context_ayahs_each_side", 2)
    if not isinstance(context_radius, int) or not 0 <= context_radius <= 10:
        raise ShortFormStrategyError("context_ayahs_each_side must be from 0 to 10")
    words_per_minute = strategy.get("narration_words_per_minute", 145)
    if not isinstance(words_per_minute, int) or not 80 <= words_per_minute <= 240:
        raise ShortFormStrategyError("narration_words_per_minute must be from 80 to 240")

    weights = strategy.get("scoring_weights")
    seeds = strategy.get("candidates")
    if not isinstance(weights, dict) or not isinstance(seeds, list) or not seeds:
        raise ShortFormStrategyError("Strategy requires scoring_weights and candidates")

    seen_ids: set[str] = set()
    catalog: list[dict[str, Any]] = []
    for seed in seeds:
        if not isinstance(seed, dict):
            raise ShortFormStrategyError("Every candidate must be an object")
        candidate_id = seed.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ShortFormStrategyError("Every candidate needs candidate_id")
        if candidate_id in seen_ids:
            raise ShortFormStrategyError(f"Duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)

        start_ref = str(seed.get("start_ref", ""))
        end_ref = str(seed.get("end_ref", ""))
        start_surah, start_ayah = _parse_ref(start_ref)
        end_surah, end_ayah = _parse_ref(end_ref)
        if start_surah != end_surah or end_ayah < start_ayah:
            raise ShortFormStrategyError(
                f"Candidate {candidate_id} must be a forward range within one Surah"
            )

        refs = [f"{start_surah}:{ayah}" for ayah in range(start_ayah, end_ayah + 1)]
        missing = [ref for ref in refs if ref not in by_ref]
        if missing:
            raise ShortFormStrategyError(f"Candidate {candidate_id} is missing refs {missing}")
        selected = [by_ref[ref] for ref in refs]
        thought_text = " ".join(str(row["translation"]).strip() for row in selected)
        word_count = len(re.findall(r"\b[\w'-]+\b", thought_text, flags=re.UNICODE))

        before_start = max(1, start_ayah - context_radius)
        after_end = min(surah_limits[start_surah], end_ayah + context_radius)
        before_refs = [f"{start_surah}:{ayah}" for ayah in range(before_start, start_ayah)]
        after_refs = [f"{start_surah}:{ayah}" for ayah in range(end_ayah + 1, after_end + 1)]

        scores = seed.get("scores")
        if not isinstance(scores, dict):
            raise ShortFormStrategyError(f"Candidate {candidate_id} has no score object")
        score_100 = _score_candidate(scores, weights)

        editorial = {
            key: value
            for key, value in seed.items()
            if key not in {"candidate_id", "start_ref", "end_ref", "scores"}
        }
        catalog.append(
            {
                "candidate_id": candidate_id,
                "start_ref": start_ref,
                "end_ref": end_ref,
                "refs": refs,
                "surah": start_surah,
                "surah_name": selected[0].get("surah_name_en"),
                "surah_meaning": selected[0].get("surah_meaning_en"),
                "exact_translation": thought_text,
                "word_count": word_count,
                "estimated_translation_seconds": round(
                    word_count * 60.0 / words_per_minute, 1
                ),
                "context_before": [by_ref[ref] for ref in before_refs],
                "context_after": [by_ref[ref] for ref in after_refs],
                "scores": scores,
                "editorial_score_100": score_100,
                "editorial": editorial,
                "production_status": "editorial_review",
                "urdu_mapping_status": "pending",
            }
        )

    catalog.sort(key=lambda item: (-item["editorial_score_100"], item["candidate_id"]))
    payload = {
        "version": "quran-short-form-candidate-catalog-v1",
        "strategy_name": strategy.get("strategy_name", "The Open Door"),
        "source": {
            "release_path": str(release_path),
            "release_sha256": file_sha256(release_path),
            "release_run_id": release.get("run_id"),
            "ayah_count": len(ayahs),
            "strategy_path": str(strategy_path),
            "strategy_sha256": file_sha256(strategy_path),
        },
        "rules": {
            "exact_release_text_only": True,
            "contiguous_single_surah_units_only": True,
            "context_ayahs_each_side": context_radius,
            "automated_publication_allowed": False,
        },
        "totals": {
            "candidates": len(catalog),
            "editorial_review": len(catalog),
            "approved_for_production": 0,
        },
        "candidates": catalog,
    }
    atomic_json(output_path, payload)
    return payload
