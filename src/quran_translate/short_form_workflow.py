"""Editorial eligibility and honest, age-matched short-form learning records.

Human/contextual judgments are explicit attestations, not claims that a numeric
score or this validator can establish a passage's meaning.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import file_sha256
from .production_packets import atomic_json
from .state_safety import exclusive_lock

VERSION = "quran-short-form-episode-v2"
FORMATS = {"everyday_dilemma", "unexpected_principle", "short_narrative"}
CHECKS = {
    "complete_thought",
    "context_preserved",
    "religious_meaning_preserved",
    "accessible_without_prior_knowledge",
    "claims_supported",
    "teen_safe",
    "risk_flags_resolved",
}
PLATFORMS = {"instagram", "tiktok", "youtube"}
METRICS = {
    "views",
    "reach",
    "non_follower_reach",
    "engaged_views",
    "likes",
    "comments",
    "shares",
    "saves",
    "profile_visits",
    "follows",
    "average_watch_seconds",
    "completion_rate",
    "initial_retention",
    "chose_to_view_rate",
    "average_percentage_viewed",
    "rewatches",
}
RATE_METRICS = {"completion_rate", "initial_retention", "chose_to_view_rate"}
WINDOWS = ((24, 6), (72, 12), (168, 24))


class WorkflowError(ValueError):
    """Reject incomplete evidence rather than invent an approval or a result."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WorkflowError(f"Expected object: {path}")
    return value


def stamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise WorkflowError("Timestamps require an explicit timezone")
    return parsed.astimezone(timezone.utc)


def require_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"Missing {label}")


def finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise WorkflowError(f"{label} must be finite numeric data")
    return float(value)


def validate_editorial(spec: dict, candidate: dict, catalog_sha256: str) -> dict:
    if spec.get("visual", {}).get("background_type", "image") not in {"image", "video"}:
        raise WorkflowError("Background must be an image or video")
    review = spec.get("editorial_review", {})
    if (
        review.get("status") != "passed"
        or review.get("catalog_sha256") != catalog_sha256
    ):
        raise WorkflowError(
            "Editorial review must pass and bind the current catalog checksum"
        )
    quote_hash = hashlib.sha256(
        spec["source"]["exact_translation"].encode()
    ).hexdigest()
    if review.get("quote_sha256") != quote_hash:
        raise WorkflowError("Editorial review does not bind the exact quote")
    for key in (
        "reviewer",
        "reviewed_at",
        "context_explanation",
        "distinctive_insight",
        "religious_meaning",
        "risk_resolution",
        "comprehension_notes",
    ):
        require_text(review.get(key), key)
    stamp(review["reviewed_at"])
    if set(review.get("checks", {})) != CHECKS or any(
        review["checks"][key] is not True for key in CHECKS
    ):
        raise WorkflowError(
            "All editorial eligibility checks must pass; scores cannot override them"
        )
    refs = set(review.get("context_refs_reviewed", []))
    if not set(candidate["refs"]).issubset(refs):
        raise WorkflowError("Context review must include the entire passage")
    neighbors = {
        row["ref"]
        for key in ("context_before", "context_after")
        for row in candidate.get(key, [])
    }
    if not neighbors.issubset(refs):
        raise WorkflowError(
            "Review adjacent context and expand it when the narrative requires"
        )
    risks = set(candidate.get("editorial", {}).get("risk_flags", []))
    if not risks.issubset(set(review.get("risk_flags_reviewed", []))):
        raise WorkflowError("Unreviewed candidate risk flags")
    brief = spec.get("brief", {})
    if brief.get("format") not in FORMATS:
        raise WorkflowError("Unknown editorial format")
    for key in ("situation", "audience_need", "insight", "payoff", "visual_purpose"):
        require_text(brief.get(key), key)
    experiment = spec.get("experiment", {})
    for key in ("id", "variable", "variant", "hypothesis", "fixed_controls"):
        require_text(experiment.get(key), f"experiment.{key}")
    if experiment.get("causal_claim") is not False:
        raise WorkflowError("Organic experiments must not claim causal proof")
    for platform in PLATFORMS:
        require_text(spec.get("platform_copy", {}).get(platform), f"{platform} copy")
    require_text(spec["platform_copy"].get("youtube_title"), "YouTube title")
    return review


def validate_timing(spec: dict) -> dict:
    """V2 commentary never competes with verse captions or clips their audio."""
    source, creative, render = spec["source"], spec["creative"], spec["render"]
    quote = finite(source["clip_end_seconds"], "clip end") - finite(
        source["clip_start_seconds"], "clip start"
    )
    offset = finite(creative.get("quote_start_seconds", 0), "quote offset")
    hook_end = finite(creative.get("hook_end_seconds", 0), "hook end")
    duration = finite(render["duration_seconds"], "duration")
    post_roll = finite(source.get("post_roll_seconds", 0), "post roll")
    pre_roll = finite(source.get("pre_roll_seconds", 0), "pre roll")
    if not 0 <= pre_roll <= 0.5:
        raise WorkflowError("Source pre-roll must be between zero and 0.5 seconds")
    quote += pre_roll
    if quote <= 0 or duration <= 0 or post_roll < 0.2:
        raise WorkflowError("Invalid quote duration or protected post-roll")
    setup = creative.get("setup")
    if not 0 <= hook_end <= 1.8 or offset < hook_end or offset > (8 if setup else 1.8):
        raise WorkflowError(
            "Hook must finish before narration; hook max 1.8s, narrative setup max 8s total"
        )
    if creative.get("hook") and hook_end <= 0:
        raise WorkflowError("Visible hook requires a positive display window")
    if not creative.get("hook") and hook_end > 0:
        raise WorkflowError("An empty hook must not create a blank lead-in")
    if setup and (
        spec.get("brief", {}).get("format") != "short_narrative" or offset <= hook_end
    ):
        raise WorkflowError(
            "Separate context setup requires a short narrative and its own reading window"
        )
    if offset and not setup and (not creative.get("hook") or hook_end != offset):
        raise WorkflowError("No blank lead-in between hook and verse")
    closing = duration - offset - quote
    if not post_roll < closing <= min(1.5, duration * 0.2) + 0.001:
        raise WorkflowError(
            "Closing must protect post-roll and stay within 1.5s / 20% of video"
        )
    if (render.get("width"), render.get("height")) != (1080, 1920):
        raise WorkflowError("V2 uses the verified 1080x1920 safe-area layout")
    return {
        "quote_start_seconds": offset,
        "quote_end_seconds": offset + quote,
        "closing_seconds": round(closing, 3),
        "one_reading_task": True,
    }


def validate_ready(folder: Path) -> dict:
    """Validate the exact asset plus explicit post-render creative review."""
    spec = read_json(folder / "EPISODE_SPEC.json")
    if spec.get("version") != VERSION:
        raise WorkflowError("Ready buffer must use strategy v2")
    marker = read_json(folder / "RENDER_COMPLETE.json")
    if not marker.get("artifacts"):
        raise WorkflowError("Missing immutable artifact manifest")
    for relative, expected in marker["artifacts"].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()) or file_sha256(path) != expected:
            raise WorkflowError("Buffer artifact checksum mismatch")
    review = read_json(folder / "CREATIVE_REVIEW.json")
    qa = read_json(folder / "QA.json")
    master = qa["artifacts"]["master"]
    if (
        review.get("master_sha256") != master["sha256"]
        or review.get("status") != "passed"
        or any(
            review.get(k) is not True
            for k in ("visual_review", "audio_review", "comprehension_review")
        )
    ):
        raise WorkflowError(
            "Buffer requires bound visual, audio and comprehension review"
        )
    for key in ("reviewer", "reviewed_at", "notes"):
        require_text(review.get(key), key)
    stamp(review["reviewed_at"])
    if (
        qa.get("video", {}).get("decode_passed") is not True
        or qa.get("spoken_audio", {}).get("passed") is not True
        or qa.get("encoded_spoken_audio", {}).get("passed") is not True
    ):
        raise WorkflowError("Technical and spoken-audio QA must pass")
    if file_sha256(Path(master["path"])) != master["sha256"]:
        raise WorkflowError("Master checksum differs")
    return {
        "episode_id": spec["episode_id"],
        "master_sha256": master["sha256"],
        "ready": True,
    }


def build_plan(catalog: dict, policy: dict, used_refs: set[str], start: str) -> dict:
    """Create a dated experiment queue, not fabricated editorial approvals."""
    first = datetime.strptime(start, "%Y-%m-%d").date()
    available = sorted(
        (
            row
            for row in catalog["candidates"]
            if not used_refs.intersection(row["refs"])
        ),
        key=lambda row: row["candidate_id"],
    )
    slots = []
    for block_index, block in enumerate(policy["experiment_cycle"], 1):
        for index in range(block["days"]):
            day = len(slots)
            row = available[day] if day < len(available) else None
            variant = block["variants"][index % len(block["variants"])]
            slots.append(
                {
                    "date": str(first + timedelta(days=day)),
                    "candidate_id": row["candidate_id"] if row else None,
                    "refs": row["refs"] if row else [],
                    "status": "context_review_required"
                    if row
                    else "new_candidate_required",
                    "format": variant
                    if block["variable"] == "format"
                    else "everyday_dilemma",
                    "experiment_id": f"{start}-block-{block_index}",
                    "variable": block["variable"],
                    "variant": variant,
                    "fixed_controls": block["fixed"],
                    "selection_note": "Provisional discovery queue, not a score-ranked recommendation. Rebalance topics/durations after review.",
                }
            )
    return {
        "version": "short-form-learning-plan-v2",
        "start_date": start,
        "buffer_target": policy["operations"]["buffer_target"],
        "slots": slots,
        "weekly_review_due": [str(first + timedelta(days=d)) for d in (7, 14, 21, 30)],
    }


def normalize_observation(row: dict) -> dict:
    for key in (
        "episode_id",
        "platform",
        "post_id",
        "observed_at",
        "published_at",
        "evidence",
    ):
        require_text(row.get(key), key)
    if row["platform"] not in PLATFORMS:
        raise WorkflowError("Unknown platform")
    if row.get("publication_time_basis") not in {
        "platform",
        "verified_receipt",
        "estimated",
    }:
        raise WorkflowError("Record whether publication time is exact or estimated")
    if row.get("distribution") not in {"organic", "trial", "paid"}:
        raise WorkflowError("Record organic, trial or paid distribution separately")
    age = (
        stamp(row["observed_at"]) - stamp(row["published_at"])
    ).total_seconds() / 3600
    if age < 0:
        raise WorkflowError("Observation predates publication")
    duration = finite(row.get("duration_seconds"), "duration")
    if duration <= 0:
        raise WorkflowError("Duration must be positive")
    supplied = row.get("metrics", {})
    if not isinstance(supplied, dict) or set(supplied) - METRICS:
        raise WorkflowError("Unknown metric names")
    metrics = {key: supplied.get(key) for key in sorted(METRICS)}
    reasons = row.get("unavailable_reasons", {})
    definitions = row.get("metric_definitions", {})
    for key, value in metrics.items():
        if value is None:
            require_text(reasons.get(key), f"unavailable reason: {key}")
        elif finite(value, key) < 0 or (key in RATE_METRICS and value > 1):
            raise WorkflowError(
                f"Invalid metric: {key}; rates use 0..1, not percentages"
            )
        else:
            require_text(
                definitions.get(key), f"metric definition/unit/denominator: {key}"
            )
    window = next(
        (hours for hours, tolerance in WINDOWS if abs(age - hours) <= tolerance), None
    )
    if row["publication_time_basis"] == "estimated":
        window = None
    rates = {}
    for denominator in ("views", "reach"):
        for numerator in ("shares", "saves", "follows"):
            a, b = metrics[numerator], metrics[denominator]
            rates[f"{numerator}_per_1000_{denominator}"] = (
                round(a / b * 1000, 3)
                if a is not None and b is not None and b > 0
                else None
            )
    return {
        **row,
        "metrics": metrics,
        "age_hours": round(age, 3),
        "window_hours": window,
        "duration_band": "0-15s"
        if duration <= 15
        else "15-30s"
        if duration <= 30
        else "30s+",
        "rates": rates,
        "version": "short-form-observation-v2",
    }


def save_observation(root: Path, row: dict) -> Path:
    normalized = normalize_observation(row)
    identity = {key: normalized[key] for key in ("platform", "post_id", "observed_at")}
    name = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = root / "observations" / f"{name}.json"
    with exclusive_lock(root / ".observations.lock"):
        if destination.exists():
            if read_json(destination) != normalized:
                raise WorkflowError(
                    "Observation conflict; preserve history and record a new observation"
                )
        else:
            atomic_json(destination, normalized)
    return destination


def learning_summary(rows: list[dict], policy: dict) -> dict:
    """One snapshot per post/window; never pool unlike platforms or ages."""
    best = {}
    for raw in rows:
        row = normalize_observation(raw)
        window = row["window_hours"]
        if window is None:
            continue
        key = (row["platform"], row["post_id"], window)
        if key not in best or abs(row["age_hours"] - window) < abs(
            best[key]["age_hours"] - window
        ):
            best[key] = row
    groups: dict[tuple, list] = {}
    for row in best.values():
        exp = row.get("experiment", {})
        key = (
            row["platform"],
            row["distribution"],
            row["window_hours"],
            row["duration_band"],
            exp.get("id", "unassigned"),
            exp.get("variable", "unassigned"),
            exp.get("variant", "unassigned"),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, posts in sorted(groups.items()):
        metrics = {}
        for name in METRICS:
            definitions = {
                row.get("metric_definitions", {}).get(name)
                for row in posts
                if row["metrics"][name] is not None
            }
            if len(definitions) > 1:
                raise WorkflowError(
                    f"Incompatible definitions for {name}; separate the experiment/cohort"
                )
            values = sorted(
                row["metrics"][name]
                for row in posts
                if row["metrics"][name] is not None
            )
            count = len(values)
            metrics[name] = {
                "known_posts": count,
                "unknown_posts": len(posts) - count,
                "median": (values[(count - 1) // 2] + values[count // 2]) / 2
                if count
                else None,
            }
        views = [
            row["metrics"]["views"]
            for row in posts
            if row["metrics"]["views"] is not None
        ]
        enough = (
            len(posts) >= policy["measurement"]["minimum_comparison_posts_per_variant"]
            and len(views) == len(posts)
            and sum(views)
            >= policy["measurement"]["minimum_comparison_views_per_variant"]
        )
        paired_rates = {}
        for denominator in ("views", "reach"):
            for numerator in ("shares", "saves", "follows"):
                pairs = [
                    (row["metrics"][numerator], row["metrics"][denominator])
                    for row in posts
                    if row["metrics"][numerator] is not None
                    and row["metrics"][denominator] is not None
                ]
                total = sum(b for _, b in pairs)
                paired_rates[f"{numerator}_per_1000_{denominator}"] = {
                    "known_posts": len(pairs),
                    "value": sum(a for a, _ in pairs) / total * 1000 if total else None,
                }
        result.append(
            dict(
                zip(
                    (
                        "platform",
                        "distribution",
                        "window_hours",
                        "duration_band",
                        "experiment_id",
                        "variable",
                        "variant",
                    ),
                    key,
                )
            )
            | {
                "posts": len(posts),
                "metrics": metrics,
                "paired_rates": paired_rates,
                "review_status": "eligible_for_editorial_review"
                if enough
                else "insufficient_evidence",
            }
        )
    return {
        "version": "short-form-learning-summary-v2",
        "cohorts": result,
        "causal_claim": False,
        "automatic_winner": None,
        "note": "Thresholds prompt review, not statistical significance. Inspect comprehension and sharing, not views alone.",
    }


def inventory(root: Path, now: datetime) -> dict:
    used_refs: set[str] = set()
    due, buffer, blocked_buffer = [], [], []
    for folder in sorted((root / "output/short-form/episodes").glob("*")):
        state_path = folder / "PUBLICATION_STATE.json"
        if not state_path.exists():
            continue
        state = read_json(state_path)
        source_path = folder / "SOURCE_RECEIPT.json"
        if source_path.exists():
            ref = read_json(source_path).get("ref", "")
            if ":" in ref:
                surah, verses = ref.split(":", 1)
                start, _, end = verses.partition("-")
                used_refs.update(
                    f"{surah}:{n}" for n in range(int(start), int(end or start) + 1)
                )
        accounts = state.get("accounts", {})
        for platform, account in accounts.items():
            if platform not in PLATFORMS or account.get("state") not in {
                "published",
                "public_receipt_verified",
            }:
                continue
            published = (
                account.get("published_at")
                or account.get("published_verified_at")
                or state.get("verified_at")
            )
            if not published:
                continue
            for hours, tolerance in WINDOWS:
                target = stamp(published) + timedelta(hours=hours)
                age = (now - target).total_seconds() / 3600
                due.append(
                    {
                        "episode_id": state["episode_id"],
                        "platform": platform,
                        "url": account.get("url"),
                        "published_at": published,
                        "publication_time_basis": "verified_receipt"
                        if account.get("published_at")
                        else "estimated",
                        "window_hours": hours,
                        "due_at": target.isoformat(),
                        "status": "missed_window"
                        if age > tolerance
                        else "due"
                        if age >= -tolerance
                        else "upcoming",
                    }
                )
        if accounts and all(
            a.get("state") == "not_uploaded" for a in accounts.values()
        ):
            try:
                validate_ready(folder)
                buffer.append(state["episode_id"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                blocked_buffer.append(
                    {"episode_id": state["episode_id"], "reason": str(exc)}
                )
    return {
        "used_refs": sorted(used_refs),
        "measurement_schedule": due,
        "ready_buffer": buffer,
        "buffer_target": 3,
        "buffer_shortfall": max(0, 3 - len(buffer)),
        "blocked_buffer": blocked_buffer,
    }
