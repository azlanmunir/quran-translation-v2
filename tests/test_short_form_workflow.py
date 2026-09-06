from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

from quran_translate import short_form_production as render
from quran_translate.short_form_workflow import (
    CHECKS,
    METRICS,
    VERSION,
    WorkflowError,
    build_plan,
    learning_summary,
    normalize_observation,
    save_observation,
    validate_editorial,
    validate_timing,
    validate_ready,
    inventory,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    return json.loads((ROOT / "configs/short_form_strategy_v2.json").read_text())


@pytest.fixture
def spec():
    return {
        "version": VERSION,
        "episode_id": "test-v2",
        "source": {
            "exact_translation": "Exact complete thought.",
            "ref": "1:1",
            "clip_start_seconds": 10,
            "clip_end_seconds": 16,
            "post_roll_seconds": 0.2,
        },
        "creative": {
            "hook": "A question?",
            "hook_end_seconds": 1.2,
            "quote_start_seconds": 1.2,
            "closing": "A reflection.",
        },
        "render": {"width": 1080, "height": 1920, "duration_seconds": 8.2, "fps": 30},
        "brief": {
            "format": "everyday_dilemma",
            "situation": "A real situation",
            "audience_need": "A question",
            "insight": "A distinction",
            "payoff": "A useful reflection",
            "visual_purpose": "Illustrates the situation",
        },
        "experiment": {
            "id": "opening-1",
            "variable": "opening",
            "variant": "scenario_first",
            "hypothesis": "Earlier clarity may help retention",
            "fixed_controls": "Duration and visual",
            "causal_claim": False,
        },
        "editorial_review": {
            "status": "passed",
            "catalog_sha256": "catalog-hash",
            "quote_sha256": hashlib.sha256(b"Exact complete thought.").hexdigest(),
            "reviewer": "fixture",
            "reviewed_at": "2026-09-05T12:00:00Z",
            "context_explanation": "Fixture context",
            "distinctive_insight": "Fixture insight",
            "religious_meaning": "Preserved",
            "risk_resolution": "Reviewed",
            "comprehension_notes": "Reviewed at normal speed",
            "checks": dict.fromkeys(CHECKS, True),
            "context_refs_reviewed": ["1:1", "1:2"],
            "risk_flags_reviewed": ["context"],
        },
        "platform_copy": dict.fromkeys(
            ["instagram", "tiktok", "youtube", "youtube_title"], "Source and context"
        ),
        "caption_segments": [
            {"start_seconds": 0, "end_seconds": 6, "text": "Exact complete thought."}
        ],
        "visual": {"background_path": "background.png", "background_type": "image"},
    }


def candidate():
    return {
        "refs": ["1:1"],
        "context_after": [{"ref": "1:2"}],
        "editorial": {"risk_flags": ["context"]},
    }


def test_gate_is_not_an_engagement_score(spec):
    assert validate_editorial(spec, candidate(), "catalog-hash")["status"] == "passed"
    spec["editorial_review"]["checks"]["context_preserved"] = False
    spec["editorial_review"]["selection_score"] = 100
    with pytest.raises(WorkflowError, match="All editorial"):
        validate_editorial(spec, candidate(), "catalog-hash")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("quote_sha256", "wrong", "exact quote"),
        ("catalog_sha256", "wrong", "catalog checksum"),
        ("context_refs_reviewed", ["1:1"], "adjacent context"),
        ("risk_flags_reviewed", [], "risk flags"),
        ("reviewer", "", "reviewer"),
    ],
)
def test_gate_rejects_stale_or_missing_evidence(spec, field, value, match):
    spec["editorial_review"][field] = value
    with pytest.raises(WorkflowError, match=match):
        validate_editorial(spec, candidate(), "catalog-hash")


def test_sequential_timing_and_optional_question(spec):
    assert validate_timing(spec)["quote_end_seconds"] == 7.2
    spec["creative"].update(hook="", hook_end_seconds=0, quote_start_seconds=0)
    spec["render"]["duration_seconds"] = 7
    assert validate_timing(spec)["closing_seconds"] == 1


@pytest.mark.parametrize(
    "change", ["overlap", "tail", "nan", "short_tail", "blank_intro"]
)
def test_invalid_timing_fails(spec, change):
    if change == "overlap":
        spec["creative"]["quote_start_seconds"] = 0
    elif change == "tail":
        spec["render"]["duration_seconds"] = 12
    elif change == "nan":
        spec["render"]["duration_seconds"] = float("nan")
    elif change == "short_tail":
        spec["render"]["duration_seconds"] = 7.3
    else:
        spec["creative"]["hook"] = ""
    with pytest.raises(WorkflowError):
        validate_timing(spec)


def test_narrative_has_separate_context_window(spec):
    spec["brief"]["format"] = "short_narrative"
    spec["creative"].update(
        setup="Necessary context, clearly commentary.", quote_start_seconds=5
    )
    spec["render"]["duration_seconds"] = 12
    assert validate_timing(spec)["quote_start_seconds"] == 5


def observation(**changes):
    row = {
        "episode_id": "episode-1",
        "platform": "tiktok",
        "post_id": "post-1",
        "observed_at": "2026-09-06T12:00:00Z",
        "published_at": "2026-09-05T12:00:00Z",
        "publication_time_basis": "platform",
        "distribution": "organic",
        "duration_seconds": 12,
        "evidence": "Native post insights captured at observation time",
        "metrics": {"views": 100, "shares": 2, "likes": 0},
        "metric_definitions": {
            "views": "Native cumulative video views",
            "shares": "Native cumulative shares",
            "likes": "Native cumulative likes",
        },
        "unavailable_reasons": dict.fromkeys(
            METRICS - {"views", "shares", "likes"},
            "Not exposed in this account's insights",
        ),
        "experiment": {
            "id": "opening-1",
            "variable": "opening",
            "variant": "scenario_first",
        },
    }
    return row | changes


def test_unknown_zero_denominator_and_estimated_time():
    row = normalize_observation(observation())
    assert row["metrics"]["likes"] == 0 and row["metrics"]["saves"] is None
    assert row["rates"]["shares_per_1000_views"] == 20
    assert row["rates"]["shares_per_1000_reach"] is None
    assert row["window_hours"] == 24
    assert (
        normalize_observation(observation(publication_time_basis="estimated"))[
            "window_hours"
        ]
        is None
    )
    row = observation()
    row["metrics"]["views"] = 0
    assert normalize_observation(row)["rates"]["shares_per_1000_views"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_at": "2026-09-04T12:00:00Z"},
        {"observed_at": "2026-09-06T12:00:00"},
        {"metrics": {"views": -1}},
        {"metrics": {"views": float("inf")}},
        {"unavailable_reasons": {}},
        {"metric_definitions": {}},
    ],
)
def test_bad_metrics_fail(changes):
    with pytest.raises(WorkflowError):
        normalize_observation(observation(**changes))


def test_snapshots_are_idempotent_and_immutable(tmp_path):
    path = save_observation(tmp_path, observation())
    assert save_observation(tmp_path, observation()) == path
    with pytest.raises(WorkflowError, match="conflict"):
        save_observation(
            tmp_path, observation(metrics={"views": 101, "shares": 2, "likes": 0})
        )


def test_summary_separates_platform_age_distribution_and_deduplicates(policy):
    rows = [
        observation(),
        observation(observed_at="2026-09-06T13:00:00Z"),
        observation(platform="instagram"),
        observation(distribution="trial", post_id="trial-2"),
        observation(observed_at="2026-09-08T12:00:00Z"),
        observation(duration_seconds=40, post_id="long-3"),
    ]
    result = learning_summary(rows, policy)
    assert len(result["cohorts"]) == 5
    assert all(row["posts"] == 1 for row in result["cohorts"])
    assert result["automatic_winner"] is None
    assert all(
        row["review_status"] == "insufficient_evidence" for row in result["cohorts"]
    )
    assert result["cohorts"][0]["paired_rates"]["shares_per_1000_views"]["value"] == 20


def test_plan_never_approves_candidates_or_reuses_passages(policy):
    catalog = {
        "candidates": [
            {"candidate_id": "used", "refs": ["1:1", "1:2"]},
            {"candidate_id": "new", "refs": ["1:3"]},
        ]
    }
    plan = build_plan(catalog, policy, {"1:2"}, "2026-09-06")
    assert len(plan["slots"]) == 30
    assert plan["slots"][0]["candidate_id"] == "new"
    assert plan["slots"][0]["status"] == "context_review_required"
    assert plan["slots"][1]["status"] == "new_candidate_required"
    assert plan["slots"][-1]["date"] == "2026-10-05"


def test_srt_and_render_offsets_match(tmp_path, spec, monkeypatch):
    render._write_platform_copy(spec, tmp_path)
    assert "00:00:01,200 --> 00:00:07,200" in (tmp_path / "captions.en.srt").read_text()
    assert (tmp_path / "caption-youtube_title.txt").exists()
    commands = []
    monkeypatch.setattr(render, "_run", commands.append)
    layers = {
        key: tmp_path / f"{key}.png" for key in ("common", "hook", "caption_1", "outro")
    }
    render._render_video(
        spec, tmp_path, layers, tmp_path / "audio.wav", tmp_path / "master.mp4"
    )
    graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "adelay=1200:all=1" in graph
    assert "gte(t,1.200)*lt(t,7.200)" in graph
    assert "gte(t,0.000)*lt(t,1.200)" in graph


def test_protected_pre_roll_keeps_audio_and_captions_in_sync(tmp_path, spec, monkeypatch):
    spec["source"]["pre_roll_seconds"] = 0.4
    spec["render"]["duration_seconds"] += 0.4
    assert validate_timing(spec)["quote_end_seconds"] == pytest.approx(7.6)
    render._write_platform_copy(spec, tmp_path)
    assert "00:00:01,600 --> 00:00:07,600" in (tmp_path / "captions.en.srt").read_text()
    commands = []
    monkeypatch.setattr(render, "_run", commands.append)
    render._extract_audio(spec, tmp_path / "source.mp4", tmp_path / "audio.wav")
    assert float(commands[0][commands[0].index("-ss") + 1]) == pytest.approx(spec["source"]["clip_start_seconds"] - 0.4)
    assert float(commands[0][commands[0].index("-t") + 1]) == pytest.approx(6.6)
    layers = {key: tmp_path / f"{key}.png" for key in ("common", "hook", "caption_1", "outro")}
    render._render_video(spec, tmp_path, layers, tmp_path / "audio.wav", tmp_path / "master.mp4")
    graph = commands[1][commands[1].index("-filter_complex") + 1]
    assert "adelay=1200:all=1" in graph
    assert "gte(t,1.600)*lt(t,7.600)" in graph
    assert "gte(t,7.600)" in graph


@pytest.mark.parametrize("pre_roll", [-0.1, 0.501, float("nan"), float("inf")])
def test_invalid_pre_roll_rejected(spec, pre_roll):
    spec["source"]["pre_roll_seconds"] = pre_roll
    with pytest.raises(WorkflowError, match="pre.roll"):
        validate_timing(spec)



def test_ready_requires_creative_review(tmp_path):
    (tmp_path / "EPISODE_SPEC.json").write_text(json.dumps({"version": VERSION}))
    (tmp_path / "RENDER_COMPLETE.json").write_text(json.dumps({"artifacts": {}}))
    with pytest.raises(WorkflowError, match="manifest"):
        validate_ready(tmp_path)


def test_inventory_reads_legacy_receipts_without_fabricating_post_time(tmp_path):
    directory = tmp_path / "output/short-form/episodes/legacy"
    directory.mkdir(parents=True)
    (directory / "PUBLICATION_STATE.json").write_text(
        json.dumps(
            {
                "episode_id": "legacy",
                "verified_at": "2026-09-05T12:00:00Z",
                "accounts": {
                    "tiktok": {
                        "state": "public_receipt_verified",
                        "url": "https://example.test/post",
                    }
                },
            }
        )
    )
    result = inventory(tmp_path, datetime(2026, 9, 6, 12, tzinfo=timezone.utc))
    assert len(result["measurement_schedule"]) == 3
    assert result["measurement_schedule"][0]["publication_time_basis"] == "estimated"
    assert result["measurement_schedule"][0]["status"] == "due"
    assert result["ready_buffer"] == []


def test_ready_rejects_unbound_creative_review_and_changed_master(tmp_path):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"fixture media")
    digest = hashlib.sha256(master.read_bytes()).hexdigest()
    payloads = {
        "EPISODE_SPEC.json": {"version": VERSION, "episode_id": "fixture"},
        "QA.json": {
            "video": {"decode_passed": True},
            "spoken_audio": {"passed": True},
            "encoded_spoken_audio": {"passed": True},
            "artifacts": {"master": {"path": str(master), "sha256": digest}},
        },
    }
    manifest = {"master.mp4": digest}
    for name, data in payloads.items():
        path = tmp_path / name
        path.write_text(json.dumps(data))
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "RENDER_COMPLETE.json").write_text(json.dumps({"artifacts": manifest}))
    review = {
        "status": "passed",
        "master_sha256": "wrong",
        "reviewer": "fixture",
        "reviewed_at": "2026-09-05T12:00:00Z",
        "notes": "Reviewed",
        "visual_review": True,
        "audio_review": True,
        "comprehension_review": True,
    }
    (tmp_path / "CREATIVE_REVIEW.json").write_text(json.dumps(review))
    with pytest.raises(WorkflowError, match="bound visual"):
        validate_ready(tmp_path)
    review["master_sha256"] = digest
    (tmp_path / "CREATIVE_REVIEW.json").write_text(json.dumps(review))
    assert validate_ready(tmp_path)["ready"]
    master.write_bytes(b"changed")
    with pytest.raises(WorkflowError, match="checksum"):
        validate_ready(tmp_path)


def test_summary_rejects_incompatible_metric_definitions(policy):
    first = observation()
    second = observation(post_id="second")
    second["metric_definitions"]["views"] = "Engaged views, not all views"
    with pytest.raises(WorkflowError, match="Incompatible definitions"):
        learning_summary([first, second], policy)


def test_video_background_renders_without_its_original_audio(tmp_path, spec):
    import shutil
    import subprocess
    from PIL import Image

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    background = tmp_path / "motion.mp4"
    wav = tmp_path / "voice.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=192x320:rate=30",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(background),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "1",
            str(wav),
        ],
        check=True,
    )
    layers = {}
    for key in ("common", "hook", "caption_1", "outro"):
        layers[key] = tmp_path / f"{key}.png"
        Image.new("RGBA", (192, 320)).save(layers[key])
    spec["visual"].update(background_type="video", background_path=str(background))
    spec["render"].update(width=192, height=320, duration_seconds=2.5)
    spec["source"]["clip_end_seconds"] = 11
    spec["caption_segments"][0]["end_seconds"] = 1
    master = tmp_path / "master.mp4"
    render._render_video(spec, tmp_path, layers, wav, master)
    probe = render._ffprobe(master)
    assert {s["codec_name"] for s in probe["streams"]} == {"h264", "aac"}
    assert abs(float(probe["format"]["duration"]) - 2.5) < 0.1
    render._run(["ffmpeg", "-v", "error", "-i", str(master), "-f", "null", "-"])
