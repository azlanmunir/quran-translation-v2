from unittest.mock import Mock

import pytest

from scripts import run_urdu_video_with_layout_recovery as wrapper


COMPLETE = {"run_id": "fixture", "segments": 343, "para_videos": 30,
            "surah_videos": 114, "decode_checked": True}


def test_success_exits_after_one_pipeline_call(monkeypatch):
    pipeline = Mock(return_value=COMPLETE)
    monkeypatch.setattr(wrapper, "failed_layout_unit", lambda _: None)
    monkeypatch.setattr(wrapper.production, "pipeline", pipeline)
    wrapper.run("fixture")
    assert pipeline.call_count == 1


def test_one_layout_repair_then_success(monkeypatch):
    repair = Mock()
    monkeypatch.setattr(wrapper, "repair", repair)
    monkeypatch.setattr(wrapper, "failed_layout_unit", lambda _: 6)
    monkeypatch.setattr(wrapper.production, "pipeline", lambda **k: COMPLETE)
    wrapper.run("fixture")
    repair.assert_called_once_with(run_id="fixture", unit_index=6)


def test_repeated_layout_failure_is_bounded(monkeypatch):
    repair = Mock()
    monkeypatch.setattr(wrapper, "repair", repair)
    monkeypatch.setattr(wrapper, "failed_layout_unit", lambda _: 6)
    pipeline = Mock(side_effect=wrapper.production.UrduVideoProductionError("layout"))
    monkeypatch.setattr(wrapper.production, "pipeline", pipeline)
    with pytest.raises(RuntimeError, match="exhausted"):
        wrapper.run("fixture")
    assert repair.call_count == 1 and pipeline.call_count == 1


def test_non_layout_error_propagates(monkeypatch):
    monkeypatch.setattr(wrapper, "failed_layout_unit", lambda _: None)
    monkeypatch.setattr(wrapper.production, "pipeline",
                        Mock(side_effect=wrapper.production.UrduVideoProductionError("integrity")))
    with pytest.raises(wrapper.production.UrduVideoProductionError, match="integrity"):
        wrapper.run("fixture")


@pytest.mark.parametrize("key,value", [
    ("run_id", "wrong"), ("segments", 342), ("para_videos", 29),
    ("surah_videos", 113), ("decode_checked", False),
])
def test_incomplete_or_wrong_run_marker_is_not_success(monkeypatch, key, value):
    monkeypatch.setattr(wrapper, "failed_layout_unit", lambda _: None)
    monkeypatch.setattr(wrapper.production, "pipeline", lambda **k: dict(COMPLETE, **{key: value}))
    with pytest.raises(RuntimeError, match="validated completion"):
        wrapper.run("fixture")
