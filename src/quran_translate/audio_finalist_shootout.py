"""Blind final narrator shootout using pinned and reusable source audio."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio_bakeoff import (
    DEFAULT_BAKEOFF_ROOT,
    EXPECTED_FINAL_TEXT_SHA256,
    NARRATION_DIRECTION,
    RELEASE_VERSION,
    CandidateSpec,
    _ffprobe,
    _generate_gemini,
    _normalize_audio,
    passage_payloads,
)
from .audio_google_voice_audition import DEFAULT_ROOT as GOOGLE_AUDITION_ROOT
from .config import OUTPUT_DIR, file_sha256, text_sha256
from .db import utc_now
from .production_packets import atomic_json, atomic_text


SHOOTOUT_ID = "quran-v2.4.1-tts-finalists-20260811"
DEFAULT_ROOT = OUTPUT_DIR / "audio" / "bakeoffs" / SHOOTOUT_ID
PASSAGE_IDS = ("consolation", "oaths", "mary")
NATHAN_CANDIDATE_ID = "eleven-nathan-v3"
CHARON_CANDIDATE = CandidateSpec(
    "gemini-pro-charon",
    "gemini",
    "gemini-2.5-pro-preview-tts",
    "Charon",
    "Charon - Informative (male)",
    ".wav",
    "pcm_s16le_24000_mono",
)


@dataclass(frozen=True)
class FinalistSpec:
    finalist_id: str
    private_label: str
    mode: str
    pitch_semitones: float | None = None


FINALISTS = (
    FinalistSpec("nathan-v3-original", "Nathan v3 - original", "reuse_nathan"),
    FinalistSpec(
        "nathan-v3-minus-075",
        "Nathan v3 - 0.75 semitone lower",
        "pitch_nathan",
        -0.75,
    ),
    FinalistSpec(
        "nathan-v3-minus-125",
        "Nathan v3 - 1.25 semitones lower",
        "pitch_nathan",
        -1.25,
    ),
    FinalistSpec("gemini-pro-charon", "Gemini Pro - Charon", "charon"),
)


class ShootoutError(RuntimeError):
    """Raised when shootout provenance, generation, or packaging fails."""


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ShootoutError(f"Expected a JSON object: {path}")
    return payload


def _source_from_run(
    run_path: Path,
    candidate_id: str,
    passage_id: str | None = None,
) -> dict[str, str]:
    state = _read_object(run_path)
    matches = [
        job
        for job in state.get("jobs", [])
        if job.get("candidate_id") == candidate_id
        and (passage_id is None or job.get("passage_id") == passage_id)
    ]
    if len(matches) != 1 or matches[0].get("status") != "complete":
        raise ShootoutError(
            f"Missing one complete source for {candidate_id} {passage_id or ''}".strip()
        )
    job = matches[0]
    source_path = Path(str(job["raw_path"]))
    expected_hash = str(job.get("raw_sha256") or "")
    if not source_path.exists() or not expected_hash:
        raise ShootoutError(f"Incomplete source provenance for {source_path}")
    actual_hash = file_sha256(source_path)
    if actual_hash != expected_hash:
        raise ShootoutError(f"Source hash mismatch: {source_path}")
    return {"path": str(source_path), "sha256": actual_hash}


def _passages() -> list[dict[str, Any]]:
    by_id = {passage["passage_id"]: passage for passage in passage_payloads()}
    missing = [passage_id for passage_id in PASSAGE_IDS if passage_id not in by_id]
    if missing:
        raise ShootoutError(f"Frozen release lacks passages: {missing}")
    return [by_id[passage_id] for passage_id in PASSAGE_IDS]


def _source_inventory() -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    for passage_id in PASSAGE_IDS:
        sources[f"nathan::{passage_id}"] = _source_from_run(
            DEFAULT_BAKEOFF_ROOT / "RUN.json",
            NATHAN_CANDIDATE_ID,
            passage_id,
        )
    sources["charon::consolation"] = _source_from_run(
        GOOGLE_AUDITION_ROOT / "RUN.json",
        CHARON_CANDIDATE.candidate_id,
    )
    return sources


def _rubberband_version() -> str:
    try:
        completed = subprocess.run(
            ["rubberband", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ShootoutError("Rubber Band 4.x is required for pitch variants") from exc
    version = completed.stdout.strip() or completed.stderr.strip()
    if not version.startswith("4."):
        raise ShootoutError(f"Unreviewed Rubber Band version: {version}")
    return version


def _fingerprint(
    passages: list[dict[str, Any]],
    sources: dict[str, dict[str, str]],
    rubberband_version: str,
) -> str:
    payload = {
        "version": "tts-finalist-shootout-v1",
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "narration_direction": NARRATION_DIRECTION,
        "passages": passages,
        "finalists": [asdict(finalist) for finalist in FINALISTS],
        "charon_candidate": asdict(CHARON_CANDIDATE),
        "source_hashes": sources,
        "rubberband_version": rubberband_version,
        "pitch_engine": "Rubber Band R3 fine engine with formant preservation",
    }
    return text_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def prepare(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    passages = _passages()
    sources = _source_inventory()
    rubberband_version = _rubberband_version()
    fingerprint = _fingerprint(passages, sources, rubberband_version)
    state_path = root / "RUN.json"
    if state_path.exists():
        state = _read_object(state_path)
        if state.get("input_fingerprint") != fingerprint:
            raise ShootoutError("Shootout inputs changed; refusing a mixed resume")
        return state

    jobs: list[dict[str, Any]] = []
    for finalist in FINALISTS:
        for passage in passages:
            passage_id = str(passage["passage_id"])
            if finalist.mode.startswith(("reuse_nathan", "pitch_nathan")):
                source = sources[f"nathan::{passage_id}"]
            elif passage_id == "consolation":
                source = sources["charon::consolation"]
            else:
                source = {}
            extension = ".mp3" if finalist.mode == "reuse_nathan" else ".wav"
            jobs.append(
                {
                    "job_id": f"{finalist.finalist_id}--{passage_id}",
                    "finalist_id": finalist.finalist_id,
                    "passage_id": passage_id,
                    "mode": finalist.mode,
                    "pitch_semitones": finalist.pitch_semitones,
                    "source_path": source.get("path"),
                    "source_sha256": source.get("sha256"),
                    "raw_path": str(
                        root / "raw" / finalist.finalist_id / f"{passage_id}{extension}"
                    ),
                    "status": "pending",
                    "attempts": 0,
                    "last_error": None,
                }
            )
    state = {
        "version": "tts-finalist-shootout-run-v1",
        "shootout_id": SHOOTOUT_ID,
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "input_fingerprint": fingerprint,
        "narration_direction_sha256": text_sha256(NARRATION_DIRECTION),
        "prepared_at": utc_now(),
        "updated_at": utc_now(),
        "status": "prepared",
        "passages": passages,
        "finalists": [asdict(finalist) for finalist in FINALISTS],
        "charon_candidate": asdict(CHARON_CANDIDATE),
        "source_inventory": sources,
        "pitch_processing": {
            "tool": "rubberband",
            "version": rubberband_version,
            "engine": "R3 fine",
            "formant_preservation": True,
        },
        "jobs": jobs,
    }
    atomic_json(state_path, state)
    atomic_text(root / "NARRATION_DIRECTION.txt", NARRATION_DIRECTION + "\n")
    return state


def _pitch_command(source_wav: Path, destination_wav: Path, semitones: float) -> list[str]:
    return [
        "rubberband",
        "--quiet",
        "--fine",
        "--formant",
        "--pitch",
        str(semitones),
        str(source_wav),
        str(destination_wav),
    ]


def _pitch_shift(
    source: Path,
    destination: Path,
    semitones: float,
    work_dir: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    decoded = work_dir / f"{destination.parent.name}-{destination.stem}-source.wav"
    temp = destination.with_name(f".{destination.name}.tmp.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ar",
            "44100",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(decoded),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        _pitch_command(decoded, temp, semitones),
        check=True,
        capture_output=True,
    )
    os.replace(temp, destination)
    decoded.unlink(missing_ok=True)
    return {
        "bytes": destination.stat().st_size,
        "pitch_semitones": semitones,
        "engine": "Rubber Band R3 fine with formant preservation",
    }


def _blind_mapping(root: Path) -> dict[str, str]:
    path = root / "PRIVATE_BLIND_KEY.json"
    finalist_ids = [finalist.finalist_id for finalist in FINALISTS]
    if path.exists():
        mapping = _read_object(path).get("finalist_to_code")
        if not isinstance(mapping, dict) or set(mapping) != set(finalist_ids):
            raise ShootoutError("Blind key does not match the finalist set")
        return {str(key): str(value) for key, value in mapping.items()}
    codes = [f"Voice {chr(65 + index)}" for index in range(len(finalist_ids))]
    secrets.SystemRandom().shuffle(codes)
    mapping = dict(zip(finalist_ids, codes, strict=True))
    atomic_json(
        path,
        {
            "version": "tts-finalist-shootout-key-v1",
            "created_at": utc_now(),
            "finalist_to_code": mapping,
        },
    )
    os.chmod(path, 0o600)
    return mapping


def _listening_html(manifest: dict[str, Any]) -> str:
    passages_json = json.dumps(manifest["passages"], ensure_ascii=False)
    voices_json = json.dumps(manifest["voice_codes"], ensure_ascii=False)
    clips_json = json.dumps(manifest["clips"], ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final Quran Narrator Shootout</title><style>
*{{box-sizing:border-box}} body{{margin:0;color:#171717;font:16px/1.5 Arial,sans-serif}} header,main{{width:min(1100px,calc(100% - 40px));margin:auto}}
header{{padding:34px 0 22px;border-bottom:1px solid #ddd}} h1{{font:600 31px Georgia,serif;margin:0 0 8px}} .muted{{color:#666;margin:0}}
.passage{{padding:26px 0 32px;border-bottom:1px solid #ddd}} h2{{font:600 24px Georgia,serif;margin:0 0 5px}} .meta{{font-size:13px;color:#9c2525;font-weight:700}}
.transcript{{max-width:900px;font:18px/1.58 Georgia,serif}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.voice{{border:1px solid #d8d8d8;border-radius:6px;padding:15px}} h3{{font-size:16px;margin:0 0 10px}} audio{{width:100%}}
textarea{{width:100%;min-height:58px;margin-top:10px;padding:8px}} @media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Final Quran Narrator Shootout</h1><p class="muted">Fresh blind codes. Compare naturalness, vocal depth, restraint, pacing, pronunciation, and long-listening comfort.</p></header>
<main id="app"></main><script>
const passages={passages_json},voices={voices_json},clips={clips_json};const app=document.getElementById('app');
const esc=v=>String(v).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
for(const p of passages){{const s=document.createElement('section');s.className='passage';s.innerHTML=`<div class="meta">${{esc(p.start_ref)}}-${{esc(p.end_ref)}} · ${{esc(p.register)}}</div><h2>${{esc(p.label)}}</h2><p class="transcript">${{esc(p.text)}}</p><div class="grid"></div>`;const g=s.querySelector('.grid');for(const v of voices){{const c=clips.find(x=>x.passage_id===p.passage_id&&x.voice_code===v);const b=document.createElement('div');b.className='voice';b.innerHTML=`<h3>${{esc(v)}}</h3><audio controls preload="metadata" src="${{encodeURI(c.file)}}"></audio><textarea placeholder="Naturalness, depth, fatigue, artifacts"></textarea>`;g.appendChild(b)}}app.appendChild(s)}}
</script></body></html>"""


def package(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    state = _read_object(root / "RUN.json")
    if any(job["status"] != "complete" for job in state["jobs"]):
        raise ShootoutError("Cannot package an incomplete shootout")
    mapping = _blind_mapping(root)
    blind_dir = root / "blind"
    clips: list[dict[str, Any]] = []
    for job in state["jobs"]:
        code = mapping[job["finalist_id"]]
        destination = (
            blind_dir
            / "clips"
            / f"{job['passage_id']}--{code.lower().replace(' ', '-')}.mp3"
        )
        _normalize_audio(Path(job["raw_path"]), destination)
        clips.append(
            {
                "passage_id": job["passage_id"],
                "voice_code": code,
                "file": str(destination.relative_to(blind_dir)),
                "sha256": file_sha256(destination),
                "probe": _ffprobe(destination),
            }
        )
    manifest = {
        "version": "tts-finalist-shootout-package-v1",
        "shootout_id": SHOOTOUT_ID,
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "created_at": utc_now(),
        "voice_codes": sorted(mapping.values()),
        "passages": state["passages"],
        "clips": sorted(
            clips, key=lambda item: (item["passage_id"], item["voice_code"])
        ),
        "normalization": "mono MP3, 44.1 kHz, 192 kbps, -18 LUFS target, -1.5 dBTP",
    }
    atomic_json(blind_dir / "BLIND_MANIFEST.json", manifest)
    atomic_text(blind_dir / "listen.html", _listening_html(manifest))
    state["status"] = "ready_for_blind_review"
    state["updated_at"] = utc_now()
    state["blind_package"] = str(blind_dir)
    atomic_json(root / "RUN.json", state)
    return manifest


def run(root: Path = DEFAULT_ROOT, *, max_attempts: int = 2) -> dict[str, Any]:
    state = prepare(root)
    passages = {item["passage_id"]: item for item in state["passages"]}
    state["status"] = "running"
    atomic_json(root / "RUN.json", state)
    for job in state["jobs"]:
        if job["status"] == "complete":
            continue
        output_path = Path(job["raw_path"])
        for _ in range(max_attempts - int(job["attempts"])):
            job["attempts"] += 1
            started = time.monotonic()
            try:
                mode = job["mode"]
                if mode == "reuse_nathan" or (
                    mode == "charon" and job["source_path"]
                ):
                    source = Path(str(job["source_path"]))
                    if file_sha256(source) != job["source_sha256"]:
                        raise ShootoutError(f"Source drift for {job['job_id']}")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if output_path != source:
                        temp = output_path.with_name(f".{output_path.name}.tmp")
                        temp.write_bytes(source.read_bytes())
                        os.replace(temp, output_path)
                    usage = {"mode": "hash-validated reuse"}
                elif mode == "pitch_nathan":
                    usage = _pitch_shift(
                        Path(str(job["source_path"])),
                        output_path,
                        float(job["pitch_semitones"]),
                        root / "work",
                    )
                elif mode == "charon":
                    usage = _generate_gemini(
                        CHARON_CANDIDATE,
                        passages[job["passage_id"]]["text"],
                        output_path,
                    )
                else:
                    raise ShootoutError(f"Unknown shootout mode: {mode}")
                job.update(
                    {
                        "status": "complete",
                        "last_error": None,
                        "completed_at": utc_now(),
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "raw_sha256": file_sha256(output_path),
                        "provider_usage": usage,
                        "probe": _ffprobe(output_path),
                    }
                )
                break
            except Exception as exc:  # Provider and audio tools expose varied errors.
                job["status"] = "failed"
                job["last_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                state["updated_at"] = utc_now()
                atomic_json(root / "RUN.json", state)
        if job["status"] != "complete":
            state["status"] = "blocked"
            atomic_json(root / "RUN.json", state)
            raise ShootoutError(f"Shootout failed: {job['last_error']}")
    state["status"] = "generated"
    atomic_json(root / "RUN.json", state)
    package(root)
    return _read_object(root / "RUN.json")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
