"""Resumable, release-pinned text-to-speech bakeoff."""

from __future__ import annotations

import base64
import json
import os
import secrets
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .config import OUTPUT_DIR, file_sha256, load_dotenv, text_sha256
from .db import utc_now
from .elevenlabs_tts import ElevenLabsError, synthesize_text
from .production_packets import atomic_json, atomic_text


BAKEOFF_ID = "quran-v2.4.1-tts-20260809-r2"
RELEASE_VERSION = "v2.4.1"
EXPECTED_FINAL_TEXT_SHA256 = (
    "09e302204ea5ebf90e5e1fcd13ac924495dd7f6bd8585105f1f1933c0557291b"
)
RELEASE_DIR = OUTPUT_DIR / "release" / f"quran-translation-{RELEASE_VERSION}"
RELEASE_MANIFEST = RELEASE_DIR / "MANIFEST.json"
LISTENING_EDITION = RELEASE_DIR / "quran-listening-edition.json"
DEFAULT_BAKEOFF_ROOT = OUTPUT_DIR / "audio" / "bakeoffs" / BAKEOFF_ID

NARRATION_DIRECTION = (
    "Read the quoted text exactly, without adding, omitting, or paraphrasing any words. "
    "Use natural, mature audiobook narration in clear American English. Keep the delivery "
    "measured, warm, and attentive, with quiet authority. Be reverent without sounding "
    "ceremonial, theatrical, preachy, or like a movie trailer. Let punctuation shape the "
    "pauses. Give quoted dialogue subtle emotional distinction without performing character "
    "voices. Do not announce these directions and do not announce the passage reference."
)


@dataclass(frozen=True)
class PassageSpec:
    passage_id: str
    label: str
    refs: tuple[str, ...]
    register: str


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    provider: str
    model_id: str
    voice_id: str
    voice_label: str
    output_extension: str
    source_format: str


PASSAGES = (
    PassageSpec(
        "opening",
        "The Opening",
        tuple(f"1:{ayah}" for ayah in range(1, 8)),
        "foundational prayer and cadence",
    ),
    PassageSpec(
        "consolation",
        "Morning Light",
        tuple(f"93:{ayah}" for ayah in range(1, 12)),
        "consolation, tenderness, and restraint",
    ),
    PassageSpec(
        "oaths",
        "The Folding Up",
        tuple(f"81:{ayah}" for ayah in range(1, 15)),
        "eschatological oaths and accumulating force",
    ),
    PassageSpec(
        "mary",
        "Mary",
        tuple(f"19:{ayah}" for ayah in range(16, 27)),
        "narrative, dialogue, grief, and reassurance",
    ),
    PassageSpec(
        "legal",
        "The Light",
        tuple(f"24:{ayah}" for ayah in range(2, 6)),
        "legal prose and moral gravity",
    ),
    PassageSpec(
        "refrain",
        "The Merciful",
        tuple(f"55:{ayah}" for ayah in range(1, 14)),
        "lyric movement and refrain delivery",
    ),
)

CANDIDATES = (
    CandidateSpec(
        "eleven-nathan-v3",
        "elevenlabs",
        "eleven_v3",
        "lWDDHwXsJXJM7nv2YgHY",
        "Nathan - Natural Narrator",
        ".mp3",
        "mp3_44100_128",
    ),
    CandidateSpec(
        "eleven-nathan-multilingual-v2",
        "elevenlabs",
        "eleven_multilingual_v2",
        "lWDDHwXsJXJM7nv2YgHY",
        "Nathan - Natural Narrator",
        ".mp3",
        "mp3_44100_128",
    ),
    CandidateSpec(
        "gemini-pro-gacrux",
        "gemini",
        "gemini-2.5-pro-preview-tts",
        "Gacrux",
        "Gacrux - Mature",
        ".wav",
        "pcm_s16le_24000_mono",
    ),
    CandidateSpec(
        "gemini-31-gacrux",
        "gemini",
        "gemini-3.1-flash-tts-preview",
        "Gacrux",
        "Gacrux - Mature",
        ".wav",
        "pcm_s16le_24000_mono",
    ),
)


class BakeoffError(RuntimeError):
    """Raised when bakeoff integrity or provider generation fails."""


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BakeoffError(f"Expected a JSON object: {path}")
    return payload


def _release_rows() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not RELEASE_MANIFEST.exists() or not LISTENING_EDITION.exists():
        raise BakeoffError(f"Missing frozen {RELEASE_VERSION} release artifacts")
    manifest = _read_object(RELEASE_MANIFEST)
    if manifest.get("qa_passed") is not True:
        raise BakeoffError("Refusing audio generation from a release that did not pass QA")
    if manifest.get("final_text_sha256") != EXPECTED_FINAL_TEXT_SHA256:
        raise BakeoffError("Frozen release text hash does not match the approved v2.4.1 text")
    expected_artifact_hash = manifest.get("artifacts", {}).get(
        LISTENING_EDITION.name
    )
    if expected_artifact_hash != file_sha256(LISTENING_EDITION):
        raise BakeoffError("Listening-edition artifact hash does not match its release manifest")

    payload = _read_object(LISTENING_EDITION)
    ayahs = payload.get("ayahs")
    if not isinstance(ayahs, list) or len(ayahs) != 6236:
        raise BakeoffError("Listening edition must contain exactly 6,236 ayahs")
    rows: dict[str, dict[str, Any]] = {}
    hash_material: list[str] = []
    for row in ayahs:
        if not isinstance(row, dict) or not isinstance(row.get("ref"), str):
            raise BakeoffError("Malformed ayah in listening edition")
        ref = row["ref"]
        translation = row.get("translation")
        if ref in rows or not isinstance(translation, str) or not translation.strip():
            raise BakeoffError(f"Malformed or duplicate listening-edition ayah: {ref}")
        rows[ref] = row
        hash_material.append(f"{ref}\t{translation}")
    calculated = text_sha256("\n".join(hash_material))
    if calculated != EXPECTED_FINAL_TEXT_SHA256:
        raise BakeoffError("Listening-edition text does not reproduce the approved final hash")
    return rows, manifest


def passage_payloads() -> list[dict[str, Any]]:
    rows, _ = _release_rows()
    payloads: list[dict[str, Any]] = []
    for spec in PASSAGES:
        missing = [ref for ref in spec.refs if ref not in rows]
        if missing:
            raise BakeoffError(f"Bakeoff passage lacks refs: {missing}")
        text = " ".join(str(rows[ref]["translation"]).strip() for ref in spec.refs)
        payloads.append(
            {
                **asdict(spec),
                "refs": list(spec.refs),
                "start_ref": spec.refs[0],
                "end_ref": spec.refs[-1],
                "text": text,
                "text_sha256": text_sha256(text),
                "char_count": len(text),
            }
        )
    return payloads


def _input_fingerprint(passages: list[dict[str, Any]]) -> str:
    payload = {
        "version": "tts-bakeoff-v1",
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "narration_direction": NARRATION_DIRECTION,
        "passages": passages,
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
    }
    return text_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def prepare_bakeoff(root: Path = DEFAULT_BAKEOFF_ROOT) -> dict[str, Any]:
    passages = passage_payloads()
    fingerprint = _input_fingerprint(passages)
    state_path = root / "RUN.json"
    if state_path.exists():
        state = _read_object(state_path)
        if state.get("input_fingerprint") != fingerprint:
            raise BakeoffError("Bakeoff inputs changed; refusing a mixed-version resume")
        return state

    jobs: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for passage in passages:
            raw_path = (
                root
                / "raw"
                / candidate.candidate_id
                / f"{passage['passage_id']}{candidate.output_extension}"
            )
            jobs.append(
                {
                    "job_id": f"{candidate.candidate_id}--{passage['passage_id']}",
                    "candidate_id": candidate.candidate_id,
                    "passage_id": passage["passage_id"],
                    "raw_path": str(raw_path),
                    "status": "pending",
                    "attempts": 0,
                    "last_error": None,
                }
            )
    state = {
        "version": "tts-bakeoff-run-v1",
        "bakeoff_id": BAKEOFF_ID,
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "input_fingerprint": fingerprint,
        "narration_direction_sha256": text_sha256(NARRATION_DIRECTION),
        "prepared_at": utc_now(),
        "updated_at": utc_now(),
        "status": "prepared",
        "passages": passages,
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "jobs": jobs,
    }
    atomic_json(state_path, state)
    atomic_text(root / "NARRATION_DIRECTION.txt", NARRATION_DIRECTION + "\n")
    return state


def _write_wav(path: Path, audio: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with wave.open(str(temp), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(audio)
    os.replace(temp, path)


def _gemini_audio_bytes(response: Any) -> bytes:
    try:
        data = response.candidates[0].content.parts[0].inline_data.data
    except (AttributeError, IndexError, TypeError) as exc:
        raise BakeoffError("Gemini TTS response did not contain inline audio") from exc
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except ValueError as exc:
            raise BakeoffError("Gemini TTS returned malformed base64 audio") from exc
    raise BakeoffError("Gemini TTS returned an unsupported audio representation")


def _generate_gemini(
    candidate: CandidateSpec,
    text: str,
    output_path: Path,
) -> dict[str, Any]:
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise BakeoffError("Missing GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = f'{NARRATION_DIRECTION}\n\nText to read:\n"""{text}"""'
    response = client.models.generate_content(
        model=candidate.model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=candidate.voice_id
                    )
                )
            ),
        ),
    )
    audio = _gemini_audio_bytes(response)
    if len(audio) < 1024:
        raise BakeoffError("Gemini TTS returned implausibly small audio")
    _write_wav(output_path, audio)
    usage = getattr(response, "usage_metadata", None)
    return {
        "bytes": output_path.stat().st_size,
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


def _generate_elevenlabs(
    candidate: CandidateSpec,
    text: str,
    output_path: Path,
) -> dict[str, Any]:
    try:
        synthesize_text(
            text=text,
            voice_id=candidate.voice_id,
            output_path=output_path,
            model_id=candidate.model_id,
            output_format=candidate.source_format,
            seed=240109,
            request_timeout_seconds=600,
        )
    except ElevenLabsError as exc:
        raise BakeoffError(str(exc)) from exc
    if output_path.stat().st_size < 1024:
        raise BakeoffError("ElevenLabs returned implausibly small audio")
    return {"bytes": output_path.stat().st_size, "characters": len(text)}


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    format_data = payload.get("format") or {}
    duration = float(format_data.get("duration") or 0)
    if duration <= 1.0:
        raise BakeoffError(f"Audio duration is implausibly short: {path}")
    return {
        "duration_seconds": round(duration, 3),
        "bytes": int(format_data.get("size") or path.stat().st_size),
        "bit_rate": int(format_data.get("bit_rate") or 0),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def _normalize_audio(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp.mp3")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-af",
        "loudnorm=I=-18:TP=-1.5:LRA=11",
        "-ar",
        "44100",
        "-ac",
        "1",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(temp),
    ]
    subprocess.run(command, check=True, capture_output=True)
    os.replace(temp, destination)


def run_bakeoff(
    root: Path = DEFAULT_BAKEOFF_ROOT,
    *,
    max_attempts: int = 2,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    state = prepare_bakeoff(root)
    passages = {item["passage_id"]: item for item in state["passages"]}
    candidates = {
        item["candidate_id"]: CandidateSpec(**item) for item in state["candidates"]
    }
    state["status"] = "running"
    atomic_json(root / "RUN.json", state)

    for job in state["jobs"]:
        if job["status"] == "complete":
            continue
        candidate = candidates[job["candidate_id"]]
        passage = passages[job["passage_id"]]
        output_path = Path(job["raw_path"])
        for _ in range(max_attempts - int(job["attempts"])):
            job["attempts"] += 1
            job["last_error"] = None
            started = time.monotonic()
            try:
                if candidate.provider == "elevenlabs":
                    provider_usage = _generate_elevenlabs(
                        candidate, passage["text"], output_path
                    )
                elif candidate.provider == "gemini":
                    provider_usage = _generate_gemini(
                        candidate, passage["text"], output_path
                    )
                else:
                    raise BakeoffError(f"Unknown provider: {candidate.provider}")
                probe = _ffprobe(output_path)
                job.update(
                    {
                        "status": "complete",
                        "completed_at": utc_now(),
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "raw_sha256": file_sha256(output_path),
                        "provider_usage": provider_usage,
                        "probe": probe,
                    }
                )
                break
            except Exception as exc:  # Provider SDKs expose several error families.
                job["status"] = "failed"
                job["last_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                state["updated_at"] = utc_now()
                atomic_json(root / "RUN.json", state)
        if job["status"] != "complete":
            state["status"] = "blocked"
            state["updated_at"] = utc_now()
            atomic_json(root / "RUN.json", state)
            raise BakeoffError(f"Generation failed for {job['job_id']}: {job['last_error']}")

    state["status"] = "generated"
    state["updated_at"] = utc_now()
    atomic_json(root / "RUN.json", state)
    package_blind_bakeoff(root)
    return _read_object(root / "RUN.json")


def _blind_key(root: Path, candidate_ids: list[str]) -> dict[str, str]:
    path = root / "PRIVATE_BLIND_KEY.json"
    if path.exists():
        payload = _read_object(path)
        mapping = payload.get("candidate_to_code")
        if not isinstance(mapping, dict) or set(mapping) != set(candidate_ids):
            raise BakeoffError("Existing blind key does not match candidate set")
        return {str(key): str(value) for key, value in mapping.items()}
    codes = [f"Voice {chr(65 + index)}" for index in range(len(candidate_ids))]
    secrets.SystemRandom().shuffle(codes)
    mapping = dict(zip(candidate_ids, codes, strict=True))
    atomic_json(
        path,
        {
            "version": "tts-bakeoff-blind-key-v1",
            "created_at": utc_now(),
            "candidate_to_code": mapping,
        },
    )
    os.chmod(path, 0o600)
    return mapping


def _listening_html(blind_manifest: dict[str, Any]) -> str:
    passages_json = json.dumps(blind_manifest["passages"], ensure_ascii=False)
    voices_json = json.dumps(blind_manifest["voice_codes"], ensure_ascii=False)
    clips_json = json.dumps(blind_manifest["clips"], ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quran Narration Bakeoff</title>
<style>
:root {{ color-scheme: light; --ink:#171717; --muted:#666; --line:#d8d8d8; --accent:#a72525; --soft:#f4f5f2; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#fff; color:var(--ink); font-family:Arial,Helvetica,sans-serif; line-height:1.45; }}
header {{ border-bottom:1px solid var(--line); padding:32px max(20px,calc((100vw - 1180px)/2)); }}
h1 {{ margin:0 0 8px; font-family:Georgia,serif; font-size:32px; font-weight:600; }}
header p {{ margin:0; color:var(--muted); max-width:820px; }}
main {{ width:min(1180px,calc(100% - 40px)); margin:0 auto; padding:28px 0 60px; }}
.instructions {{ padding:16px 0 24px; border-bottom:1px solid var(--line); }}
.passage {{ padding:28px 0 34px; border-bottom:1px solid var(--line); }}
.passage h2 {{ margin:0 0 4px; font-family:Georgia,serif; font-size:24px; }}
.meta {{ color:var(--accent); font-size:13px; font-weight:700; text-transform:uppercase; }}
.transcript {{ max-width:900px; margin:14px 0 20px; font-family:Georgia,serif; font-size:18px; }}
.voices {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.voice {{ border:1px solid var(--line); border-radius:6px; padding:14px; min-width:0; }}
.voice h3 {{ margin:0 0 10px; font-size:16px; }}
audio {{ display:block; width:100%; height:40px; margin-bottom:12px; }}
.scores {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; }}
label {{ display:block; color:var(--muted); font-size:12px; }}
select {{ width:100%; margin-top:3px; padding:7px 5px; border:1px solid var(--line); border-radius:4px; background:white; }}
textarea {{ width:100%; min-height:58px; margin-top:10px; padding:8px; border:1px solid var(--line); border-radius:4px; resize:vertical; }}
.actions {{ position:sticky; bottom:0; display:flex; justify-content:space-between; gap:12px; align-items:center; padding:14px 0; background:rgba(255,255,255,.96); border-top:1px solid var(--line); }}
button {{ border:0; border-radius:6px; padding:11px 16px; background:var(--ink); color:white; font-weight:700; cursor:pointer; }}
#status {{ color:var(--muted); font-size:13px; }}
@media (max-width:760px) {{ .voices {{ grid-template-columns:1fr; }} .scores {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} h1 {{ font-size:27px; }} }}
</style>
</head>
<body>
<header><h1>Quran Narration Bakeoff</h1><p>Listen with headphones. Score what you hear, not what you expect from a provider. All clips use the exact same approved text and have been level-matched.</p></header>
<main>
<section class="instructions"><strong>Scale:</strong> 1 is poor, 3 is acceptable, 5 is exceptional. Penalize any omitted, repeated, or invented word under fidelity.</section>
<div id="app"></div>
<div class="actions"><span id="status">Scores stay in this browser until exported.</span><button id="export">Export scores</button></div>
</main>
<script>
const passages={passages_json};
const voices={voices_json};
const clips={clips_json};
const metrics=['Naturalness','Restraint','Pronunciation','Pacing','Fidelity'];
const storageKey='quranTtsBakeoffScores::{BAKEOFF_ID}';
const saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}');
const esc=value=>String(value).replace(/[&<>"']/g,char=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[char]));
const app=document.getElementById('app');
for (const passage of passages) {{
  const section=document.createElement('section'); section.className='passage';
  section.innerHTML=`<div class="meta">${{esc(passage.start_ref)}}-${{esc(passage.end_ref)}} · ${{esc(passage.register)}}</div><h2>${{esc(passage.label)}}</h2><p class="transcript">${{esc(passage.text)}}</p><div class="voices"></div>`;
  const grid=section.querySelector('.voices');
  for (const voice of voices) {{
    const clip=clips.find(x=>x.passage_id===passage.passage_id&&x.voice_code===voice);
    const key=`${{passage.passage_id}}::${{voice}}`; const prior=saved[key]||{{}};
    const box=document.createElement('div'); box.className='voice';
    box.innerHTML=`<h3>${{esc(voice)}}</h3><audio controls preload="metadata" src="${{encodeURI(clip.file)}}"></audio><div class="scores"></div><textarea placeholder="Specific words, pauses, or artifacts">${{esc(prior.notes||'')}}</textarea>`;
    const scores=box.querySelector('.scores');
    for (const metric of metrics) {{
      const label=document.createElement('label'); label.textContent=metric;
      const select=document.createElement('select'); select.dataset.metric=metric.toLowerCase();
      select.innerHTML='<option value="">-</option>'+[1,2,3,4,5].map(n=>`<option value="${{n}}">${{n}}</option>`).join('');
      select.value=prior[metric.toLowerCase()]||''; label.appendChild(select); scores.appendChild(label);
    }}
    const save=()=>{{ const value={{notes:box.querySelector('textarea').value}}; box.querySelectorAll('select').forEach(s=>value[s.dataset.metric]=s.value?Number(s.value):null); saved[key]=value; localStorage.setItem(storageKey,JSON.stringify(saved)); document.getElementById('status').textContent='Saved locally.'; }};
    box.addEventListener('change',save); box.querySelector('textarea').addEventListener('input',save); grid.appendChild(box);
  }}
  app.appendChild(section);
}}
document.getElementById('export').onclick=()=>{{ const payload={{bakeoff_id:'{BAKEOFF_ID}',exported_at:new Date().toISOString(),scores:saved}}; const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='quran-tts-bakeoff-scores.json'; a.click(); URL.revokeObjectURL(a.href); }};
</script>
</body>
</html>
"""


def package_blind_bakeoff(root: Path = DEFAULT_BAKEOFF_ROOT) -> dict[str, Any]:
    state = _read_object(root / "RUN.json")
    incomplete = [job["job_id"] for job in state["jobs"] if job["status"] != "complete"]
    if incomplete:
        raise BakeoffError(f"Cannot blind an incomplete bakeoff: {incomplete}")
    candidate_ids = [item["candidate_id"] for item in state["candidates"]]
    mapping = _blind_key(root, candidate_ids)
    blind_dir = root / "blind"
    clips: list[dict[str, Any]] = []
    for job in state["jobs"]:
        code = mapping[job["candidate_id"]]
        code_slug = code.lower().replace(" ", "-")
        destination = blind_dir / "clips" / f"{job['passage_id']}--{code_slug}.mp3"
        source = Path(job["raw_path"])
        _normalize_audio(source, destination)
        probe = _ffprobe(destination)
        clips.append(
            {
                "passage_id": job["passage_id"],
                "voice_code": code,
                "file": str(destination.relative_to(blind_dir)),
                "sha256": file_sha256(destination),
                "probe": probe,
            }
        )
    manifest = {
        "version": "tts-bakeoff-blind-package-v1",
        "bakeoff_id": BAKEOFF_ID,
        "release_version": RELEASE_VERSION,
        "final_text_sha256": EXPECTED_FINAL_TEXT_SHA256,
        "created_at": utc_now(),
        "voice_codes": sorted(mapping.values()),
        "passages": state["passages"],
        "clips": sorted(clips, key=lambda item: (item["passage_id"], item["voice_code"])),
        "normalization": "mono MP3, 44.1 kHz, 192 kbps, -18 LUFS target, -1.5 dBTP",
    }
    atomic_json(blind_dir / "BLIND_MANIFEST.json", manifest)
    atomic_text(blind_dir / "listen.html", _listening_html(manifest))
    state["status"] = "ready_for_blind_review"
    state["blind_package"] = str(blind_dir)
    state["updated_at"] = utc_now()
    atomic_json(root / "RUN.json", state)
    return manifest


def bakeoff_status(root: Path = DEFAULT_BAKEOFF_ROOT) -> dict[str, Any]:
    state = prepare_bakeoff(root)
    counts: dict[str, int] = {}
    for job in state["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    generated_seconds = sum(
        float(job.get("probe", {}).get("duration_seconds") or 0)
        for job in state["jobs"]
    )
    return {
        "bakeoff_id": state["bakeoff_id"],
        "status": state["status"],
        "jobs": counts,
        "generated_duration_seconds": round(generated_seconds, 3),
        "root": str(root),
        "listening_page": str(root / "blind" / "listen.html"),
    }
