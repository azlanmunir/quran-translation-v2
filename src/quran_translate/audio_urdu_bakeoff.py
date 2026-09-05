"""Blind, resumable Urdu TTS bakeoff for Quran audiobook narration."""

from __future__ import annotations

import argparse
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


BAKEOFF_ID = "quran-urdu-tts-20260818-v1"
DEFAULT_ROOT = OUTPUT_DIR / "audio" / "bakeoffs" / BAKEOFF_ID

GEMINI_DIRECTION = (
    "Read only the Urdu text inside <text>. Do not read these instructions or the tags. "
    "Use natural, mature Pakistani Urdu audiobook narration with measured pacing and quiet "
    "authority. Preserve Urdu pronunciation, izafat, and Quranic Arabic names. Do not drift "
    "into Hindi diction, theatrical recitation, or a promotional voice. Read every word exactly."
)


@dataclass(frozen=True)
class Passage:
    passage_id: str
    label: str
    register: str
    text: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    provider: str
    model_id: str
    voice_id: str
    private_label: str
    locale: str
    extension: str


PASSAGES = (
    Passage(
        "modern-prose",
        "Modern spoken Urdu",
        "clarity, flow, and unmistakably Urdu diction",
        (
            "یہ ترجمہ اُس سننے والے کے لیے ہے جو قرآن کا مفہوم صاف، رواں اور آج کی اردو میں "
            "سمجھنا چاہتا ہے، مگر اصل عربی کی قوت، تہہ داری، اور جہاں ضروری ہو وہاں اس کے ابہام "
            "کو کھونا نہیں چاہتا۔ آواز میں وقار ہو، لیکن بناوٹ نہ ہو؛ ٹھہراؤ ہو، لیکن سستی نہ ہو۔"
        ),
    ),
    Passage(
        "consolation",
        "Consolation and prayer",
        "tenderness, restraint, and sustained phrasing",
        (
            "صبح کی روشنی کی قسم، اور رات کی جب وہ چھا جائے: تمہارے رب نے نہ تمہیں چھوڑا ہے، "
            "نہ تم سے بیزار ہوا ہے۔ آنے والا وقت تمہارے لیے گزرے ہوئے وقت سے بہتر ہوگا۔ وہ تمہیں "
            "اتنا عطا کرے گا کہ تم راضی ہو جاؤ گے۔ پس یتیم پر سختی نہ کرو، سوال کرنے والے کو نہ جھڑکو، "
            "اور اپنے رب کی نعمت کا ذکر کرتے رہو۔"
        ),
    ),
    Passage(
        "force-and-names",
        "Force, names, and Quranic terms",
        "eschatological force, consonants, izafat, and Arabic names",
        (
            "جب سورج لپیٹ دیا جائے گا، جب ستارے بے نور ہو کر جھڑ پڑیں گے، اور جب پہاڑ چلائے جائیں گے، "
            "تب ہر جان جان لے گی کہ وہ کیا لے کر آئی ہے۔ موسیٰ، عیسیٰ، ابراہیم، یوسف اور مریم نے اپنے رب "
            "پر بھروسا رکھا۔ صراطِ مستقیم، آخرت، تقویٰ، مغفرت اور ذمّے داری جیسے لفظ صاف ادا ہوں؛ ق، غ، "
            "خ، ع، ڑ، ٹ اور ڈ کی آوازیں نگلی نہ جائیں۔"
        ),
    ),
)

CANDIDATES = (
    Candidate(
        "azure-asad",
        "edge",
        "azure-neural",
        "ur-PK-AsadNeural",
        "Azure Asad",
        "ur-PK",
        ".mp3",
    ),
    Candidate(
        "azure-salman",
        "edge",
        "azure-neural",
        "ur-IN-SalmanNeural",
        "Azure Salman",
        "ur-IN",
        ".mp3",
    ),
    Candidate(
        "eleven-aakif",
        "elevenlabs",
        "eleven_v3",
        "9TT97IL5ILrM6ph1Ougs",
        "ElevenLabs Aakif",
        "ur",
        ".mp3",
    ),
    Candidate(
        "eleven-habban",
        "elevenlabs",
        "eleven_v3",
        "hV1InkNCNhrjdXJO4OkG",
        "ElevenLabs Habban",
        "ur",
        ".mp3",
    ),
    Candidate(
        "eleven-ghufran",
        "elevenlabs",
        "eleven_v3",
        "rk17WPSrYqFXDYMAKFjN",
        "ElevenLabs Ghufran",
        "ur",
        ".mp3",
    ),
    Candidate(
        "gemini-charon",
        "gemini",
        "gemini-2.5-pro-preview-tts",
        "Charon",
        "Gemini Charon",
        "ur",
        ".wav",
    ),
)


class UrduBakeoffError(RuntimeError):
    """Raised when generation or bakeoff integrity fails."""


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UrduBakeoffError(f"Expected a JSON object: {path}")
    return payload


def _fingerprint() -> str:
    payload = {
        "version": "urdu-tts-bakeoff-v1",
        "direction": GEMINI_DIRECTION,
        "passages": [asdict(item) for item in PASSAGES],
        "candidates": [asdict(item) for item in CANDIDATES],
    }
    return text_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def prepare(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    state_path = root / "RUN.json"
    fingerprint = _fingerprint()
    if state_path.exists():
        state = _read_object(state_path)
        if state.get("input_fingerprint") != fingerprint:
            raise UrduBakeoffError("Bakeoff inputs changed; refusing a mixed-version resume")
        return state

    jobs = []
    for candidate in CANDIDATES:
        for passage in PASSAGES:
            jobs.append(
                {
                    "job_id": f"{candidate.candidate_id}--{passage.passage_id}",
                    "candidate_id": candidate.candidate_id,
                    "passage_id": passage.passage_id,
                    "raw_path": str(
                        root
                        / "raw"
                        / candidate.candidate_id
                        / f"{passage.passage_id}{candidate.extension}"
                    ),
                    "status": "pending",
                    "attempts": 0,
                    "last_error": None,
                }
            )
    state = {
        "version": "urdu-tts-bakeoff-run-v1",
        "bakeoff_id": BAKEOFF_ID,
        "input_fingerprint": fingerprint,
        "prepared_at": utc_now(),
        "updated_at": utc_now(),
        "status": "prepared",
        "passages": [asdict(item) for item in PASSAGES],
        "candidates": [asdict(item) for item in CANDIDATES],
        "jobs": jobs,
    }
    atomic_json(state_path, state)
    atomic_text(root / "GEMINI_DIRECTION.txt", GEMINI_DIRECTION + "\n")
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


def _gemini_audio(response: Any) -> bytes:
    try:
        data = response.candidates[0].content.parts[0].inline_data.data
    except (AttributeError, IndexError, TypeError) as exc:
        raise UrduBakeoffError("Gemini response did not contain audio") from exc
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    raise UrduBakeoffError("Gemini returned unsupported audio data")


def _generate_gemini(candidate: Candidate, passage: Passage, output: Path) -> dict[str, Any]:
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise UrduBakeoffError("Missing GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=candidate.model_id,
        contents=f"{GEMINI_DIRECTION}\n\n<text>{passage.text}</text>",
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
    _write_wav(output, _gemini_audio(response))
    usage = getattr(response, "usage_metadata", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


def _generate_eleven(candidate: Candidate, passage: Passage, output: Path) -> dict[str, Any]:
    try:
        synthesize_text(
            text=passage.text,
            voice_id=candidate.voice_id,
            output_path=output,
            model_id=candidate.model_id,
            output_format="mp3_44100_128",
            language_code="ur",
            seed=240109,
            request_timeout_seconds=600,
        )
    except ElevenLabsError as exc:
        raise UrduBakeoffError(str(exc)) from exc
    return {"characters": len(passage.text)}


def _generate_edge(candidate: Candidate, passage: Passage, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(Path(os.sys.executable).with_name("edge-tts")),
        "--voice",
        candidate.voice_id,
        "--text",
        passage.text,
        "--write-media",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if completed.returncode:
        raise UrduBakeoffError(completed.stderr.strip() or "edge-tts failed")
    return {"characters": len(passage.text), "test_transport": "edge-tts"}


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = (payload.get("streams") or [{}])[0]
    format_data = payload.get("format") or {}
    duration = float(format_data.get("duration") or 0)
    if duration < 3:
        raise UrduBakeoffError(f"Implausibly short audio: {path}")
    return {
        "duration_seconds": round(duration, 3),
        "bytes": int(format_data.get("size") or path.stat().st_size),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def _normalize(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp.mp3")
    subprocess.run(
        [
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
        ],
        check=True,
        capture_output=True,
    )
    os.replace(temp, destination)


def _blind_mapping(root: Path, candidate_ids: list[str]) -> dict[str, str]:
    path = root / "PRIVATE_BLIND_KEY.json"
    if path.exists():
        mapping = _read_object(path).get("candidate_to_code")
        if not isinstance(mapping, dict) or set(mapping) != set(candidate_ids):
            raise UrduBakeoffError("Blind key does not match the candidate set")
        return {str(key): str(value) for key, value in mapping.items()}
    codes = [f"Voice {chr(65 + index)}" for index in range(len(candidate_ids))]
    secrets.SystemRandom().shuffle(codes)
    mapping = dict(zip(candidate_ids, codes, strict=True))
    atomic_json(path, {"created_at": utc_now(), "candidate_to_code": mapping})
    os.chmod(path, 0o600)
    return mapping


def _html(manifest: dict[str, Any]) -> str:
    data = json.dumps(manifest, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Urdu Quran Voice Bakeoff</title>
<style>
:root{{--ink:#171717;--muted:#686868;--line:#d6d6d2;--red:#9f2525;--soft:#f5f5f2}}*{{box-sizing:border-box}}
body{{margin:0;color:var(--ink);font-family:Arial,sans-serif;line-height:1.45}}header,main{{width:min(1160px,calc(100% - 36px));margin:auto}}
header{{padding:30px 0 22px;border-bottom:1px solid var(--line)}}h1{{font:600 32px Georgia,serif;margin:0 0 8px}}header p{{max-width:850px;color:var(--muted);margin:0}}
.rules{{padding:18px 0;border-bottom:1px solid var(--line)}}.passage{{padding:27px 0;border-bottom:1px solid var(--line)}}h2{{font:600 23px Georgia,serif;margin:0}}
.register{{color:var(--red);font-size:13px;font-weight:700;text-transform:uppercase}}.urdu{{direction:rtl;text-align:right;font-family:"Noto Nastaliq Urdu","Noto Naskh Arabic",serif;font-size:22px;line-height:2;margin:12px 0 20px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}}.voice{{border:1px solid var(--line);border-radius:6px;padding:13px}}h3{{margin:0 0 9px;font-size:16px}}audio{{width:100%;height:40px}}
.scores{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:10px}}label{{font-size:11px;color:var(--muted)}}select,textarea{{width:100%;border:1px solid var(--line);border-radius:4px;background:#fff}}select{{padding:6px;margin-top:2px}}textarea{{min-height:52px;margin-top:9px;padding:7px}}
.fatal{{display:flex;align-items:center;gap:7px;color:var(--red);font-size:12px;margin-top:8px}}.fatal input{{width:auto}}.footer{{position:sticky;bottom:0;background:#fff;border-top:1px solid var(--line);padding:12px 0;display:flex;justify-content:space-between;align-items:center}}button{{border:0;border-radius:6px;background:var(--ink);color:#fff;padding:10px 15px;font-weight:700}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}.scores{{grid-template-columns:repeat(2,minmax(0,1fr))}}h1{{font-size:27px}}}}
</style></head><body><header><h1>Urdu Quran Voice Bakeoff</h1><p>All voices are male and provider identities are hidden. Listen for Pakistani Urdu, not merely a pleasant voice reading Urdu script.</p></header><main>
<section class="rules"><strong>Score 1-5.</strong> “Urdu authenticity” and “Quranic terms” are gates. Mark <strong>wrong-language drift</strong> for Hindi diction, non-Urdu phonology, or an accent that would be unacceptable for a full Urdu Quran audiobook.</section><div id="app"></div>
<div class="footer"><span id="status">Scores are saved in this browser.</span><button id="export">Export scores</button></div></main>
<script>
const manifest={data};const metrics=['Naturalness','Urdu authenticity','Quranic terms','Pacing','Gravitas','Long-form comfort'];const key='urduTtsBakeoff::{BAKEOFF_ID}';const saved=JSON.parse(localStorage.getItem(key)||'{{}}');const app=document.getElementById('app');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
for(const passage of manifest.passages){{const section=document.createElement('section');section.className='passage';section.innerHTML=`<div class="register">${{esc(passage.register)}}</div><h2>${{esc(passage.label)}}</h2><p class="urdu" lang="ur">${{esc(passage.text)}}</p><div class="grid"></div>`;const grid=section.querySelector('.grid');for(const voice of manifest.voice_codes){{const clip=manifest.clips.find(x=>x.passage_id===passage.passage_id&&x.voice_code===voice);const id=`${{passage.passage_id}}::${{voice}}`;const prior=saved[id]||{{}};const box=document.createElement('div');box.className='voice';box.innerHTML=`<h3>${{voice}}</h3><audio controls preload="metadata" src="${{encodeURI(clip.file)}}"></audio><div class="scores"></div><label class="fatal"><input type="checkbox" ${{prior.wrong_language_drift?'checked':''}}> Wrong-language / Hindi drift</label><textarea placeholder="Which word, consonant, pause, or cadence made the difference?">${{esc(prior.notes||'')}}</textarea>`;const scores=box.querySelector('.scores');for(const metric of metrics){{const label=document.createElement('label');label.textContent=metric;const select=document.createElement('select');select.dataset.metric=metric.toLowerCase().replaceAll(' ','_');select.innerHTML='<option value="">-</option>'+[1,2,3,4,5].map(n=>`<option>${{n}}</option>`).join('');select.value=prior[select.dataset.metric]||'';label.appendChild(select);scores.appendChild(label)}}const save=()=>{{const value={{notes:box.querySelector('textarea').value,wrong_language_drift:box.querySelector('input').checked}};box.querySelectorAll('select').forEach(s=>value[s.dataset.metric]=s.value?Number(s.value):null);saved[id]=value;localStorage.setItem(key,JSON.stringify(saved));document.getElementById('status').textContent='Saved locally.'}};box.addEventListener('change',save);box.querySelector('textarea').addEventListener('input',save);grid.appendChild(box)}}app.appendChild(section)}}
document.getElementById('export').onclick=()=>{{const blob=new Blob([JSON.stringify({{bakeoff_id:'{BAKEOFF_ID}',exported_at:new Date().toISOString(),scores:saved}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='urdu-tts-bakeoff-scores.json';a.click();URL.revokeObjectURL(a.href)}};
</script></body></html>"""


def package(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    state = _read_object(root / "RUN.json")
    incomplete = [job["job_id"] for job in state["jobs"] if job["status"] != "complete"]
    if incomplete:
        raise UrduBakeoffError(f"Cannot package incomplete jobs: {incomplete}")
    candidate_ids = [item["candidate_id"] for item in state["candidates"]]
    mapping = _blind_mapping(root, candidate_ids)
    blind_dir = root / "blind"
    clips = []
    for job in state["jobs"]:
        code = mapping[job["candidate_id"]]
        destination = blind_dir / "clips" / f"{job['passage_id']}--{code.lower().replace(' ', '-')}.mp3"
        _normalize(Path(job["raw_path"]), destination)
        clips.append(
            {
                "passage_id": job["passage_id"],
                "voice_code": code,
                "file": str(destination.relative_to(blind_dir)),
                "sha256": file_sha256(destination),
                "probe": _probe(destination),
            }
        )
    manifest = {
        "version": "urdu-tts-blind-package-v1",
        "bakeoff_id": BAKEOFF_ID,
        "created_at": utc_now(),
        "voice_codes": sorted(mapping.values()),
        "passages": state["passages"],
        "clips": sorted(clips, key=lambda item: (item["passage_id"], item["voice_code"])),
        "normalization": "mono MP3, 44.1 kHz, 192 kbps, -18 LUFS target",
    }
    atomic_json(blind_dir / "BLIND_MANIFEST.json", manifest)
    atomic_text(blind_dir / "listen.html", _html(manifest))
    state["status"] = "ready_for_blind_review"
    state["updated_at"] = utc_now()
    atomic_json(root / "RUN.json", state)
    return manifest


def run(root: Path = DEFAULT_ROOT, max_attempts: int = 2) -> dict[str, Any]:
    state = prepare(root)
    passages = {item["passage_id"]: Passage(**item) for item in state["passages"]}
    candidates = {item["candidate_id"]: Candidate(**item) for item in state["candidates"]}
    state["status"] = "running"
    atomic_json(root / "RUN.json", state)
    for job in state["jobs"]:
        if job["status"] == "complete":
            continue
        candidate = candidates[job["candidate_id"]]
        passage = passages[job["passage_id"]]
        output = Path(job["raw_path"])
        for _ in range(max_attempts - int(job["attempts"])):
            job["attempts"] += 1
            started = time.monotonic()
            try:
                if candidate.provider == "edge":
                    usage = _generate_edge(candidate, passage, output)
                elif candidate.provider == "elevenlabs":
                    usage = _generate_eleven(candidate, passage, output)
                elif candidate.provider == "gemini":
                    usage = _generate_gemini(candidate, passage, output)
                else:
                    raise UrduBakeoffError(f"Unknown provider: {candidate.provider}")
                job.update(
                    {
                        "status": "complete",
                        "completed_at": utc_now(),
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "provider_usage": usage,
                        "raw_sha256": file_sha256(output),
                        "probe": _probe(output),
                        "last_error": None,
                    }
                )
                break
            except Exception as exc:
                job["status"] = "failed"
                job["last_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                state["updated_at"] = utc_now()
                atomic_json(root / "RUN.json", state)
        if job["status"] != "complete":
            state["status"] = "blocked"
            atomic_json(root / "RUN.json", state)
            raise UrduBakeoffError(f"Generation failed: {job['job_id']}: {job['last_error']}")
    state["status"] = "generated"
    atomic_json(root / "RUN.json", state)
    package(root)
    return _read_object(root / "RUN.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    result = prepare(args.root) if args.prepare_only else run(args.root)
    print(json.dumps({"status": result["status"], "root": str(args.root)}, indent=2))


if __name__ == "__main__":
    main()
