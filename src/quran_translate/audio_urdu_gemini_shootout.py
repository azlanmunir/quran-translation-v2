"""Blind Charon shootout: Gemini 2.5 Pro TTS versus 3.1 Flash TTS."""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .audio_urdu_bakeoff import (
    DEFAULT_ROOT as URDU_BAKEOFF_ROOT,
    GEMINI_DIRECTION,
    PASSAGES,
    Candidate,
    UrduBakeoffError,
    _generate_gemini,
    _normalize,
    _probe,
)
from .config import OUTPUT_DIR, file_sha256, text_sha256
from .db import utc_now
from .production_packets import atomic_json, atomic_text


SHOOTOUT_ID = "quran-urdu-gemini-model-shootout-20260818-v1"
DEFAULT_ROOT = OUTPUT_DIR / "audio" / "bakeoffs" / SHOOTOUT_ID

MODELS = (
    Candidate(
        "gemini-25-pro-charon",
        "gemini",
        "gemini-2.5-pro-preview-tts",
        "Charon",
        "Gemini 2.5 Pro TTS / Charon",
        "ur",
        ".wav",
    ),
    Candidate(
        "gemini-31-flash-charon",
        "gemini",
        "gemini-3.1-flash-tts-preview",
        "Charon",
        "Gemini 3.1 Flash TTS / Charon",
        "ur",
        ".wav",
    ),
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UrduBakeoffError(f"Expected JSON object: {path}")
    return payload


def _old_jobs() -> dict[str, dict[str, Any]]:
    state = _read(URDU_BAKEOFF_ROOT / "RUN.json")
    jobs = {
        job["passage_id"]: job
        for job in state["jobs"]
        if job["candidate_id"] == "gemini-charon" and job["status"] == "complete"
    }
    if set(jobs) != {passage.passage_id for passage in PASSAGES}:
        raise UrduBakeoffError("The completed Gemini 2.5 Urdu baseline is incomplete")
    return jobs


def _fingerprint() -> str:
    old_jobs = _old_jobs()
    payload = {
        "version": "urdu-gemini-model-shootout-v1",
        "direction": GEMINI_DIRECTION,
        "passages": [asdict(passage) for passage in PASSAGES],
        "models": [asdict(model) for model in MODELS],
        "baseline_hashes": {
            passage_id: job["raw_sha256"] for passage_id, job in old_jobs.items()
        },
    }
    return text_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def prepare(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    state_path = root / "RUN.json"
    fingerprint = _fingerprint()
    if state_path.exists():
        state = _read(state_path)
        if state.get("input_fingerprint") != fingerprint:
            raise UrduBakeoffError("Shootout inputs changed; refusing a mixed resume")
        return state
    state = {
        "version": "urdu-gemini-model-shootout-run-v1",
        "shootout_id": SHOOTOUT_ID,
        "input_fingerprint": fingerprint,
        "status": "prepared",
        "prepared_at": utc_now(),
        "updated_at": utc_now(),
        "passages": [asdict(passage) for passage in PASSAGES],
        "models": [asdict(model) for model in MODELS],
        "jobs": [
            {
                "job_id": f"gemini-31-flash-charon--{passage.passage_id}",
                "candidate_id": "gemini-31-flash-charon",
                "passage_id": passage.passage_id,
                "raw_path": str(root / "raw" / f"{passage.passage_id}.wav"),
                "status": "pending",
                "attempts": 0,
                "last_error": None,
            }
            for passage in PASSAGES
        ],
    }
    atomic_json(state_path, state)
    return state


def _mapping(root: Path) -> dict[str, str]:
    path = root / "PRIVATE_BLIND_KEY.json"
    ids = [model.candidate_id for model in MODELS]
    if path.exists():
        mapping = _read(path).get("candidate_to_code")
        if not isinstance(mapping, dict) or set(mapping) != set(ids):
            raise UrduBakeoffError("Blind key does not match model set")
        return {str(key): str(value) for key, value in mapping.items()}
    codes = ["Version A", "Version B"]
    secrets.SystemRandom().shuffle(codes)
    mapping = dict(zip(ids, codes, strict=True))
    atomic_json(path, {"created_at": utc_now(), "candidate_to_code": mapping})
    os.chmod(path, 0o600)
    return mapping


def _html(manifest: dict[str, Any]) -> str:
    data = json.dumps(manifest, ensure_ascii=False)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Charon Urdu Model Shootout</title>
<style>*{{box-sizing:border-box}}body{{margin:0;color:#171717;font-family:Arial,sans-serif;line-height:1.45}}header,main{{width:min(1080px,calc(100% - 36px));margin:auto}}header{{padding:30px 0 22px;border-bottom:1px solid #d6d6d2}}h1{{font:600 31px Georgia,serif;margin:0 0 7px}}header p{{color:#666;max-width:830px}}section{{padding:26px 0;border-bottom:1px solid #d6d6d2}}h2{{font:600 23px Georgia,serif;margin:0}}.urdu{{direction:rtl;text-align:right;font-family:"Noto Nastaliq Urdu","Noto Naskh Arabic",serif;font-size:22px;line-height:2}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.clip{{border:1px solid #d6d6d2;border-radius:6px;padding:14px}}h3{{margin:0 0 9px}}audio{{width:100%}}.scores{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:10px}}label{{font-size:11px;color:#666}}select,textarea{{width:100%;border:1px solid #d6d6d2;border-radius:4px;background:white}}select{{padding:6px}}textarea{{min-height:54px;margin-top:9px;padding:7px}}.fatal{{display:flex;gap:7px;align-items:center;color:#9f2525;margin-top:8px}}.fatal input{{width:auto}}.footer{{position:sticky;bottom:0;background:white;border-top:1px solid #d6d6d2;padding:12px 0;display:flex;justify-content:space-between}}button{{border:0;border-radius:6px;background:#171717;color:white;padding:10px 15px;font-weight:700}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><header><h1>Charon Urdu Model Shootout</h1><p>Same Charon voice, same Urdu, same direction, same loudness. Only the Gemini TTS model generation changed. Score mistakes before beauty.</p></header><main><div id="app"></div><div class="footer"><span id="status">Scores save locally.</span><button id="export">Export</button></div></main>
<script>const m={data};const metrics=['Naturalness','Urdu accuracy','Pacing','Gravitas','Long-form comfort'];const key='urduGeminiShootout::{SHOOTOUT_ID}';const saved=JSON.parse(localStorage.getItem(key)||'{{}}');const app=document.getElementById('app');const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));for(const p of m.passages){{const section=document.createElement('section');section.innerHTML=`<h2>${{esc(p.label)}}</h2><p class="urdu" lang="ur">${{esc(p.text)}}</p><div class="grid"></div>`;const grid=section.querySelector('.grid');for(const version of m.version_codes){{const clip=m.clips.find(x=>x.passage_id===p.passage_id&&x.version_code===version);const id=`${{p.passage_id}}::${{version}}`;const prior=saved[id]||{{}};const box=document.createElement('div');box.className='clip';box.innerHTML=`<h3>${{version}}</h3><audio controls preload="metadata" src="${{encodeURI(clip.file)}}"></audio><div class="scores"></div><label class="fatal"><input type="checkbox" ${{prior.mistake?'checked':''}}> Pronunciation or text mistake</label><textarea placeholder="Name the exact word, cadence, or artifact">${{esc(prior.notes||'')}}</textarea>`;for(const metric of metrics){{const label=document.createElement('label');label.textContent=metric;const select=document.createElement('select');select.dataset.metric=metric.toLowerCase().replaceAll(' ','_');select.innerHTML='<option value="">-</option>'+[1,2,3,4,5].map(n=>`<option>${{n}}</option>`).join('');select.value=prior[select.dataset.metric]||'';label.appendChild(select);box.querySelector('.scores').appendChild(label)}}const save=()=>{{const v={{mistake:box.querySelector('input').checked,notes:box.querySelector('textarea').value}};box.querySelectorAll('select').forEach(s=>v[s.dataset.metric]=s.value?Number(s.value):null);saved[id]=v;localStorage.setItem(key,JSON.stringify(saved));document.getElementById('status').textContent='Saved locally.'}};box.addEventListener('change',save);box.querySelector('textarea').addEventListener('input',save);grid.appendChild(box)}}app.appendChild(section)}}document.getElementById('export').onclick=()=>{{const blob=new Blob([JSON.stringify({{shootout_id:'{SHOOTOUT_ID}',exported_at:new Date().toISOString(),scores:saved}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='charon-urdu-model-shootout.json';a.click();URL.revokeObjectURL(a.href)}};</script></body></html>"""


def package(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    state = _read(root / "RUN.json")
    if any(job["status"] != "complete" for job in state["jobs"]):
        raise UrduBakeoffError("Cannot package incomplete 3.1 clips")
    mapping = _mapping(root)
    old_jobs = _old_jobs()
    new_jobs = {job["passage_id"]: job for job in state["jobs"]}
    blind = root / "blind"
    clips = []
    for model in MODELS:
        code = mapping[model.candidate_id]
        for passage in PASSAGES:
            source = Path(
                old_jobs[passage.passage_id]["raw_path"]
                if model.candidate_id == "gemini-25-pro-charon"
                else new_jobs[passage.passage_id]["raw_path"]
            )
            destination = blind / "clips" / f"{passage.passage_id}--{code.lower().replace(' ', '-')}.mp3"
            _normalize(source, destination)
            clips.append(
                {
                    "passage_id": passage.passage_id,
                    "version_code": code,
                    "file": str(destination.relative_to(blind)),
                    "sha256": file_sha256(destination),
                    "probe": _probe(destination),
                }
            )
    manifest = {
        "version": "urdu-gemini-model-shootout-blind-v1",
        "shootout_id": SHOOTOUT_ID,
        "created_at": utc_now(),
        "version_codes": sorted(mapping.values()),
        "passages": state["passages"],
        "clips": sorted(clips, key=lambda item: (item["passage_id"], item["version_code"])),
    }
    atomic_json(blind / "BLIND_MANIFEST.json", manifest)
    atomic_text(blind / "listen.html", _html(manifest))

    comparison = {}
    for model in MODELS:
        jobs = (
            list(old_jobs.values())
            if model.candidate_id == "gemini-25-pro-charon"
            else list(new_jobs.values())
        )
        prompt_tokens = sum(int(job["provider_usage"].get("prompt_tokens") or 0) for job in jobs)
        output_tokens = sum(int(job["provider_usage"].get("output_tokens") or 0) for job in jobs)
        comparison[model.candidate_id] = {
            "model_id": model.model_id,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "duration_seconds": round(sum(float(job["probe"]["duration_seconds"]) for job in jobs), 3),
            "latency_seconds": round(sum(float(job.get("latency_seconds") or 0) for job in jobs), 3),
            "standard_paid_cost_usd": round(prompt_tokens / 1_000_000 + output_tokens * 20 / 1_000_000, 6),
            "batch_paid_cost_usd": round(prompt_tokens * 0.5 / 1_000_000 + output_tokens * 10 / 1_000_000, 6),
        }
    atomic_json(root / "PRIVATE_MODEL_COMPARISON.json", comparison)
    state["status"] = "ready_for_blind_review"
    state["updated_at"] = utc_now()
    atomic_json(root / "RUN.json", state)
    return manifest


def run(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    state = prepare(root)
    model = MODELS[1]
    passages = {passage.passage_id: passage for passage in PASSAGES}
    state["status"] = "running"
    atomic_json(root / "RUN.json", state)
    for job in state["jobs"]:
        if job["status"] == "complete":
            continue
        job["attempts"] += 1
        started = time.monotonic()
        try:
            usage = _generate_gemini(model, passages[job["passage_id"]], Path(job["raw_path"]))
            job.update(
                {
                    "status": "complete",
                    "provider_usage": usage,
                    "probe": _probe(Path(job["raw_path"])),
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "raw_sha256": file_sha256(Path(job["raw_path"])),
                    "last_error": None,
                }
            )
        except Exception as exc:
            job["status"] = "failed"
            job["last_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            state["updated_at"] = utc_now()
            atomic_json(root / "RUN.json", state)
        if job["status"] != "complete":
            state["status"] = "blocked"
            atomic_json(root / "RUN.json", state)
            raise UrduBakeoffError(f"3.1 generation failed: {job['last_error']}")
    package(root)
    return _read(root / "RUN.json")


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "root": str(DEFAULT_ROOT)}, indent=2))
