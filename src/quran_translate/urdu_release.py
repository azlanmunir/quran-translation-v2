"""Freeze the completed Urdu translation into a versioned reader release."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pymupdf

from .config import DEFAULT_SOURCE_XML, OUTPUT_DIR, PROJECT_ROOT, WORK_DIR, file_sha256
from .db import utc_now
from .metadata import SURAHS
from .production_packets import atomic_json, atomic_text


RUN_ID = "quran-urdu-production-v1-20260818"
RELEASE_ID = "quran-urdu-v1.0.0-20260819"
LAYOUT_VERSION = "urdu-reader-layout-v3"
RUN_ROOT = WORK_DIR / "urdu-production-v1" / RUN_ID
RELEASE_ROOT = OUTPUT_DIR / "urdu" / "releases" / RELEASE_ID
SELECTION_PATH = PROJECT_ROOT / "data" / "evidence" / "urdu-tts-selection-v1.json"
DECISIONS_PATH = PROJECT_ROOT / "data" / "evidence" / "urdu-production-final-adjudications-v1.json"


class UrduReleaseError(RuntimeError):
    """Raised when release inputs or output integrity fail."""


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UrduReleaseError(f"Expected JSON object: {path}")
    return payload


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise UrduReleaseError(f"Expected JSON array: {path}")
    rows = [dict(row) for row in payload]
    expected = [
        (surah.number, ayah)
        for surah in SURAHS
        for ayah in range(1, surah.ayah_count + 1)
    ]
    actual = [(int(row["surah"]), int(row["ayah"])) for row in rows]
    if actual != expected:
        raise UrduReleaseError("Urdu rows do not match canonical 6,236-ayah order")
    if any(not str(row.get("urdu", "")).strip() for row in rows):
        raise UrduReleaseError("Urdu release contains an empty translation")
    return rows


def _surah_names() -> dict[int, str]:
    root = ET.parse(DEFAULT_SOURCE_XML).getroot()
    names = {
        int(node.attrib["index"]): str(node.attrib["name"])
        for node in root.findall("sura")
    }
    if set(names) != set(range(1, 115)):
        raise UrduReleaseError("Pinned source XML does not contain 114 surah names")
    return names


def _urdu_digits(value: int) -> str:
    return str(value).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def _reader_html(rows: list[dict[str, Any]], names: dict[int, str]) -> str:
    grouped: dict[int, list[dict[str, Any]]] = {number: [] for number in range(1, 115)}
    for row in rows:
        grouped[int(row["surah"])].append(row)
    sections: list[str] = []
    for surah in SURAHS:
        verses = "\n".join(
            '<div class="ayah"><span class="ayah-number">'
            f'﴿{_urdu_digits(int(row["ayah"]))}﴾</span> '
            f'{html.escape(str(row["urdu"]))}</div>'
            for row in grouped[surah.number]
        )
        sections.append(
            f'<section class="surah" id="surah-{surah.number}">'
            f'<h2>سورۃ {html.escape(names[surah.number])}</h2>'
            f'<p class="surah-meta">سورۃ {_urdu_digits(surah.number)} · '
            f'{html.escape(surah.transliteration)} · {_urdu_digits(surah.ayah_count)} آیات</p>'
            f'{verses}</section>'
        )
    return """<!doctype html>
<html lang="ur" dir="rtl"><head><meta charset="utf-8"><title>قرآن مجید - جدید اردو ترجمانی</title>
<style>
@page { size: 6in 9in; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; color: #171d1a; background: white; }
body { font-family: "Noto Nastaliq Urdu", "DecoType Naskh", serif; direction: rtl; }
.title-page { min-height: 7.7in; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; break-after: page; }
.kicker { color: #8a3c2e; font: 600 10pt Georgia, serif; text-transform: uppercase; }
h1 { margin: 0.22in 0 0.08in; font-size: 34pt; line-height: 1.7; color: #173c34; }
.subtitle { max-width: 4.5in; font-size: 15pt; line-height: 2.05; }
.author { margin-top: 0.44in; font-size: 13pt; }
.front-matter { break-after: page; padding-top: 0.18in; }
.front-matter h2 { margin-top: 0; }
.front-matter p { font-size: 11.2pt; line-height: 2.05; }
.surah { break-before: page; padding-top: 0.38in; }
.surah h2 { margin: 0 0 0.04in; text-align: center; color: #173c34; font-size: 22pt; line-height: 1.8; }
.surah-meta { margin: 0 0 0.24in; text-align: center; color: #8a3c2e; font: 9pt Georgia, serif; direction: rtl; }
.ayah { font-size: 12.7pt; line-height: 2.05; margin: 0 0 0.07in; text-align: right; orphans: 2; widows: 2; }
.ayah-number { color: #8a3c2e; font-family: "Noto Naskh Arabic", serif; font-size: 9.6pt; white-space: nowrap; }
</style></head><body>
<section class="title-page">
  <div class="kicker">Evidence-Audited Modern Urdu Translation</div>
  <h1>قرآن مجید</h1>
  <div class="subtitle">معنی پر مبنی، شواہد سے جانچی ہوئی جدید اردو ترجمانی</div>
  <div class="author">ازلان منیر</div>
</section>
<section class="front-matter">
  <h2>اس ترجمے کے بارے میں</h2>
  <p>یہ ترجمہ عربی متن کے معنی، سیاق اور لفظی قوت کو مقدم رکھتا ہے۔ جہاں اصل عبارت میں گنجائش یا ابہام ہے، وہاں غیر ضروری توضیح کو متن کے اندر شامل کرنے سے گریز کیا گیا ہے۔</p>
  <p>اس کام میں زبان ماڈلز سے مدد لی گئی اور متن کو متعدد آزاد تنقیدی، اصطلاحی، تکراری اور صوتی جانچوں سے گزارا گیا۔ یہ اردو ترجمانی عربی قرآن کا بدل نہیں؛ اصل حجت عربی متن ہی ہے۔</p>
  <p>نسخہ: v1.0.0 · 6,236 آیات · حتمی معیار جانچ: کامیاب</p>
</section>
""" + "\n".join(sections) + "\n</body></html>\n"


def _render_pdf(html_path: Path, pdf_path: Path) -> None:
    node = os.environ.get("CODEX_NODE") or str(
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    )
    modules = os.environ.get("NODE_PATH") or str(
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
    )
    script = PROJECT_ROOT / "scripts" / "render_urdu_pdf.mjs"
    if not Path(node).exists():
        found = shutil.which("node")
        if not found:
            raise UrduReleaseError("Node.js is required to render the Urdu PDF")
        node = found
    env = dict(os.environ)
    env["NODE_PATH"] = modules
    completed = subprocess.run(
        [node, str(script), str(html_path.resolve()), str(pdf_path.resolve())],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode:
        raise UrduReleaseError(completed.stderr.strip() or "Urdu PDF rendering failed")
    if not pdf_path.exists() or pdf_path.stat().st_size < 1_000_000:
        raise UrduReleaseError("Rendered Urdu PDF is missing or implausibly small")
    with pymupdf.open(pdf_path) as document:
        if document.page_count < 100:
            raise UrduReleaseError(
                f"Rendered Urdu PDF has implausibly few pages: {document.page_count}"
            )


def _fingerprint(inputs: dict[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in sorted(inputs.items())}


def build_release(root: Path = RELEASE_ROOT, *, render_pdf: bool = True) -> dict[str, Any]:
    complete_path = RUN_ROOT / "PRODUCTION_COMPLETE.json"
    qa_path = RUN_ROOT / "QA_REPORT.json"
    adjudication_path = RUN_ROOT / "FINAL_ADJUDICATION_COMPLETE.json"
    source_json = RUN_ROOT / "output" / "quran-urdu.json"
    source_text = RUN_ROOT / "output" / "quran-urdu.txt"
    review_queue = RUN_ROOT / "REVIEW_QUEUE.json"
    inputs = {
        "PRODUCTION_COMPLETE.json": complete_path,
        "QA_REPORT.json": qa_path,
        "FINAL_ADJUDICATION_COMPLETE.json": adjudication_path,
        "quran-urdu.json": source_json,
        "quran-urdu.txt": source_text,
        "REVIEW_QUEUE.json": review_queue,
        "final-adjudications.json": DECISIONS_PATH,
        "tts-selection.json": SELECTION_PATH,
    }
    missing = [name for name, path in inputs.items() if not path.exists()]
    if missing:
        raise UrduReleaseError(f"Missing release inputs: {missing}")
    complete = _read_object(complete_path)
    qa = _read_object(qa_path)
    adjudication = _read_object(adjudication_path)
    if complete.get("run_id") != RUN_ID or int(complete.get("ayahs", 0)) != 6236:
        raise UrduReleaseError("Production completion marker does not match the frozen run")
    if not qa.get("quality", {}).get("passed") or qa.get("unresolved"):
        raise UrduReleaseError("Refusing to release Urdu text that did not pass final QA")
    expected_hashes = adjudication.get("output_sha256", {})
    if expected_hashes.get("output/quran-urdu.json") != file_sha256(source_json):
        raise UrduReleaseError("Final Urdu JSON changed after adjudication")
    if expected_hashes.get("output/quran-urdu.txt") != file_sha256(source_text):
        raise UrduReleaseError("Final Urdu text changed after adjudication")
    rows = _read_rows(source_json)
    input_hashes = _fingerprint(inputs)
    manifest_path = root / "MANIFEST.json"
    rebuilding_layout = False
    if manifest_path.exists():
        existing = _read_object(manifest_path)
        if existing.get("input_sha256") != input_hashes:
            raise UrduReleaseError("Existing release has different frozen inputs")
        if existing.get("layout_version") == LAYOUT_VERSION:
            return existing
        rebuilding_layout = True
    if root.exists() and any(root.iterdir()) and not rebuilding_layout:
        raise UrduReleaseError(f"Refusing to mix a partial release at {root}")
    root.mkdir(parents=True, exist_ok=True)
    copies = {
        "quran-urdu.json": source_json,
        "quran-urdu.txt": source_text,
        "QA_REPORT.json": qa_path,
        "REVIEW_QUEUE.json": review_queue,
        "FINAL_ADJUDICATION_COMPLETE.json": adjudication_path,
        "final-adjudications.json": DECISIONS_PATH,
        "tts-selection.json": SELECTION_PATH,
    }
    for name, source in copies.items():
        shutil.copy2(source, root / name)
    html_path = root / "quran-urdu-reader.html"
    atomic_text(html_path, _reader_html(rows, _surah_names()))
    pdf_path = root / "quran-urdu-reader.pdf"
    if render_pdf:
        pending_pdf = root / ".quran-urdu-reader.pending.pdf"
        _render_pdf(html_path, pending_pdf)
        os.replace(pending_pdf, pdf_path)
    budget = _read_object(RUN_ROOT / "RUN.json").get("budget", {})
    readme = f"""# Quran Urdu v1.0.0

Evidence-audited modern Urdu translation of all 6,236 Quran ayahs.

- Source production run: `{RUN_ID}`
- Final deterministic QA: PASS, zero errors and zero warnings
- Final adjudication: {adjudication['decisions']} decisions across {len(adjudication['adjudicated_refs'])} ayahs
- Recorded translation pipeline spend: `${float(budget.get('spent_usd', 0)):.8f}`
- TTS candidate pinned separately in `tts-selection.json`; narration is not part of this text release

The translation was produced with language-model assistance and passed staged drafting, independent criticism, selective revision, verification, refrain, and final deterministic QA. The Arabic Quran remains authoritative.
"""
    atomic_text(root / "README.md", readme)
    artifact_paths = sorted(
        path for path in root.iterdir() if path.name not in {"MANIFEST.json", "SHA256SUMS.txt"}
    )
    artifact_hashes = {path.name: file_sha256(path) for path in artifact_paths}
    manifest = {
        "version": "quran-urdu-release-manifest-v1",
        "release_id": RELEASE_ID,
        "layout_version": LAYOUT_VERSION,
        "run_id": RUN_ID,
        "created_at": utc_now(),
        "ayahs": len(rows),
        "surahs": 114,
        "urdu_characters": sum(len(str(row["urdu"])) for row in rows),
        "qa_status": "PASS",
        "recorded_translation_spend_usd": float(budget.get("spent_usd", 0)),
        "input_sha256": input_hashes,
        "artifact_sha256": artifact_hashes,
    }
    atomic_json(manifest_path, manifest)
    checksum_lines = [
        f"{digest}  {name}" for name, digest in sorted(artifact_hashes.items())
    ]
    atomic_text(root / "SHA256SUMS.txt", "\n".join(checksum_lines) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    manifest = build_release(render_pdf=not args.no_pdf)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
