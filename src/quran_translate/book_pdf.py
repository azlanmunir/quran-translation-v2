"""Printable PDF book generation for the English translation."""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch as unit_inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer
from reportlab.platypus.tableofcontents import TableOfContents

from .config import OUTPUT_DIR, PROJECT_ROOT
from .metadata import SURAHS
from .publication import publication_rows


BOOK_DIR = OUTPUT_DIR / "book"
PAGE_SIZE = (6 * inch, 9 * inch)
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
EDITION_SUBTITLE = "Evidence-Audited Modern English Translation"
DEFAULT_READING_NOTES = PROJECT_ROOT / "data" / "evidence" / "reading-notes-v2.4.1.json"


def register_fonts() -> str:
    regular = FONT_DIR / "Georgia.ttf"
    bold = FONT_DIR / "Georgia Bold.ttf"
    italic = FONT_DIR / "Georgia Italic.ttf"
    bold_italic = FONT_DIR / "Georgia Bold Italic.ttf"
    if not regular.exists():
        return "Times-Roman"

    pdfmetrics.registerFont(TTFont("BookSerif", str(regular)))
    if bold.exists():
        pdfmetrics.registerFont(TTFont("BookSerif-Bold", str(bold)))
    if italic.exists():
        pdfmetrics.registerFont(TTFont("BookSerif-Italic", str(italic)))
    if bold_italic.exists():
        pdfmetrics.registerFont(TTFont("BookSerif-BoldItalic", str(bold_italic)))
    pdfmetrics.registerFontFamily(
        "BookSerif",
        normal="BookSerif",
        bold="BookSerif-Bold" if bold.exists() else "BookSerif",
        italic="BookSerif-Italic" if italic.exists() else "BookSerif",
        boldItalic="BookSerif-BoldItalic" if bold_italic.exists() else "BookSerif",
    )
    return "BookSerif"


class QuranBookTemplate(BaseDocTemplate):
    def __init__(
        self,
        filename: str,
        *,
        title: str,
        page_ranges: dict[int, str] | None = None,
    ) -> None:
        self.book_title = title
        self.page_ranges = page_ranges or {}
        self.discovered_ranges: dict[int, list[tuple[str, str, str]]] = {}
        super().__init__(
            filename,
            pagesize=PAGE_SIZE,
            leftMargin=0.68 * unit_inch,
            rightMargin=0.58 * unit_inch,
            topMargin=0.72 * unit_inch,
            bottomMargin=0.66 * unit_inch,
            title=title,
            author="Azlan Munir",
            subject="Evidence-audited modern English Quran translation",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
        )
        self.addPageTemplates(
            [
                PageTemplate(id="normal", frames=[frame], onPage=self.draw_page),
            ]
        )

    def beforeDocument(self) -> None:
        self.discovered_ranges = {}

    def afterFlowable(self, flowable) -> None:  # noqa: ANN001 - ReportLab callback
        if isinstance(flowable, Paragraph) and flowable.style.name == "SurahHeading":
            text = flowable.getPlainText()
            self.notify("TOCEntry", (0, text, self.page))
        if hasattr(flowable, "ref_start") and hasattr(flowable, "ref_end"):
            self.discovered_ranges.setdefault(self.page, []).append(
                (flowable.ref_start, flowable.ref_end, flowable.surah_label)
            )

    def draw_page(self, canvas, doc) -> None:  # noqa: ANN001 - ReportLab callback
        canvas.saveState()
        if doc.page > 1:
            width, height = PAGE_SIZE
            canvas.setStrokeColor(colors.HexColor("#D0C7BC"))
            canvas.setLineWidth(0.35)
            canvas.line(doc.leftMargin, height - 0.49 * unit_inch, width - doc.rightMargin, height - 0.49 * unit_inch)
            canvas.setFillColor(colors.HexColor("#5B534A"))
            canvas.setFont("BookSerif" if "BookSerif" in pdfmetrics.getRegisteredFontNames() else "Times-Roman", 7.8)
            canvas.drawString(doc.leftMargin, height - 0.38 * unit_inch, self.book_title)
            page_range = self.page_ranges.get(doc.page)
            if page_range:
                canvas.drawRightString(width - doc.rightMargin, height - 0.38 * unit_inch, page_range)
            canvas.drawRightString(width - doc.rightMargin, 0.38 * unit_inch, str(doc.page))
        canvas.restoreState()


def build_styles(base_font: str) -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BookTitle",
            parent=styles["Title"],
            fontName=base_font,
            fontSize=27,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#221F1B"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BookSubtitle",
            parent=styles["Normal"],
            fontName=base_font,
            fontSize=11.5,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5B534A"),
            spaceAfter=26,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FrontMatter",
            parent=styles["Normal"],
            fontName=base_font,
            fontSize=9.7,
            leading=14,
            textColor=colors.HexColor("#2D2924"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TocTitle",
            parent=styles["Heading1"],
            fontName=base_font,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#221F1B"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SurahHeading",
            parent=styles["Heading1"],
            fontName=base_font,
            fontSize=16.2,
            leading=21,
            textColor=colors.HexColor("#221F1B"),
            spaceBefore=8,
            spaceAfter=14,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Ayah",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=10.25,
            leading=14.4,
            firstLineIndent=0,
            spaceAfter=6.2,
            textColor=colors.HexColor("#231F1A"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="AyahSmall",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9.6,
            leading=13.6,
            firstLineIndent=0,
            spaceAfter=5.4,
            textColor=colors.HexColor("#231F1A"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReaderParagraph",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=10.7,
            leading=15.3,
            firstLineIndent=17,
            spaceAfter=8.6,
            textColor=colors.HexColor("#231F1A"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReadingNote",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=8.35,
            leading=11.7,
            leftIndent=14,
            rightIndent=5,
            borderColor=colors.HexColor("#B9AEA1"),
            borderWidth=0.45,
            borderPadding=(5, 6, 5, 7),
            backColor=colors.HexColor("#F5F2EE"),
            spaceBefore=0,
            spaceAfter=8,
            textColor=colors.HexColor("#3D3731"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="EvidenceSource",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=8.15,
            leading=11.5,
            leftIndent=10,
            firstLineIndent=-10,
            spaceAfter=5,
            textColor=colors.HexColor("#3D3731"),
        )
    )
    return styles


def publication_note_parts(edition: str) -> list[str]:
    edition_notes = {
        "book": "This print edition keeps every line anchored to its standard surah and ayah reference.",
        "reader": (
            "This reader edition removes inline ayah labels from the body text so the "
            "translation can be read in paragraph form. Each page carries a surah and "
            "ayah range for navigation."
        ),
        "annotated": (
            "This annotated reading edition keeps ayah references and adds a selective "
            "set of evidence-adjudicated notes. It is not an exhaustive commentary."
        ),
    }
    if edition not in edition_notes:
        raise ValueError(f"Unknown publication edition: {edition}")
    return [
        edition_notes[edition],
        "Source Arabic text: Tanzil Quran Text, Uthmani Minimal, Version 1.1. Quranic Arabic Corpus morphology supports source analysis.",
        (
            "Method: choose the best-supported contextual sense before considering "
            "etymology; preserve material ambiguity; then shape the result into natural "
            "spoken English without adding imagery, agency, motive, or judgment."
        ),
        (
            "Production provenance: AI-assisted translation by Claude Opus 4.6, "
            "source-grounded criticism and verification by Gemini 3.1 Pro, deterministic "
            "quality gates, and documented editorial adjudication."
        ),
    ]


def _front_matter(
    story: list,
    styles: dict[str, ParagraphStyle],
    *,
    edition_label: str,
    edition_key: str,
) -> None:
    story.append(Spacer(1, 1.35 * unit_inch))
    story.append(Paragraph("The Quran", styles["BookTitle"]))
    story.append(Paragraph(EDITION_SUBTITLE, styles["BookSubtitle"]))
    story.append(Paragraph(edition_label, styles["BookSubtitle"]))
    story.append(PageBreak())
    story.append(Paragraph("Publication Note", styles["TocTitle"]))
    for part in publication_note_parts(edition_key):
        story.append(Paragraph(part, styles["FrontMatter"]))
    story.append(PageBreak())


def _toc(styles: dict[str, ParagraphStyle], base_font: str, style_name: str) -> list:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            name=style_name,
            fontName=base_font,
            fontSize=9.2,
            leading=13,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=2,
        )
    ]
    return [Paragraph("Contents", styles["TocTitle"]), toc, PageBreak()]


def _surah_heading(number: int, transliteration: str) -> str:
    return f"{number:03d}. {transliteration}"


def _ref_parts(ref: str) -> tuple[int, int]:
    surah, ayah = ref.split(":", 1)
    return int(surah), int(ayah)


def _format_range(start_ref: str, end_ref: str, surah_label: str) -> str:
    start_surah, start_ayah = _ref_parts(start_ref)
    end_surah, end_ayah = _ref_parts(end_ref)
    if start_surah == end_surah:
        if start_ayah == end_ayah:
            return f"{surah_label} {start_surah}:{start_ayah}"
        return f"{surah_label} {start_surah}:{start_ayah}-{end_ayah}"
    return f"{start_ref}-{end_ref}"


def _page_range_labels(discovered: dict[int, list[tuple[str, str, str]]]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for page, ranges in discovered.items():
        if not ranges:
            continue
        first_start, _, first_surah = ranges[0]
        _, last_end, last_surah = ranges[-1]
        if first_surah == last_surah:
            labels[page] = _format_range(first_start, last_end, first_surah)
        else:
            labels[page] = f"{first_start}-{last_end}"
    return labels


class RefParagraph(Paragraph):
    def __init__(
        self,
        text: str,
        style: ParagraphStyle,
        *,
        ref_start: str = "",
        ref_end: str = "",
        surah_label: str = "",
        **kwargs,
    ) -> None:
        super().__init__(text, style, **kwargs)
        self.ref_start = ref_start
        self.ref_end = ref_end
        self.surah_label = surah_label

    def split(self, availWidth, availHeight):  # noqa: ANN001, N802 - ReportLab API
        fragments = super().split(availWidth, availHeight)
        for fragment in fragments:
            fragment.ref_start = self.ref_start
            fragment.ref_end = self.ref_end
            fragment.surah_label = self.surah_label
        return fragments


def _reader_chunks(rows: list[sqlite3.Row], target_chars: int = 760) -> list[list[sqlite3.Row]]:
    chunks: list[list[sqlite3.Row]] = []
    current: list[sqlite3.Row] = []
    current_chars = 0
    for row in rows:
        text_len = len(row["translation"])
        if current and current_chars + text_len > target_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += text_len + 1
    if current:
        chunks.append(current)
    return chunks


def render_book_pdf(conn: sqlite3.Connection, run_id: str, output_path: Path | None = None) -> Path:
    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path or (BOOK_DIR / "quran-translation-book.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)

    base_font = register_fonts()
    styles = build_styles(base_font)
    rows = publication_rows(conn, run_id)
    if not rows:
        raise SystemExit(f"No publication layer found for {run_id}; run publication-build first.")

    by_surah: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_surah.setdefault(int(row["surah_number"]), []).append(row)

    story: list = []
    _front_matter(
        story,
        styles,
        edition_label="Print Edition",
        edition_key="book",
    )
    story.extend(_toc(styles, base_font, "TOCLevel0"))

    for index, info in enumerate(SURAHS):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(_surah_heading(info.number, info.transliteration), styles["SurahHeading"]))
        for row in by_surah.get(info.number, []):
            ref = html.escape(row["verse_key"])
            text = html.escape(row["translation"])
            style = styles["AyahSmall"] if len(row["translation"]) > 900 else styles["Ayah"]
            story.append(Paragraph(f"<b>{ref}</b>&nbsp;&nbsp;{text}", style))

    doc = QuranBookTemplate(str(path), title="The Quran")
    doc.multiBuild(story)
    return path


def _reader_story(
    rows: list[sqlite3.Row],
    styles: dict[str, ParagraphStyle],
    base_font: str,
) -> list:
    by_surah: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_surah.setdefault(int(row["surah_number"]), []).append(row)

    story: list = []
    _front_matter(
        story,
        styles,
        edition_label="Reader Edition",
        edition_key="reader",
    )
    story.extend(_toc(styles, base_font, "ReaderTOCLevel0"))

    for index, info in enumerate(SURAHS):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(_surah_heading(info.number, info.transliteration), styles["SurahHeading"]))
        surah_rows = by_surah.get(info.number, [])
        for chunk in _reader_chunks(surah_rows):
            text = " ".join(html.escape(row["translation"]) for row in chunk)
            story.append(
                RefParagraph(
                    text,
                    styles["ReaderParagraph"],
                    ref_start=chunk[0]["verse_key"],
                    ref_end=chunk[-1]["verse_key"],
                    surah_label=info.transliteration,
                )
            )
    return story


def render_reader_pdf(conn: sqlite3.Connection, run_id: str, output_path: Path | None = None) -> Path:
    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path or (BOOK_DIR / "quran-translation-reader-edition.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)

    base_font = register_fonts()
    styles = build_styles(base_font)
    rows = publication_rows(conn, run_id)
    if not rows:
        raise SystemExit(f"No publication layer found for {run_id}; run publication-build first.")

    title = "The Quran"
    probe_path = path.with_name(f"{path.stem}.range-pass.pdf")
    probe_doc = QuranBookTemplate(str(probe_path), title=title)
    probe_doc.multiBuild(_reader_story(rows, styles, base_font))
    page_ranges = _page_range_labels(probe_doc.discovered_ranges)
    probe_path.unlink(missing_ok=True)

    doc = QuranBookTemplate(str(path), title=title, page_ranges=page_ranges)
    doc.multiBuild(_reader_story(rows, styles, base_font))
    return path


def _evidence_label(evidence_ref: str, sources: dict[str, dict[str, str]]) -> str | None:
    source_id, _, locator = evidence_ref.partition(":")
    if source_id == "PILOT_NOTES":
        return None
    source = sources[source_id]
    short_names = {
        "TANZIL": "Tanzil",
        "QAC_LOCAL": "QAC morphology",
        "QAC_4_34": "QAC syntax 4:34",
        "QAC_SMD": "QAC root concordance",
        "BUKHARI_4505": "Sahih al-Bukhari 4505",
        "LANE_SMD": "Lane's Lexicon, root sad-mim-dal",
    }
    label = short_names.get(source_id, source["title"])
    return f"{label} {locator}".strip()


def _annotated_story(
    rows: list[sqlite3.Row],
    styles: dict[str, ParagraphStyle],
    base_font: str,
    notes_payload: dict[str, Any],
) -> list:
    by_surah: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_surah.setdefault(int(row["surah_number"]), []).append(row)
    notes = {entry["ref"]: entry for entry in notes_payload["notes"]}
    sources = notes_payload["sources"]

    story: list = []
    _front_matter(
        story,
        styles,
        edition_label="Annotated Reading Edition",
        edition_key="annotated",
    )
    story.extend(_toc(styles, base_font, "AnnotatedTOCLevel0"))

    for index, info in enumerate(SURAHS):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(_surah_heading(info.number, info.transliteration), styles["SurahHeading"]))
        for row in by_surah.get(info.number, []):
            ref = str(row["verse_key"])
            text = html.escape(row["translation"])
            style = styles["AyahSmall"] if len(row["translation"]) > 900 else styles["Ayah"]
            story.append(Paragraph(f"<b>{html.escape(ref)}</b>&nbsp;&nbsp;{text}", style))
            note = notes.get(ref)
            if note:
                labels = [
                    label
                    for evidence_ref in note["evidence"]
                    if (label := _evidence_label(evidence_ref, sources))
                ]
                evidence = "; ".join(labels)
                story.append(
                    Paragraph(
                        f"<b>Reading note.</b> {html.escape(note['note'])}"
                        f"<br/><font size='7.4' color='#6B6259'><i>Evidence: "
                        f"{html.escape(evidence)}.</i></font>",
                        styles["ReadingNote"],
                    )
                )

    story.append(PageBreak())
    story.append(Paragraph("Evidence Sources", styles["TocTitle"]))
    for source_id, source in sources.items():
        if source_id == "PILOT_NOTES":
            continue
        details = source["title"]
        if source.get("url"):
            details += f" - {source['url']}"
        elif source.get("path"):
            details += f" - {source['path']} (SHA-256 {source['sha256']})"
        story.append(Paragraph(html.escape(details), styles["EvidenceSource"]))
    return story


def render_annotated_pdf(
    conn: sqlite3.Connection,
    run_id: str,
    output_path: Path | None = None,
    notes_path: Path = DEFAULT_READING_NOTES,
) -> Path:
    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path or (BOOK_DIR / "quran-translation-annotated-reading-edition.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = publication_rows(conn, run_id)
    if not rows:
        raise SystemExit(f"No publication layer found for {run_id}; run publication-build first.")
    notes_payload = json.loads(notes_path.read_text(encoding="utf-8"))
    base_font = register_fonts()
    styles = build_styles(base_font)
    doc = QuranBookTemplate(str(path), title="The Quran")
    doc.multiBuild(_annotated_story(rows, styles, base_font, notes_payload))
    return path


def inspect_pdf(path: Path) -> dict[str, object]:
    import pymupdf

    doc = pymupdf.open(path)
    text_chars = 0
    pages_without_text: list[int] = []
    for page_index in range(doc.page_count):
        text = doc[page_index].get_text("text")
        text_chars += len(text)
        if not text.strip():
            pages_without_text.append(page_index + 1)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "pages": doc.page_count,
        "text_chars": text_chars,
        "pages_without_text": pages_without_text,
        "metadata": dict(doc.metadata or {}),
    }


def write_pdf_inspection(path: Path, output_path: Path | None = None) -> Path:
    report = inspect_pdf(path)
    out = output_path or path.with_suffix(".inspection.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
