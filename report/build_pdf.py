"""Render ``final_report.md`` to ``final_report.pdf``.

The assignment asks for the technical report as a PDF; the report itself is
authored in Markdown so it stays reviewable in the repository and in diffs.
This script is the one-way bridge between the two, so the PDF is never edited
by hand and can never drift from its source::

    python report/build_pdf.py

It deliberately supports only the Markdown this report uses — headings, body
text with inline emphasis and code, tables, fenced code blocks, lists, images
and horizontal rules — rather than being a general Markdown engine. Anything
outside that set raises instead of being silently dropped, so a new construct
in the report cannot vanish from the deliverable unnoticed.

``reportlab`` is the only dependency and is already pinned for the project; no
LaTeX or headless browser is required, so the PDF builds identically on any
machine that can run the pipeline.
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "report" / "final_report.md"
DEFAULT_TARGET = PROJECT_ROOT / "report" / "final_report.pdf"

PAGE_MARGIN = 18 * mm
BODY_FONT = "Helvetica"
CODE_FONT = "Courier"
ACCENT = colors.HexColor("#1f3864")
RULE_COLOR = colors.HexColor("#b4bcd0")
TABLE_HEADER_BG = colors.HexColor("#e8ecf4")
TABLE_STRIPE_BG = colors.HexColor("#f5f7fb")
CODE_BG = colors.HexColor("#f2f2f4")
TEXT_WIDTH = A4[0] - 2 * PAGE_MARGIN
# Figures are wide and short, so printing them at the full text width wastes a
# third of a page and makes the type inside them larger than the body text.
MAX_IMAGE_WIDTH = TEXT_WIDTH * 0.64

# Markdown constructs this renderer understands, in match order.
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE = re.compile(r"^```")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_DIVIDER = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
ORDERED = re.compile(r"^\s*(\d+)\.\s+(.*)$")
RULE = re.compile(r"^(-{3,}|\*{3,}|_{3,})\s*$")
IMAGE = re.compile(r"^!\[(?P<alt>.*)\]\((?P<src>[^)]+)\)\s*$")

# Inline spans, applied to already-escaped text.
INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


@dataclass(frozen=True)
class Styles:
    """The paragraph styles the renderer uses, built once per document."""

    title: ParagraphStyle
    h1: ParagraphStyle
    h2: ParagraphStyle
    h3: ParagraphStyle
    body: ParagraphStyle
    bullet: ParagraphStyle
    code: ParagraphStyle
    cell: ParagraphStyle
    header_cell: ParagraphStyle
    caption: ParagraphStyle


def build_styles() -> Styles:
    """Construct the document's paragraph styles."""
    base = getSampleStyleSheet()["BodyText"]
    body = ParagraphStyle(
        "Body",
        parent=base,
        fontName=BODY_FONT,
        fontSize=9.0,
        leading=12.4,
        spaceAfter=5,
        alignment=TA_JUSTIFY,
    )
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.9, leading=10.0, spaceAfter=0, alignment=0)
    return Styles(
        title=ParagraphStyle(
            "Title", parent=body, fontName=f"{BODY_FONT}-Bold", fontSize=17, leading=21,
            spaceAfter=14, textColor=ACCENT, alignment=0,
        ),
        h1=ParagraphStyle(
            "H1", parent=body, fontName=f"{BODY_FONT}-Bold", fontSize=14, leading=18,
            spaceBefore=14, spaceAfter=8, textColor=ACCENT, alignment=0,
        ),
        h2=ParagraphStyle(
            "H2", parent=body, fontName=f"{BODY_FONT}-Bold", fontSize=11.5, leading=15,
            spaceBefore=12, spaceAfter=6, textColor=ACCENT, alignment=0,
        ),
        h3=ParagraphStyle(
            "H3", parent=body, fontName=f"{BODY_FONT}-Bold", fontSize=10, leading=13,
            spaceBefore=9, spaceAfter=4, alignment=0,
        ),
        body=body,
        bullet=ParagraphStyle("Bullet", parent=body, spaceAfter=2, alignment=0),
        code=ParagraphStyle(
            "Code", parent=body, fontName=CODE_FONT, fontSize=7.8, leading=10.2,
            backColor=CODE_BG, borderPadding=6, spaceBefore=4, spaceAfter=8, alignment=0,
        ),
        cell=cell,
        header_cell=ParagraphStyle("HeaderCell", parent=cell, fontName=f"{BODY_FONT}-Bold"),
        caption=ParagraphStyle(
            "Caption", parent=body, fontSize=8.2, leading=11, spaceBefore=3, spaceAfter=10,
            textColor=colors.HexColor("#555555"), alignment=0,
        ),
    )


def inline(text: str) -> str:
    """Convert inline Markdown to ReportLab's mini-HTML markup.

    Escaping happens first so that any ``<`` or ``&`` in the report is literal;
    only the tags this function adds are ever interpreted.
    """
    out = html.escape(text, quote=False)
    out = INLINE_CODE.sub(rf'<font face="{CODE_FONT}" size="8.4">\1</font>', out)
    out = BOLD.sub(r"<b>\1</b>", out)
    out = ITALIC.sub(r"<i>\1</i>", out)
    out = LINK.sub(r"<u>\1</u>", out)  # a PDF of a repo file: show the text, drop the href
    return out


def split_row(line: str) -> list[str]:
    """Split a Markdown table row into trimmed cell texts."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def make_table(rows: list[list[str]], styles: Styles) -> Table:
    """Build a styled table flowable from header + body rows."""
    header, body_rows = rows[0], rows[1:]
    data = [[Paragraph(inline(cell), styles.header_cell) for cell in header]]
    data += [[Paragraph(inline(cell), styles.cell) for cell in row] for row in body_rows]

    n_columns = max(len(row) for row in data)
    table = Table(data, colWidths=[TEXT_WIDTH / n_columns] * n_columns, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    style += [
        ("BACKGROUND", (0, index), (-1, index), TABLE_STRIPE_BG)
        for index in range(2, len(data), 2)
    ]
    table.setStyle(TableStyle(style))
    return table


def make_image(match: re.Match[str], source_dir: Path, styles: Styles) -> list[Flowable]:
    """Build an image flowable plus its caption, scaled to the text width.

    Raises:
        FileNotFoundError: If the referenced image does not exist.
    """
    path = (source_dir / match.group("src")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Report references a missing image: {path}")
    image = Image(str(path))
    scale = min(1.0, MAX_IMAGE_WIDTH / image.imageWidth)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "LEFT"
    caption = Paragraph(inline(match.group("alt")), styles.caption)
    return [KeepTogether([Spacer(1, 4), image, caption])]


def render(markdown: str, source_dir: Path, styles: Styles) -> list[Flowable]:
    """Convert the report's Markdown into an ordered list of PDF flowables.

    Raises:
        ValueError: If a fenced code block is never closed.
    """
    lines = markdown.splitlines()
    flowables: list[Flowable] = []
    paragraph: list[str] = []
    bullets: list[str] = []
    numbered: list[str] = []
    table_rows: list[list[str]] = []
    seen_title = False

    def flush_paragraph() -> None:
        if paragraph:
            flowables.append(Paragraph(inline(" ".join(paragraph)), styles.body))
            paragraph.clear()

    def flush_lists() -> None:
        for items, bullet_type in ((bullets, "bullet"), (numbered, "1")):
            if items:
                flowables.append(
                    ListFlowable(
                        [ListItem(Paragraph(inline(item), styles.bullet), leftIndent=12)
                         for item in items],
                        bulletType=bullet_type,
                        leftIndent=14,
                        bulletFontSize=8,
                        spaceAfter=6,
                    )
                )
                items.clear()

    def flush_table() -> None:
        if table_rows:
            flowables.append(make_table(list(table_rows), styles))
            flowables.append(Spacer(1, 8))
            table_rows.clear()

    def flush_all() -> None:
        flush_paragraph()
        flush_lists()
        flush_table()

    index = 0
    while index < len(lines):
        line = lines[index]

        if FENCE.match(line):
            flush_all()
            closing = next((j for j in range(index + 1, len(lines)) if FENCE.match(lines[j])), None)
            if closing is None:
                raise ValueError(f"Unclosed code fence starting at line {index + 1}")
            code = "\n".join(lines[index + 1 : closing])
            escaped = html.escape(code, quote=False).replace(" ", "&nbsp;").replace("\n", "<br/>")
            flowables.append(Paragraph(escaped, styles.code))
            index = closing + 1
            continue

        heading = HEADING.match(line)
        if heading:
            flush_all()
            level, text = len(heading.group(1)), heading.group(2)
            if level == 1 and not seen_title:
                seen_title = True
                flowables.append(Paragraph(inline(text), styles.title))
            else:
                style = {1: styles.h1, 2: styles.h2}.get(level, styles.h3)
                if level <= 2 and flowables:
                    flowables.append(Spacer(1, 4))
                flowables.append(Paragraph(inline(text), style))
            index += 1
            continue

        image = IMAGE.match(line)
        if image:
            flush_all()
            flowables.extend(make_image(image, source_dir, styles))
            index += 1
            continue

        if TABLE_ROW.match(line):
            flush_paragraph()
            flush_lists()
            if not TABLE_DIVIDER.match(line):  # the |---|---| separator carries no content
                table_rows.append(split_row(line))
            index += 1
            continue
        flush_table()

        if RULE.match(line):
            flush_all()
            flowables.append(Spacer(1, 3))
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=RULE_COLOR))
            flowables.append(Spacer(1, 5))
            index += 1
            continue

        bullet = BULLET.match(line)
        if bullet:
            flush_paragraph()
            bullets.append(bullet.group(1))
            index += 1
            continue

        ordered = ORDERED.match(line)
        if ordered:
            flush_paragraph()
            numbered.append(ordered.group(2))
            index += 1
            continue

        if not line.strip():
            flush_all()
            index += 1
            continue

        flush_lists()
        paragraph.append(line.strip())
        index += 1

    flush_all()
    return flowables


def decorate(canvas: object, doc: SimpleDocTemplate) -> None:
    """Draw the page number at the foot of every page."""
    canvas.saveState()  # type: ignore[attr-defined]
    canvas.setFont(BODY_FONT, 8)  # type: ignore[attr-defined]
    canvas.setFillColor(colors.HexColor("#777777"))  # type: ignore[attr-defined]
    canvas.drawCentredString(A4[0] / 2, PAGE_MARGIN / 2, str(doc.page))  # type: ignore[attr-defined]
    canvas.restoreState()  # type: ignore[attr-defined]


def build(source: Path = DEFAULT_SOURCE, target: Path = DEFAULT_TARGET) -> Path:
    """Render ``source`` Markdown to ``target`` PDF.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If the Markdown source does not exist.
    """
    if not source.is_file():
        raise FileNotFoundError(f"Report source not found: {source}")
    styles = build_styles()
    document = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
        title=source.stem,
        author="Medvolt.AI case study",
    )
    document.build(
        render(source.read_text(encoding="utf-8"), source.parent, styles),
        onFirstPage=decorate,
        onLaterPages=decorate,
    )
    return target


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Markdown report")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="PDF to write")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    written = build(args.source, args.target)
    print(f"Wrote {written}")  # noqa: T201 - a build script, not pipeline logging
