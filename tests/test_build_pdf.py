"""Tests for ``report/build_pdf.py``, the Markdown → PDF bridge for the report.

The PDF is a deliverable generated from a Markdown source, so the risk worth
testing is **silent loss**: a construct the renderer does not understand being
dropped, leaving the PDF quietly missing a table or a figure. These tests
therefore check that each supported construct produces a flowable, that an
unreadable reference fails loudly, and that the real report still renders.

Rendering is checked at the flowable level rather than by parsing the PDF, so a
failure names the construct that broke rather than a byte offset.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from reportlab.platypus import HRFlowable, Image, KeepTogether, ListFlowable, Paragraph, Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_builder() -> Any:
    """Import ``report/build_pdf.py``, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location("build_pdf", PROJECT_ROOT / "report" / "build_pdf.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_pdf = _load_builder()


@pytest.fixture()
def styles() -> Any:
    """Paragraph styles, built once per test."""
    return build_pdf.build_styles()


def flowables(markdown: str, styles: Any, source_dir: Path | None = None) -> list[Any]:
    """Render a Markdown snippet to flowables."""
    return build_pdf.render(markdown, source_dir or PROJECT_ROOT / "report", styles)


def flatten(items: list[Any]) -> list[Any]:
    """Expand ``KeepTogether`` wrappers so inner flowables can be inspected."""
    out = []
    for item in items:
        out.extend(item._content if isinstance(item, KeepTogether) else [item])
    return out


# =============================================================================
# Inline markup
# =============================================================================
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("plain text", "plain text"),
        ("**bold**", "<b>bold</b>"),
        ("*italic*", "<i>italic</i>"),
        ("[label](https://example.com)", "<u>label</u>"),
    ],
)
def test_inline_markup_is_converted(source: str, expected: str) -> None:
    """Emphasis and links become ReportLab markup; link targets are dropped in print."""
    assert build_pdf.inline(source) == expected


def test_inline_code_uses_the_monospace_face() -> None:
    """Backticked spans are rendered in the code face so identifiers stay legible."""
    assert "Courier" in build_pdf.inline("call `run_training` now")


def test_angle_brackets_are_escaped_not_interpreted() -> None:
    """Literal ``<`` in the report must not be read as markup and silently swallowed."""
    rendered = build_pdf.inline("p < 0.05 & rising")

    assert "&lt;" in rendered
    assert "&amp;" in rendered


# =============================================================================
# Block constructs
# =============================================================================
def test_headings_produce_paragraphs(styles: Any) -> None:
    """The first H1 is the title; deeper levels map to the heading styles."""
    out = flowables("# Title\n\n## Section\n\n### Subsection\n", styles)

    paragraphs = [f for f in out if isinstance(f, Paragraph)]
    assert [p.text for p in paragraphs] == ["Title", "Section", "Subsection"]
    assert paragraphs[0].style.name == "Title"


def test_table_becomes_a_table_with_a_header_row(styles: Any) -> None:
    """The ``|---|`` separator is dropped and the header row is kept for repeats."""
    out = flowables("| Target | RMSE |\n|---|---|\n| a | 0.4 |\n| b | 0.5 |\n", styles)

    tables = [f for f in out if isinstance(f, Table)]
    assert len(tables) == 1
    assert len(tables[0]._cellvalues) == 3  # header + two body rows
    assert tables[0].repeatRows == 1


def test_fenced_code_is_preserved_verbatim(styles: Any) -> None:
    """Code blocks keep their line breaks and indentation rather than reflowing."""
    out = flowables("```bash\npython main.py\n  --config x\n```\n", styles)

    code = next(f for f in out if isinstance(f, Paragraph) and f.style.name == "Code")
    assert "python&nbsp;main.py" in code.text  # spaces are non-breaking so alignment survives
    assert "<br/>" in code.text  # the line break is kept rather than reflowed
    assert "&nbsp;&nbsp;--config" in code.text  # and so is the leading indentation


def test_unclosed_code_fence_is_rejected(styles: Any) -> None:
    """An unterminated fence would swallow the rest of the report, so it fails loudly."""
    with pytest.raises(ValueError, match="Unclosed code fence"):
        flowables("```bash\npython main.py\n", styles)


def test_bullet_and_numbered_lists_are_rendered(styles: Any) -> None:
    """Both list kinds become list flowables rather than running together as prose."""
    out = flowables("- first\n- second\n\n1. one\n2. two\n", styles)

    lists = [f for f in out if isinstance(f, ListFlowable)]
    assert len(lists) == 2


def test_horizontal_rule_is_rendered(styles: Any) -> None:
    """Section dividers survive into the PDF."""
    assert any(isinstance(f, HRFlowable) for f in flowables("a\n\n---\n\nb\n", styles))


def test_paragraph_lines_are_joined(styles: Any) -> None:
    """Hard-wrapped source lines reflow into one paragraph instead of one per line."""
    out = flowables("one line\nand its continuation\n", styles)

    paragraphs = [f for f in out if isinstance(f, Paragraph)]
    assert len(paragraphs) == 1
    assert paragraphs[0].text == "one line and its continuation"


# =============================================================================
# Images
# =============================================================================
def test_image_is_embedded_with_its_caption(styles: Any, tmp_path: Path) -> None:
    """The alt text becomes the printed caption, so a figure is never unexplained."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path = tmp_path / "figure.png"
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    figure.savefig(figure_path)
    plt.close(figure)

    out = flatten(flowables("![A caption](figure.png)\n", styles, source_dir=tmp_path))

    assert any(isinstance(f, Image) for f in out)
    assert any(isinstance(f, Paragraph) and f.text == "A caption" for f in out)


def test_image_is_scaled_within_the_text_width(styles: Any, tmp_path: Path) -> None:
    """A wide figure is scaled down rather than running off the page."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path = tmp_path / "wide.png"
    figure, axis = plt.subplots(figsize=(20, 4))
    axis.plot([0, 1], [0, 1])
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)

    image = next(
        f for f in flatten(flowables("![wide](wide.png)\n", styles, source_dir=tmp_path))
        if isinstance(f, Image)
    )

    assert image.drawWidth <= build_pdf.MAX_IMAGE_WIDTH


def test_missing_image_is_reported(styles: Any, tmp_path: Path) -> None:
    """A broken figure reference fails the build instead of producing a gap in the PDF."""
    with pytest.raises(FileNotFoundError, match="missing image"):
        flowables("![gone](nowhere.png)\n", styles, source_dir=tmp_path)


# =============================================================================
# End-to-end
# =============================================================================
def test_missing_source_is_reported(tmp_path: Path) -> None:
    """Building from a non-existent report names the file rather than writing an empty PDF."""
    with pytest.raises(FileNotFoundError, match="Report source not found"):
        build_pdf.build(tmp_path / "absent.md", tmp_path / "out.pdf")


def test_real_report_renders_to_a_pdf(tmp_path: Path) -> None:
    """The actual report still renders, so a new construct in it cannot break the deliverable."""
    target = tmp_path / "final_report.pdf"

    build_pdf.build(build_pdf.DEFAULT_SOURCE, target)

    assert target.is_file()
    assert target.read_bytes().startswith(b"%PDF")
    assert target.stat().st_size > 10_000
