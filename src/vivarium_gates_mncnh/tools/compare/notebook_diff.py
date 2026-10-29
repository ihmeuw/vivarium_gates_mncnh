"""Render an HTML page that shows what moved in one notebook."""

import difflib
import html
from dataclasses import dataclass, field

_MAX_DIFF_LINES = 5000
"""Above this many lines a side, a unified diff is shown instead of a
side-by-side table, which takes quadratic time to build."""


@dataclass
class TextDiff:
    """One kind of text to show side by side, such as stdout."""

    label: str
    reference: str
    candidate: str


@dataclass
class CellDiff:
    """What moved in one code cell, or in the cell sources of a notebook.

    Attributes:
        title: Heading, such as ``code cell 4 (In [4])``.
        details: The report's description of each difference in the cell.
        source: The cell's source, shown once; empty to show none.
        texts: The texts that differ between the two sides.
        reference_images: Base64 PNG images the reference displayed.
        candidate_images: Base64 PNG images the candidate displayed.
    """

    title: str
    details: list[str]
    source: str
    texts: list[TextDiff]
    reference_images: list[str] = field(default_factory=list)
    candidate_images: list[str] = field(default_factory=list)


def render_notebook_diff(name: str, header: list[str], cells: list[CellDiff]) -> str:
    """Render a self-contained HTML page of what moved in a notebook.

    Args:
        name: The notebook's file name, used as the page title.
        header: Lines shown under the title, such as the two paths compared.
        cells: The cells that moved, in order.

    Returns:
        The page's HTML.
    """
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(name)} diff</title>",
        f"<style>{_STYLE}</style>",
        "</head><body>",
        f"<h1>{html.escape(name)}</h1>",
        "".join(f"<p class=meta>{html.escape(line)}</p>" for line in header),
    ]
    for cell in cells:
        parts.append(f"<section><h2>{html.escape(cell.title)}</h2>")
        parts.append(
            "<ul>"
            + "".join(f"<li>{html.escape(detail)}</li>" for detail in cell.details)
            + "</ul>"
        )
        if cell.source:
            parts.append(f"<pre class=source>{html.escape(cell.source)}</pre>")
        for text in cell.texts:
            parts.append(f"<h3>{html.escape(text.label)}</h3>")
            parts.append(_text_diff(text.reference, text.candidate))
        if cell.reference_images or cell.candidate_images:
            parts.append("<h3>images</h3><div class=images>")
            for side, images in (
                ("reference", cell.reference_images),
                ("candidate", cell.candidate_images),
            ):
                parts.append(f"<div><h4>{side}</h4>")
                parts += [
                    f'<img alt="{side} image {k}" src="data:image/png;base64,{image}">'
                    for k, image in enumerate(images, start=1)
                ]
                parts.append("</div>")
            parts.append("</div>")
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def _text_diff(reference: str, candidate: str) -> str:
    ref_lines, cand_lines = reference.splitlines(), candidate.splitlines()
    if max(len(ref_lines), len(cand_lines)) > _MAX_DIFF_LINES:
        lines = difflib.unified_diff(
            ref_lines, cand_lines, "reference", "candidate", lineterm="", n=3
        )
        return f"<pre class=unified>{html.escape(chr(10).join(lines))}</pre>"
    table = difflib.HtmlDiff(wrapcolumn=80).make_table(
        ref_lines, cand_lines, "reference", "candidate", context=True, numlines=3
    )
    return f"<div class=table>{table}</div>"


_STYLE = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --panel: #f6f8fa;
  --line: #d1d9e0; --add: #d2f4d6; --sub: #ffd8d3; --chg: #fff1b8; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --panel: #161b22;
    --line: #3d444d; --add: #1f4a2c; --sub: #5c2224; --chg: #4d3d0c; }
}
body { background: var(--bg); color: var(--fg); margin: 0 auto; max-width: 1400px;
  padding: 16px; font: 14px/1.5 system-ui, sans-serif; }
.meta { color: var(--muted); margin: 2px 0; font-family: ui-monospace, monospace; }
section { border-top: 1px solid var(--line); margin-top: 24px; }
pre, table.diff { font: 12px/1.4 ui-monospace, monospace; }
pre { background: var(--panel); padding: 8px; overflow-x: auto; white-space: pre-wrap; }
.table { overflow-x: auto; }
table.diff { border-collapse: collapse; width: 100%; }
table.diff td { padding: 0 4px; vertical-align: top; white-space: pre-wrap;
  word-break: break-all; }
table.diff th, .diff_header { color: var(--muted); background: var(--panel);
  font-weight: normal; }
.diff_next { display: none; }
.diff_add { background: var(--add); }
.diff_sub { background: var(--sub); }
.diff_chg { background: var(--chg); }
.images { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.images img { max-width: 100%; background: #ffffff; display: block; margin-bottom: 8px; }
"""
