"""TeX mathematics for the reports, in one shared source of truth.

Quantities are written once as TeX (``$...$``) and rendered three ways:

* **Markdown** -- the ``$...$`` markup is emitted as-is, which GitHub,
  VS Code and pandoc all typeset.
* **PDF** -- the report is drawn with matplotlib, whose mathtext engine
  renders ``$...$`` natively in any text, table cell or axis label.
* **HTML** -- a browser has no TeX, and pulling KaTeX/MathJax from a CDN
  would break the report's promise to be self-contained. Instead each
  snippet is typeset once with matplotlib's mathtext and inlined as an
  SVG ``<img>``, so the HTML stays a single portable file that displays
  the same mathematics as the PDF with no network and no JavaScript.

Only the mathtext subset is used (``\\sigma``, ``\\tau``, ``_``/``^``,
``\\mathrm``, ``\\frac``, ``\\sqrt``, ``\\max``, ``\\approx``, ``\\cdot``),
all of which matplotlib supports natively -- no external LaTeX install.
"""

from __future__ import annotations

import re
from functools import lru_cache

#: ``$...$`` runs of TeX. The delimiters must not sit flush against an
#: alphanumeric on either side, so prose that merely mentions money
#: ("costs $5 and $6") is not mistaken for a pair of math spans.
_MATH = re.compile(r"(?<![0-9A-Za-z])\$([^$\n]+)\$(?![0-9A-Za-z])")


def tex(source: str) -> str:
    """Wrap ``source`` in inline-TeX delimiters (idempotent)."""
    source = source.strip()
    if source.startswith("$") and source.endswith("$"):
        return source
    return f"${source}$"


def strip_math(text: str) -> str:
    """Return ``text`` with ``$...$`` markers removed (plain-text fallback)."""
    return _MATH.sub(r"\1", text)


def render_html(text: str) -> str:
    """Replace every ``$...$`` in ``text`` with an inline SVG image.

    Snippets already rendered are cached, so the handful of repeated unit
    symbols across the tables cost one typesetting pass each.
    """
    return _MATH.sub(lambda m: _math_svg(m.group(1)), text)


@lru_cache(maxsize=512)
def _math_svg(expression: str) -> str:
    """Typeset one TeX expression and return it as an inline ``<img>``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(0.01, 0.01))
    try:
        fig.text(0, 0, tex(expression), fontsize=13.0)
        import io as _io

        # matplotlib writes *bytes* to a file-like object, so this must be
        # a BytesIO -- a StringIO raises TypeError on write
        buffer = _io.BytesIO()
        fig.savefig(
            buffer,
            format="svg",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.015,
        )
    except Exception:
        # an expression mathtext cannot parse must not take the report
        # down: fall back to the literal TeX source, still marked up
        return f'<code>{_escape(expression)}</code>'
    finally:
        plt.close(fig)

    svg = _extract_svg(buffer.getvalue().decode("utf-8"))
    if svg is None:
        return f"<code>{_escape(expression)}</code>"
    return (
        '<img class="math" alt="' + _escape(expression) + '" '
        'style="vertical-align:-0.22em" src="' + svg + '">'
    )


def _extract_svg(document: str):
    """Pull the ``<svg>...</svg>`` element out of an SVG document.

    The ``<defs>`` holding the glyph paths lives inside ``<svg>``, so
    slicing from the opening tag to the closing tag keeps the snippet
    self-contained.
    """
    start = document.find("<svg")
    end = document.rfind("</svg>")
    if start == -1 or end == -1:
        return None
    # drop the XML prolog / DOCTYPE that precede the root element
    return document[start:end + len("</svg>")]


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
