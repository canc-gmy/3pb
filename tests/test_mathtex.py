"""Tests for TeX markup and its rendering into the three report formats."""

import base64
import re

from sandwich3pb.mathtex import render_html, strip_math, tex
from sandwich3pb.report import build_html, build_markdown, build_pdf
from sandwich3pb.units import (
    FORCE,
    LENGTH,
    SI,
    STRESS,
    fmt_tex_list,
    fmt_tex_num,
    fmt_tex_qty,
)
from tests.conftest import make_config
from tests.test_report import FakeResults


class TestTexHelpers:
    def test_tex_wraps_once(self):
        assert tex("x^2") == "$x^2$"
        assert tex("$x^2$") == "$x^2$"

    def test_strip_math_removes_markers(self):
        assert strip_math("peak $\\sigma_{xx}$ here") == (
            "peak \\sigma_{xx} here"
        )

    def test_strip_math_leaves_unmatched_dollars(self):
        assert strip_math("cost is $5 and $6") == "cost is $5 and $6"


class TestRenderHtml:
    def test_math_becomes_inline_svg(self):
        out = render_html("value $\\sigma_{xx}$ end")
        assert "<img" in out and "class=\"math\"" in out
        assert out.startswith("value <img") and out.endswith(" end")
        # self-contained: the SVG is inlined as a data URI, nothing fetched
        assert 'src="data:image/svg+xml;base64,' in out

    def test_the_img_tag_is_well_formed(self):
        """Raw SVG in src="..." ends the attribute at its first inner quote.

        Every attribute value must be quoted and free of stray quotes, or
        the browser silently drops the image and renders the alt text.
        """
        out = render_html(r"$\sigma_{xx}$")
        tag = out[out.index("<img"):out.index(">") + 1]
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        assert list(attrs) == ["class", "alt", "style", "src"]
        payload = attrs["src"].split(",", 1)[1]
        assert attrs["src"].startswith("data:image/svg+xml;base64,")
        assert base64.b64decode(payload).lstrip().startswith(b"<svg")

    def test_html_math_images_actually_decode(self):
        """Every rendered snippet in the report must be a decodable SVG."""
        html = build_html(FakeResults(make_config(), "."))
        sources = re.findall(r'class="math"[^>]*src="([^"]*)"', html)
        assert sources
        for src in sources:
            payload = src.split(",", 1)[1]
            assert base64.b64decode(payload).lstrip().startswith(b"<svg")

    def test_prose_amounts_are_not_mistaken_for_math(self):
        """``$5 and $6`` is money, not a math span."""
        assert render_html("costs $5 and $6") == "costs $5 and $6"

    def test_rendered_svg_carries_the_tex_as_alt_text(self):
        out = render_html(r"$\tau_{xz}$")
        assert r'alt="\tau_{xz}"' in out

    def test_unparseable_math_falls_back_to_code(self):
        out = render_html(r"$\bogus{$")
        assert out.startswith("<code>") and out.endswith("</code>")

    def test_text_without_math_is_untouched(self):
        assert render_html("plain text") == "plain text"

    def test_repeated_symbols_are_cached(self):
        first = render_html(r"$\mathrm{Pa}$")
        second = render_html(r"$\mathrm{Pa}$")
        assert first == second


class TestTexFormatting:
    def test_scientific_notation_is_tex_not_python(self):
        assert fmt_tex_num(3.9e10) == r"$3.9\times 10^{10}$"
        assert fmt_tex_num(1e-9) == r"$1\times 10^{-9}$"

    def test_plain_magnitudes_stay_plain(self):
        assert fmt_tex_num(39000.0) == "$39000$"
        assert fmt_tex_num(0.3) == "$0.3$"
        assert fmt_tex_num(0.0) == "$0$"

    def test_negative_values_keep_the_sign(self):
        assert fmt_tex_num(-114.3) == "$-114.3$"

    def test_quantity_carries_a_tex_unit(self):
        assert fmt_tex_qty(39000.0, SI, STRESS) == (
            r"$3.9\times 10^{10}\,\mathrm{Pa}$"
        )
        assert fmt_tex_qty(300.0, SI, LENGTH) == r"$0.3\,\mathrm{m}$"
        assert fmt_tex_qty(920.47, SI, FORCE) == r"$920.5\,\mathrm{N}$"

    def test_dimensionless_has_no_unit(self):
        assert fmt_tex_qty(0.141, SI, "none") == "$0.141$"

    def test_missing_values_are_dashes(self):
        assert fmt_tex_num(None) == "—"
        assert fmt_tex_qty(None, SI, FORCE) == "—"
        assert fmt_tex_num(float("nan")) == "—"

    def test_lists_are_comma_separated(self):
        out = fmt_tex_list([600.2, 600.3], SI, FORCE)
        assert out == r"$600.2, 600.3\,\mathrm{N}$"

    def test_every_symbol_renders_in_mathtext(self):
        """A unit symbol that mathtext cannot parse would silently break."""
        from sandwich3pb.units import SYSTEMS

        for system in SYSTEMS.values():
            for dim in (LENGTH, STRESS, FORCE, "gradient", "rigidity",
                        "penalty"):
                symbol = system.tex_symbol(dim)
                assert symbol, f"{system.name}/{dim} has no TeX symbol"
                # a bare, undelimited \mathrm would print literally
                assert "$" not in symbol
                assert render_html(tex(symbol)).startswith("<img")


class TestReportsShareTheSameMath:
    def test_markdown_keeps_tex_as_is(self):
        md = build_markdown(FakeResults(make_config(), "."))
        assert r"$P_{\max}$" in md
        assert r"\mathrm{Pa}" in md
        assert "<img" not in md  # markdown relies on the renderer's math

    def test_html_inlines_every_math_snippet(self):
        html = build_html(FakeResults(make_config(), "."))
        # The inlined maths SVGs carry matplotlib's source in an XML
        # comment, so strip the SVG payloads before looking for leftovers
        # in the markup a reader actually sees.
        visible = re.sub(r"<svg\b.*?</svg>", "", html, flags=re.S)
        assert not re.search(r"\$[^$\n]+\$", visible)
        # ...and nothing degraded to the raw-source fallback either
        assert "<code>" not in visible
        # ...and the mathematics is really there
        assert html.count('<img class="math"') > 10

    def test_html_never_references_the_network(self):
        html = build_html(FakeResults(make_config(), "."))
        # the SVG namespace URI is a declaration, not a fetch; what must
        # not appear is any attribute a browser would go and resolve
        assert 'src="http' not in html
        assert 'href="http' not in html
        assert "<script" not in html
        assert "<link" not in html

    def test_html_marks_predicted_onset(self):
        view = FakeResults(make_config(), ".")
        view.summary["failure_onset"] = {
            "step": 2, "travel_mm": 0.4, "force_N": 800.0,
            "deflection_mm": 0.3, "criterion_keys": ["layer_0:tsai_wu"],
        }
        html = build_html(view)
        assert "Predicted first failure — step 2" in html
        assert "no progressive damage is modeled" in html
        assert 'class="onset"' in html

    def test_html_failure_rows_are_complete_and_threshold_highlighted(self):
        view = FakeResults(make_config(), ".")
        view.summary["failure_by_layer"]["layer_0"]["max_tsai_wu"] = 1.2e7
        view.summary["failure_by_layer"]["layer_2"]["max_tsai_wu"] = 0.9
        html = build_html(view)
        rows = re.findall(r'<tr class="failure">(.*?)</tr>', html, flags=re.S)
        assert len(rows) == 1
        assert rows[0].startswith("<td>layer_0</td>")
        assert rows[0].count("<td") == 10
        assert rows[0].endswith("</td>")

    def test_html_is_self_contained_without_plot_files(self):
        """Missing figures degrade to a note, not an exception."""
        html = build_html(FakeResults(make_config(), "/nonexistent-plots"))
        assert "missing:" in html
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_pdf_renders_mathtext_tables(self, tmp_path):
        path = build_pdf(FakeResults(make_config(), str(tmp_path)),
                         str(tmp_path / "r.pdf"))
        data = open(path, "rb").read()
        assert data.startswith(b"%PDF")
        assert len(data) > 4000