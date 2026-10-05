"""Tests for report generation (no FEM required)."""

import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from sandwich3pb.report import (
    _failure_rows,
    _kpi_rows,
    _layup_rows,
    _md_table,
    _model_rows,
    _response_rows,
    _solver_rows,
    build_latex,
    build_markdown,
)
from tests.conftest import make_config


class FakeResults:
    def __init__(self, cfg, tmpdir):
        self.cfg = cfg
        self.summary = self._summary()
        self._plots_dir = tmpdir

    def _summary(self):
        return {
            "case_name": "fake",
            "max_force_N": 1200.5,
            "deflection_at_max_force_mm": 0.8,
            "max_deflection_mm": 1.0,
            "final_travel_mm": 1.0,
            "force_gradient_N_per_mm": 1500.0,
            "gradient_fit_r2": 0.9995,
            "gradient_fit_points": 5,
            "apparent_flexural_rigidity_Nmm2": 2.5e8,
            "layup_flexural_rigidity_Nmm2": 2.6e8,
            "layup_flexural_rigidity_per_width_Nmm": 6.5e6,
            "neutral_axis_z_mm": 11.5,
            "rigidity_ratio_FE_over_layup": 0.96,
            "load_roller_force_N": 1200.5,
            "support_reactions_N": [600.2, 600.3],
            "support_reaction_total_N": 1200.5,
            "force_balance_residual": 1e-12,
            "max_contact_penetration_mm": 4e-4,
            "peak_contact_penetration_mm": 6e-4,
            "n_failed_steps": 0,
            "peak_newton_iterations": 5,
            "max_tsai_wu_faces": 0.12,
            "max_core_shear_ratio": 0.35,
            "max_core_crushing_ratio": 0.20,
            "n_steps": 5,
            "all_steps_converged": True,
            "n_nodes": 999,
            "n_dofs": 2997,
            "assembly_time_s": 1.0,
            "solve_time_s": 2.0,
            "failure_onset": None,
            "failure_by_layer": {
                "layer_0": {"role": "face", "material": "glass_epoxy",
                            "orientation_deg": 0.0, "max_tsai_wu": 0.12},
                "layer_1": {"role": "core", "material": "pvc_foam",
                            "max_shear_ratio": 0.35,
                            "max_crushing_ratio": 0.20},
                "layer_2": {"role": "face", "material": "glass_epoxy",
                            "orientation_deg": 90.0, "max_tsai_wu": 0.08},
            },
        }


def test_layup_rows_include_orientation(tmp_path):
    cfg = make_config()
    header, rows = _layup_rows(cfg)
    assert any("fibre orientation" in h for h in header)
    assert rows[1][4] == "$0$"  # core has no orientation value of interest
    assert rows[0][2] == "glass_epoxy"


def test_layup_units_follow_the_case_file():
    """The columnar tables announce their unit in the *header*, in the
    case file's units -- never in a repeated ``unit`` cell and never as
    bare, undelimited TeX."""
    from sandwich3pb.units import MM_N_MPA

    cfg = make_config()
    header, si_rows = _layup_rows(cfg)
    assert not any(str(h) == "unit" for h in header)
    assert r"$\mathrm{m}$" in header[3]          # thickness column, SI
    assert r"$\mathrm{Pa}$" in header[5]         # E_x column, SI
    assert not any(r"mm" in str(h) for h in header)
    # ...and no row cell smuggles a unit in
    for row in si_rows:
        assert not any(r"\mathrm" in str(cell) for cell in row), row

    cfg.units = MM_N_MPA.name
    mm_header, _ = _layup_rows(cfg)
    assert r"$\mathrm{mm}$" in mm_header[3]
    assert r"$\mathrm{MPa}$" in mm_header[5]


def test_layup_rows_show_90deg(tmp_path):
    cfg = make_config()
    cfg.stackup[0].fibre_orientation = 90.0
    _, rows = _layup_rows(cfg)
    assert rows[0][4] == "$90$"


def test_response_rows_contain_rigidity():
    cfg = make_config()
    view = FakeResults(cfg, ".")
    _, rows = _response_rows(view.summary, cfg.unit_system)
    flat = " | ".join(cell for r in rows for cell in r)
    assert "flexural rigidity" in flat
    assert "force gradient" in flat
    # the duplicated rows are gone: one deflection, one force
    labels = [str(r[0]) for r in rows]
    assert not any(label.startswith("deflection at max force") for label in labels)
    assert not any(label.startswith("final roller travel") for label in labels)
    assert not any(label.startswith("load-roller force") for label in labels)


def test_response_rows_convert_to_case_units():
    """Internal mm/N/MPa values must be presented in the case file's units,
    with the unit sitting in its own column beside the value."""
    from sandwich3pb.units import MM_N_MPA

    cfg = make_config()
    view = FakeResults(cfg, ".")

    def row_in(rows, prefix):
        return next(r for r in rows if str(r[0]).startswith(prefix))

    _, si = _response_rows(view.summary, cfg.unit_system)
    # 1200.5 N: N is the force unit in both systems
    force = row_in(si, "max force")
    assert force[1] == r"$1200$" and force[2] == r"$\mathrm{N}$"
    # 2.5e8 N*mm^2 -> 250 N*m^2 in SI
    rigidity = row_in(si, "apparent flexural rigidity")
    assert rigidity[1] == r"$250$" and rigidity[2] == r"$\mathrm{N\,m^{2}}$"
    deflection = row_in(si, "max deflection")
    assert deflection[1] == r"$0.001$"
    assert deflection[2] == r"$\mathrm{m}$"

    # a sub-micron quantity switches to scientific TeX, not e-07
    _, solver_si = _solver_rows(cfg, view.summary, cfg.unit_system)
    penetration = row_in(solver_si, "peak contact penetration")
    assert penetration[1] == r"$6\times 10^{-7}$"
    assert penetration[2] == r"$\mathrm{m}$"

    cfg.units = MM_N_MPA.name
    _, mm = _response_rows(view.summary, cfg.unit_system)
    flat_mm = " | ".join(c for r in mm for c in r)
    deflection = row_in(mm, "max deflection")
    assert deflection[1] == "$1$" and deflection[2] == r"$\mathrm{mm}$"
    rigidity = row_in(mm, "apparent flexural rigidity")
    assert rigidity[1] == r"$2.5\times 10^{8}$"
    assert rigidity[2] == r"$\mathrm{N\,mm^{2}}$"
    assert r"\mathrm{m}$" not in flat_mm


def test_response_rows_use_tex_for_scientific_stress():
    """Large SI magnitudes become ``\\times 10^{n}``, never ``e+07``."""
    from sandwich3pb.units import SI

    view = FakeResults(make_config(), ".")
    view.summary["max_force_N"] = 6.7e7
    _, rows = _response_rows(view.summary, SI)
    flat = " | ".join(c for r in rows for c in r)
    peak = next(r for r in rows if str(r[0]).startswith("max force"))
    assert peak[1] == r"$6.7\times 10^{7}$"
    assert peak[2] == r"$\mathrm{N}$"
    assert "e+07" not in flat


def test_markdown_declares_units_and_math():
    cfg = make_config()
    view = FakeResults(cfg, ".")
    md = build_markdown(view)
    assert "units: si" in md
    assert r"\mathrm{Pa}" in md
    assert "$P_{\\max}$" in md


def test_failure_rows_cover_all_layers():
    cfg = make_config()
    view = FakeResults(cfg, ".")
    header, rows = _failure_rows(view.summary)
    assert len(rows) == 4  # 2 faces + 2 core criteria
    # 8 columns: the two per-row ``unit`` cells moved into the headers
    assert len(header) == 8
    assert all(len(r) == 8 for r in rows)
    assert any("Tsai-Wu" in r[3] for r in rows)
    # the old header glued the unit onto the Voigt indices
    assert not any("11,\\mathrm" in h for h in header)
    assert r"\sigma_{11}" in header[7]
    assert r"$\mathrm{" in header[6]      # hotspot unit
    assert r"$\mathrm{" in header[7]      # stress unit


def test_failure_onset_is_highlighted_in_markdown_and_response_table():
    cfg = make_config()
    view = FakeResults(cfg, ".")
    view.summary["failure_onset"] = {
        "step": 3, "travel_mm": 0.6, "force_N": 900.0,
        "deflection_mm": 0.5, "criterion_keys": ["layer_1:shear"],
    }
    md = build_markdown(view)
    assert "Predicted first failure — step 3" in md
    assert "layer_1:shear" in md
    # headline (a deliberate promotion of the table row), the response
    # table's one onset row, and the detailed callout -- the five onset
    # rows that used to repeat it in the table are gone
    assert md.lower().count("predicted first failure") == 3
    assert "onset travel" not in md and "onset criterion" not in md
    _, rows = _response_rows(view.summary, cfg.unit_system)
    step_row = next(
        row for row in rows
        if str(row[0]).startswith("predicted first failure")
    )
    assert step_row[1] == "$3$" and step_row[2] == "—"
    force_row = next(row for row in rows if row[0] == "onset force")
    assert force_row[1] == "$900$" and force_row[2] == r"$\mathrm{N}$"


def test_markdown_renders_all_sections(tmp_path):
    cfg = make_config()
    view = FakeResults(cfg, str(tmp_path))
    md = build_markdown(view)
    for section in ("Layup / stackup", "Model", "Structural response",
                    "Failure indices", "Solver and verification",
                    "Load–deflection"):
        assert section in md
    # run health is a status line, not a trailing section of its own
    assert "## Convergence" not in md
    assert "**Run:** all steps converged" in md
    assert "fibre orientation" in md
    # markdown table syntax present
    assert "| --- |" in md or "|---|" in md.replace(" ", "")


def test_md_table_format():
    md = _md_table(["a", "b"], [["1", "2"], ["3", "4"]])
    lines = md.splitlines()
    assert lines[0].startswith("| a | b |")
    assert len(lines) == 4


def test_values_and_units_live_in_separate_columns():
    """Units belong beside the value (key-value tables) or in the column
    header (columnar tables) -- never glued into the value itself."""
    cfg = make_config()
    view = FakeResults(cfg, ".")

    key_value = [
        _model_rows(cfg, view.summary),
        _response_rows(view.summary, cfg.unit_system),
        _solver_rows(cfg, view.summary, cfg.unit_system),
    ]
    for header, rows in key_value:
        unit_cols = [i for i, h in enumerate(header) if h == "unit"]
        assert unit_cols, header
        for row in rows:
            assert len(row) == len(header)
            for i, value in enumerate(row):
                if i in unit_cols:
                    continue
                # fmt_tex_qty glues value and unit with "\\," -- never here
                assert "\\," not in str(value), (header, row)

    columnar = [_layup_rows(cfg),
                _failure_rows(view.summary, cfg.unit_system)]
    for header, rows in columnar:
        assert not any(str(h) == "unit" for h in header), header
        unit_headers = [h for h in header if r"$\mathrm{" in str(h)]
        assert unit_headers, header          # every unit is delimited math
        for row in rows:
            assert len(row) == len(header)
            assert not any(r"\\mathrm" in str(c) for c in row), row


def test_kpi_rows_pick_headline_numbers():
    cfg = make_config()
    view = FakeResults(cfg, ".")
    labels = [str(r[0]) for r in _kpi_rows(view.summary, cfg.unit_system)]
    assert any(x.startswith("max force") for x in labels)
    assert any(x.startswith("predicted first failure") for x in labels)
    assert not any(x.startswith("peak Newton") for x in labels)


def test_visible_len_strips_inline_math():
    """PDF column widths must count rendered text, not raw TeX.

    Failure-table ``criterion`` cells mix prose with inline math; when
    the whole cell isn't wrapped in ``$...$`` the old estimate kept the
    TeX, inflated that column's weight and crushed its neighbours in
    the drawn table.
    """
    from sandwich3pb.report import _visible_len

    raw = r"shear utilisation $\frac{\vert\tau_{xz}}{\tau_c}$"
    assert len(raw) >= 45
    assert _visible_len(raw) <= len(raw) // 2
    assert _visible_len(r"Tsai-Wu index $\mathrm{TW}$") == 16
    assert _visible_len(r"$\mathrm{MPa}$") == 3
    assert _visible_len("layer_1") == 7
    assert _visible_len("—") == 2


# --------------------------------------------------------------------------
# content rules: TeX where it belongs, prose where it doesn't
# --------------------------------------------------------------------------


def test_markdown_kpi_block_is_a_real_table():
    """A GFM table needs a header *and* a delimiter row; without them
    the headline block renders as literal pipe text."""
    md = build_markdown(FakeResults(make_config(), "."))
    block = md.split("**At a glance:**", 1)[1].split("##", 1)[0]
    lines = [l for l in block.strip().splitlines() if l.startswith("|")]
    assert len(lines) >= 3, lines
    assert "quantity" in lines[0] and "value" in lines[0]
    assert re.fullmatch(r"\|[\s:|-]+\|", lines[1]), lines[1]


def test_no_raw_tex_outside_math_in_markdown():
    r"""Unit symbols are wrapped, never printed as bare ``\mathrm{m}``."""
    md = build_markdown(FakeResults(make_config(), "."))
    for line in md.splitlines():
        prose = re.sub(r"\$[^$\n]+\$", "", line)
        assert r"\mathrm" not in prose, line
        assert r"\times" not in prose, line


def test_boolean_and_counts_stay_prose():
    """``yes``/``no`` and step counts are words, not mathematics."""
    cfg = make_config()
    view = FakeResults(cfg, ".")
    header, rows = _model_rows(cfg, view.summary)
    assert "unit" in header          # key-value tables keep the column
    flat = " | ".join(str(c) for r in rows for c in r)
    assert "$yes$" not in flat and "$no$" not in flat

    half = next(r for r in rows if str(r[0]).startswith("half model"))
    assert half[1] == "no" and half[2] == "—"
    steps = next(r for r in rows if str(r[0]).startswith("load steps"))
    assert steps[1] == "5"
    assert any("element order" in str(r[0]) for r in rows)


def test_dropped_duplicate_rows_are_nowhere_in_the_report():
    cfg = make_config()
    view = FakeResults(cfg, ".")
    view.summary.update(
        deflection_at_max_force_mm=1.0,
        final_travel_mm=1.0,
        load_roller_force_N=1200.5,
        n_failed_steps=0,
        max_contact_penetration_mm=4e-4,
    )
    md = build_markdown(view)
    for dropped in (
        "deflection at max force",
        "final roller travel",
        "load-roller force",
        "failed load steps",
        "final-step contact penetration",
        "## Convergence",
    ):
        assert dropped not in md, dropped


# --------------------------------------------------------------------------
# report.tex
# --------------------------------------------------------------------------


def test_report_tex_makes_numeric_lists_breakable():
    r"""TeX cannot break a line inside ``$0.1, -0.02, 0.0015$``, so a
    ``p{}`` column holding one spills over its neighbour. The exporter
    must put the separating commas back into text mode."""
    from sandwich3pb.report import _breakable_math

    broken = _breakable_math(r"$0.1, -0.02, 0.0015$")
    assert broken == r"$0.1$, $-0.02$, $0.0015$"
    scientific = _breakable_math(
        r"$8.031\times 10^{7}, 5.781\times 10^{6}$"
    )
    assert scientific == r"$8.031\times 10^{7}$, $5.781\times 10^{6}$"
    # a single quantity is left alone, and so are commas *inside* braces
    assert _breakable_math(r"$0.0015$") == r"$0.0015$"
    assert _breakable_math(r"$\frac{a, b}{c}$") == r"$\frac{a, b}{c}$"
    assert _breakable_math(r"no maths, here") == r"no maths, here"


def test_report_tex_is_valid_latex_source():
    r"""The exporter must emit LaTeX, with its mathematics untouched.

    The previous version escaped the whole cell (``$E_x$`` became
    ``$E\_x$``), emitted Markdown ``##`` headings and put ``\title``
    after ``\begin{document}``.
    """
    cfg = make_config()
    view = FakeResults(cfg, ".")
    tex_src = build_latex(view)

    assert tex_src.index(r"\title") < tex_src.index(r"\begin{document}")
    assert tex_src.index(r"\maketitle") > tex_src.index(r"\begin{document}")
    assert "\n## " not in tex_src
    for section in ("At a glance", "Layup / stackup", "Structural response",
                    "Failure indices", "Model", "Solver and verification"):
        assert rf"\section*{{{section}}}" in tex_src

    # maths survives: single backslash rules, no blanket escaping
    for line in tex_src.splitlines():
        if re.search(r"(top|mid|bottom)rule", line):
            assert "\\\\" not in line, line
    assert r"\mathrm\{" not in tex_src
    assert r"E\_x" not in tex_src
    assert r"\times 10^{" in tex_src
    assert r"$\mathrm{Pa}$" in tex_src
    # ...while prose *is* escaped
    assert r"test\_case" in tex_src
    # figures are PDFs, guarded so a missing plot is not an error
    assert r"\IfFileExists{plots/" in tex_src
    assert ".svg" not in tex_src
    # numeric lists are split so the columns can wrap
    assert r"$600.2$, $600.3$" in tex_src      # support reactions
    assert r"$600.2, 600.3$" not in tex_src


PDFLATEX = shutil.which("pdflatex")


@pytest.mark.skipif(PDFLATEX is None, reason="pdflatex not installed")
def test_report_tex_compiles(tmp_path):
    """The written source must actually produce a PDF."""
    view = FakeResults(make_config(), str(tmp_path))
    (tmp_path / "report.tex").write_text(build_latex(view))
    result = subprocess.run(
        [PDFLATEX, "-interaction=nonstopmode", "-halt-on-error",
         "report.tex"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:]
    assert (tmp_path / "report.pdf").read_bytes().startswith(b"%PDF")
