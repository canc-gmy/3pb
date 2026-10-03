"""Tests for report generation (no FEM required)."""

import os

import numpy as np
import pytest

from sandwich3pb.report import (
    _failure_rows,
    _global_rows,
    _kpi_rows,
    _layup_rows,
    _md_table,
    _model_rows,
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
    assert rows[1][5] == "$0$"  # core has no orientation value of interest
    assert rows[0][2] == "glass_epoxy"


def test_layup_units_follow_the_case_file():
    """Unit cells must announce the case file's units, not the internal ones."""
    from sandwich3pb.units import MM_N_MPA

    cfg = make_config()
    header, si_rows = _layup_rows(cfg)
    assert header[4] == header[7] == "unit"
    si_units = [si_rows[0][4], si_rows[0][7]]
    assert any(r"\mathrm{m}" in c for c in si_units)   # SI default
    assert any(r"\mathrm{Pa}" in c for c in si_units)
    assert not any(r"\mathrm{mm}" in c for c in si_units)

    cfg.units = MM_N_MPA.name
    _, mm_rows = _layup_rows(cfg)
    mm_units = [mm_rows[0][4], mm_rows[0][7]]
    assert any(r"\mathrm{mm}" in c for c in mm_units)
    assert any(r"\mathrm{MPa}" in c for c in mm_units)


def test_layup_rows_show_90deg(tmp_path):
    cfg = make_config()
    cfg.stackup[0].fibre_orientation = 90.0
    _, rows = _layup_rows(cfg)
    assert rows[0][5] == "$90$"


def test_global_rows_contain_rigidity():
    cfg = make_config()
    view = FakeResults(cfg, ".")
    _, rows = _global_rows(view.summary, cfg.unit_system)
    flat = " | ".join(cell for r in rows for cell in r)
    assert "flexural rigidity" in flat
    assert "force gradient" in flat


def test_global_rows_convert_to_case_units():
    """Internal mm/N/MPa values must be presented in the case file's units,
    with the unit sitting in its own column beside the value."""
    from sandwich3pb.units import MM_N_MPA

    cfg = make_config()
    view = FakeResults(cfg, ".")

    def row_in(rows, prefix):
        return next(r for r in rows if str(r[0]).startswith(prefix))

    _, si = _global_rows(view.summary, cfg.unit_system)
    # 1200.5 N: N is the force unit in both systems
    force = row_in(si, "max force")
    assert force[1] == r"$1200$" and force[2] == r"$\mathrm{N}$"
    # 2.5e8 N*mm^2 -> 250 N*m^2 in SI
    rigidity = row_in(si, "apparent flexural rigidity")
    assert rigidity[1] == r"$250$" and rigidity[2] == r"$\mathrm{N\,m^{2}}$"
    # 0.8 mm -> 8e-4 m (below the readability threshold, so scientific)
    deflection = row_in(si, "deflection at max force")
    assert deflection[1] == r"$8\times 10^{-4}$"
    assert deflection[2] == r"$\mathrm{m}$"

    cfg.units = MM_N_MPA.name
    _, mm = _global_rows(view.summary, cfg.unit_system)
    flat_mm = " | ".join(c for r in mm for c in r)
    deflection = row_in(mm, "deflection at max force")
    assert deflection[1] == "$0.8$" and deflection[2] == r"$\mathrm{mm}$"
    rigidity = row_in(mm, "apparent flexural rigidity")
    assert rigidity[2] == r"$\mathrm{N\,mm^{2}}$"
    assert r"\mathrm{m}$" not in flat_mm


def test_global_rows_use_tex_for_scientific_stress():
    """Large SI magnitudes become ``\\times 10^{n}``, never ``e+07``."""
    from sandwich3pb.units import SI

    view = FakeResults(make_config(), ".")
    view.summary["load_roller_force_N"] = 6.7e7
    _, rows = _global_rows(view.summary, SI)
    flat = " | ".join(c for r in rows for c in r)
    roller = next(r for r in rows if str(r[0]).startswith("load-roller force"))
    assert roller[1] == r"$6.7\times 10^{7}$"
    assert roller[2] == r"$\mathrm{N}$"
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
    assert len(header) == 10
    assert any("Tsai-Wu" in r[3] for r in rows)


def test_failure_onset_is_highlighted_in_markdown_and_global_table():
    cfg = make_config()
    view = FakeResults(cfg, ".")
    view.summary["failure_onset"] = {
        "step": 3, "travel_mm": 0.6, "force_N": 900.0,
        "deflection_mm": 0.5, "criterion_keys": ["layer_1:shear"],
    }
    md = build_markdown(view)
    assert "Predicted first failure — step 3" in md
    assert "layer_1:shear" in md
    _, rows = _global_rows(view.summary, cfg.unit_system)
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
    for section in ("Layup / stackup", "Model", "Global results",
                    "Failure indices", "Load–deflection"):
        assert section in md
    assert "fibre orientation" in md
    # markdown table syntax present
    assert "| --- |" in md or "|---|" in md.replace(" ", "")


def test_md_table_format():
    md = _md_table(["a", "b"], [["1", "2"], ["3", "4"]])
    lines = md.splitlines()
    assert lines[0].startswith("| a | b |")
    assert len(lines) == 4


def test_values_and_units_live_in_separate_columns():
    """No value cell may carry the unit: units belong to the unit column."""
    cfg = make_config()
    view = FakeResults(cfg, ".")
    tables = [
        _layup_rows(cfg),
        _model_rows(cfg, view.summary),
        _global_rows(view.summary, cfg.unit_system),
        _failure_rows(view.summary, cfg.unit_system),
    ]
    for header, rows in tables:
        unit_cols = [i for i, h in enumerate(header) if h == "unit"]
        assert unit_cols, header
        for row in rows:
            assert len(row) == len(header)
            for i, value in enumerate(row):
                if i in unit_cols:
                    continue
                # fmt_tex_qty glues value and unit with "\\," -- never here
                assert "\\," not in str(value), (header, row)


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
