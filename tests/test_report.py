"""Tests for report generation (no FEM required)."""

import os

import numpy as np
import pytest

from sandwich3pb.report import (
    _failure_rows,
    _global_rows,
    _layup_rows,
    _md_table,
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
            "max_tsai_wu_faces": 0.12,
            "max_core_shear_ratio": 0.35,
            "max_core_crushing_ratio": 0.20,
            "n_steps": 5,
            "all_steps_converged": True,
            "n_nodes": 999,
            "n_dofs": 2997,
            "assembly_time_s": 1.0,
            "solve_time_s": 2.0,
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
    """A header must announce the case file's units, not the internal ones."""
    from sandwich3pb.units import MM_N_MPA, SI

    cfg = make_config()
    si_header, _ = _layup_rows(cfg)
    assert any(r"\mathrm{m}" in h for h in si_header)   # SI default
    assert any(r"\mathrm{Pa}" in h for h in si_header)
    assert not any(r"\mathrm{mm}" in h for h in si_header)

    cfg.units = MM_N_MPA.name
    mm_header, _ = _layup_rows(cfg)
    assert any(r"\mathrm{mm}" in h for h in mm_header)
    assert any(r"\mathrm{MPa}" in h for h in mm_header)


def test_layup_rows_show_90deg(tmp_path):
    cfg = make_config()
    cfg.stackup[0].fibre_orientation = 90.0
    _, rows = _layup_rows(cfg)
    assert rows[0][4] == "$90$"


def test_global_rows_contain_rigidity():
    cfg = make_config()
    view = FakeResults(cfg, ".")
    rows = _global_rows(view.summary, cfg.unit_system)
    flat = " | ".join(cell for r in rows for cell in r)
    assert "flexural rigidity" in flat
    assert "force gradient" in flat


def test_global_rows_convert_to_case_units():
    """Internal mm/N/MPa values must be presented in the case file's units."""
    from sandwich3pb.units import MM_N_MPA

    cfg = make_config()
    view = FakeResults(cfg, ".")

    si = _global_rows(view.summary, cfg.unit_system)
    flat_si = " | ".join(c for r in si for c in r)
    # 1200.5 N is 1.2005 kN-scale: N is the unit in both systems
    assert r"$1200\,\mathrm{N}$" in flat_si
    # 2.5e8 N*mm^2 -> 250 N*m^2
    assert r"250" in flat_si and r"\mathrm{N\,m^{2}}" in flat_si
    # 0.8 mm -> 8e-4 m (below the readability threshold, so scientific)
    assert r"$8\times 10^{-4}\,\mathrm{m}$" in flat_si

    cfg.units = MM_N_MPA.name
    mm = _global_rows(view.summary, cfg.unit_system)
    flat_mm = " | ".join(c for r in mm for c in r)
    assert r"$0.8\,\mathrm{mm}$" in flat_mm
    assert r"\mathrm{N\,mm^{2}}" in flat_mm
    assert r"\mathrm{m}$" not in flat_mm


def test_global_rows_use_tex_for_scientific_stress():
    """Large SI magnitudes become ``\\times 10^{n}``, never ``e+07``."""
    from sandwich3pb.units import SI

    view = FakeResults(make_config(), ".")
    view.summary["load_roller_force_N"] = 6.7e7
    rows = _global_rows(view.summary, SI)
    flat = " | ".join(c for r in rows for c in r)
    assert r"$6.7\times 10^{7}\,\mathrm{N}$" in flat
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
    assert any("Tsai-Wu" in r[3] for r in rows)


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
