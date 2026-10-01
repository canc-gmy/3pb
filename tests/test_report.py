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
    assert "fibre orientation (°)" in header
    assert rows[1][4] == "0"  # core has no orientation value of interest
    assert rows[0][2] == "glass_epoxy"


def test_layup_rows_show_90deg(tmp_path):
    cfg = make_config()
    cfg.stackup[0].fibre_orientation = 90.0
    _, rows = _layup_rows(cfg)
    assert rows[0][4] == "90"


def test_global_rows_contain_rigidity():
    cfg = make_config()
    view = FakeResults(cfg, ".")
    rows = _global_rows(view.summary)
    flat = " | ".join(cell for r in rows for cell in r)
    assert "flexural rigidity" in flat
    assert "force gradient" in flat


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
