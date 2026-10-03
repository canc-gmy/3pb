"""Figure-level behaviour: predicted-onset markers and figure gating."""

from __future__ import annotations

import numpy as np

from sandwich3pb.figures import (
    fig_load_deflection,
    fig_stress_along_span,
    save_all_figures,
)
from sandwich3pb.units import FORCE, LENGTH, STRESS
from tests.conftest import make_config


class CaseView:
    """Minimal stand-in for CaseResults (cfg/summary/history/span_profile)."""


def make_view(with_onset: bool = True) -> CaseView:
    cfg = make_config()
    view = CaseView()
    view.cfg = cfg
    view.profile = None
    view.history = [
        {"step": i, "force_N": 100.0 * i, "deflection_mm": 0.05 * i,
         "travel_mm": 0.05 * i, "converged": True}
        for i in range(1, 6)
    ]
    xs = np.linspace(-100.0, 100.0, 9)
    view.span_profile = {
        "x": xs,
        "x_station": [xs] * 3,
        "station_layer": [0, 1, 2],
        "station_z": [0.75, 11.5, 22.25],
        "station_label": ["glass_epoxy", "pvc_foam", "glass_epoxy"],
        "sigma_xx": [np.linspace(-50.0, 50.0, 9) for _ in range(3)],
        "tau_xz": [np.linspace(-5.0, 5.0, 9) for _ in range(3)],
    }
    view.summary = {"force_gradient_N_per_mm": 2000.0}
    if with_onset:
        view.summary["failure_onset"] = {
            "step": 3, "travel_mm": 0.15, "force_N": 300.0,
            "deflection_mm": 0.15, "criterion_keys": ["layer_1:shear"],
            "criterion_values": {"layer_1:shear": 1.05},
            "failure_by_layer": {
                "layer_1": {
                    "role": "core", "material": "pvc_foam",
                    "max_shear_ratio": 1.05,
                    "max_shear_location_mm": [62.5, 0.0, 11.5],
                    "max_shear_stress_MPa": 8.0,
                },
            },
        }
    return view


def _markers(ax):
    return [(p.get_xdata()[0], p.get_ydata()[0]) for p in ax.lines
            if p.get_marker() == "*"]


def test_load_deflection_marks_onset_at_case_units():
    """The onset star must land at onset force/deflection in *display* units."""
    view = make_view()
    cfg = view.cfg
    fig = fig_load_deflection(view)
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert any("predicted first failure (step 3)" in label for label in labels)

    stars = _markers(fig.axes[0])
    assert len(stars) == 1
    expected = (
        cfg.unit_system.from_internal(LENGTH, 0.15),   # deflection
        cfg.unit_system.from_internal(FORCE, 300.0),    # force
    )
    np.testing.assert_allclose(stars[0], expected, rtol=0, atol=1e-15)


def test_load_deflection_without_onset_has_no_star():
    labels = fig_load_deflection(make_view(with_onset=False)) \
        .axes[0].get_legend_handles_labels()[1]
    assert not any("predicted first failure" in label for label in labels)


def test_span_figure_marks_onset_at_nearest_station():
    """The onset star sits on the onset layer's station nearest the hotspot."""
    view = make_view()
    cfg = view.cfg
    fig = fig_stress_along_span(view)
    shear_ax = fig.axes[1]

    stars = _markers(shear_ax)
    assert len(stars) == 1

    xs = cfg.unit_system.from_internal(LENGTH, view.span_profile["x"])
    onset_x = cfg.unit_system.from_internal(LENGTH, 62.5)
    nearest = int(np.argmin(np.abs(xs - onset_x)))
    expected = (
        xs[nearest],
        cfg.unit_system.from_internal(
            STRESS, view.span_profile["tau_xz"][1][nearest]
        ),
    )
    np.testing.assert_allclose(stars[0], expected, rtol=0, atol=1e-12)

    labels = [label for ax in fig.axes
              for label in ax.get_legend_handles_labels()[1]]
    assert any("first predicted failure location" in label for label in labels)


def test_span_figure_without_onset_layer_station_still_draws():
    """An onset on a layer with no station must not crash the figure."""
    view = make_view()
    view.span_profile["station_layer"] = [0, 2]
    view.span_profile["x_station"] = view.span_profile["x_station"][:2]
    view.span_profile["sigma_xx"] = view.span_profile["sigma_xx"][:2]
    view.span_profile["tau_xz"] = view.span_profile["tau_xz"][:2]
    view.span_profile["station_label"] = \
        view.span_profile["station_label"][:2]
    view.span_profile["station_z"] = view.span_profile["station_z"][:2]
    fig = fig_stress_along_span(view)
    labels = [label for ax in fig.axes
              for label in ax.get_legend_handles_labels()[1]]
    assert not any("first predicted failure location" in label
                   for label in labels)


def test_save_all_figures_skips_missing_profiles(tmp_path):
    view = make_view(with_onset=False)
    view.span_profile = {}
    saved = save_all_figures(view, str(tmp_path))
    assert "load_deflection" in saved
    assert "laminate_stackup" in saved
    assert "thickness_profile" not in saved
    assert "stress_along_span" not in saved
