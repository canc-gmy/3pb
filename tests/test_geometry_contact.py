"""Tests for mesh generation, tagging and the contact model."""

import numpy as np
import pytest

from sandwich3pb.geometry import (
    FACET_TAG_LOAD,
    FACET_TAG_SUPPORT,
    build_mesh_data,
    roller_axes,
)
from sandwich3pb.contact import build_contact_system
from tests.conftest import make_config


def _x_lines(md):
    return np.unique(md.x[:, 0])


def test_mesh_node_sharing_between_layers():
    cfg = make_config()
    md = build_mesh_data(cfg)
    # planes per x-y layer: face 2 elems -> 3 z-planes each, core 4 elems ->
    # 5 planes; interface planes are shared, so 3 + 5 + 3 - 2 = 9 planes
    nx_lines = _x_lines(md).size
    nxy = nx_lines * (cfg.mesh.elements_w + 1)
    assert md.x.shape[0] == nxy * (3 + 5 + 3 - 2)
    assert md.cells.shape[1] == 8


def test_roller_crown_stations_are_nodes():
    """Support-roller crowns (x = +/- span/2) and the load crown (x = 0)
    must be mesh stations so the crown is always a contact point."""
    cfg = make_config()
    md = build_mesh_data(cfg)
    xl = _x_lines(md)
    # 40 base elements + 2 inserted support crowns (0 is already a station)
    assert xl.size == cfg.mesh.elements_x + 3
    half_span = cfg.geometry.span / 2.0
    for station in (0.0, -half_span, half_span):
        assert np.any(np.isclose(xl, station, atol=1e-9))


def test_layer_tags():
    cfg = make_config()
    md = build_mesh_data(cfg)
    assert set(np.unique(md.cell_tags)) == {1, 2, 3}
    nx = _x_lines(md).size - 1
    nw = cfg.mesh.elements_w
    counts = [np.sum(md.cell_tags == t) for t in (1, 2, 3)]
    assert counts == [nx * nw * 2, nx * nw * 4, nx * nw * 2]


def test_contact_facets_tagged():
    cfg = make_config()
    md = build_mesh_data(cfg)
    xl = _x_lines(md)
    nw = cfg.mesh.elements_w

    def columns_touching(lo, hi):
        n = 0
        for a, b in zip(xl[:-1], xl[1:]):
            if (lo <= a <= hi) or (lo <= b <= hi):
                n += 1
        return n

    half_span = cfg.geometry.span / 2.0
    r_load = cfg.contact.roller_radius_load
    r_sup = cfg.contact.roller_radius_support
    # load roller tags only the TOP surface; support rollers only the bottom
    n_load_cols = columns_touching(-r_load, r_load)
    n_sup_cols = sum(
        columns_touching(s - r_sup, s + r_sup) for s in (-half_span, half_span)
    )
    assert (md.facet_tags == FACET_TAG_LOAD).sum() == n_load_cols * nw
    assert (md.facet_tags == FACET_TAG_SUPPORT).sum() == n_sup_cols * nw


def test_all_surface_facets_tagged():
    cfg = make_config()
    md = build_mesh_data(cfg)
    nx = _x_lines(md).size - 1
    nw = cfg.mesh.elements_w
    # every top and bottom facet carries exactly one tag
    total = sum((md.facet_tags == t).sum() for t in (1, 2, 3, 4))
    assert total == 2 * nx * nw


def test_gap_signs_undeformed():
    cfg = make_config()
    md = build_mesh_data(cfg)
    rollers = roller_axes(cfg)
    contact = build_contact_system(md, rollers, 3, 4, penalty=1e6)

    u_z = np.zeros(md.x.shape[0])
    gaps = contact.gaps_all(u_z)
    # at zero travel the midspan node (dx=0) of the load roller touches
    g_load = gaps["load"][0]
    assert g_load.min() == pytest.approx(0.0, abs=1e-12)
    assert g_load.max() > 0.0
    # supports: all gaps non-negative; the minimum equals R - sqrt(R^2 -
    # dx_min^2) at the node closest to the roller axis (no node at dx=0 on
    # the coarse test mesh)
    for pair, g in zip(contact.pairs["support"], gaps["support"]):
        assert g.min() >= 0.0
        dx_min = np.abs(pair.point_dx).min()
        expected = pair.roller.radius - np.sqrt(pair.roller.radius**2 - dx_min**2)
        assert g.min() == pytest.approx(expected, abs=1e-12)


def test_gap_negative_on_penetration():
    cfg = make_config()
    md = build_mesh_data(cfg)
    rollers = roller_axes(cfg)
    contact = build_contact_system(md, rollers, 3, 4, penalty=1e6)
    pair = contact.pairs["load"][0]
    # travel the roller 0.2 mm past the touching configuration (travel is
    # measured from first crown contact): with the surface held fixed the
    # crown-point gap is then exactly -0.2 mm
    pair.travel = 0.2
    u_z = np.zeros(md.x.shape[0])
    g = pair.gaps(u_z)
    assert g.min() < 0.0
    at_crown = g[np.abs(pair.point_dx) == np.abs(pair.point_dx).min()]
    assert at_crown.min() == pytest.approx(-0.2, abs=1e-9)
    # the gap opens away from the crown
    assert np.all(g >= at_crown.min() - 1e-12)


def test_pressure_zero_when_open():
    from sandwich3pb.contact import ContactPair
    from sandwich3pb.geometry import RollerAxis

    roller = RollerAxis(center_x=0, center_y=0, radius=5.0,
                        initial_gap=0.0, side="top")
    pair = ContactPair(
        roller=roller,
        point_ids=np.array([0, 1]),
        point_areas=np.array([1.0, 1.0]),
        point_dx=np.array([0.0, 1.0]),
        side="top",
    )
    g = np.array([0.5, 1.0])
    assert np.all(pair.pressures(g, 1e3) == 0.0)
    assert pair.resultant_force(g, 1e3) == 0.0


def test_resultant_force_balance_area():
    """Uniform penetration over known area must give p*A."""
    cfg = make_config()
    md = build_mesh_data(cfg)
    rollers = roller_axes(cfg)
    contact = build_contact_system(md, rollers, 3, 4, penalty=100.0)
    pair = contact.pairs["support"][0]
    u_z = np.zeros(md.x.shape[0])
    u_z[pair.point_ids] = -1.0  # 1 mm indentation everywhere
    # gaps: R + u - cyl(dx); penetration varies with dx; only check total
    # against trapezoid-consistent nodal integration
    g = pair.gaps(u_z)
    p = pair.pressures(g, 100.0)
    expected = float(np.dot(p, pair.point_areas))
    assert pair.resultant_force(g, 100.0) == pytest.approx(expected)


def test_al_multiplier_update_monotone():
    cfg = make_config()
    md = build_mesh_data(cfg)
    rollers = roller_axes(cfg)
    contact = build_contact_system(md, rollers, 3, 4, penalty=10.0)
    pair = contact.pairs["load"][0]
    # penetrate 0.1 mm everywhere except one point which stays open
    g = -0.1 * np.ones(pair.point_ids.size)
    g[1] = 0.2
    v1 = pair.update_multipliers(g, 10.0, rate=1.0)
    lam1 = pair.lambda_mult.copy()
    v2 = pair.update_multipliers(g, 10.0, rate=1.0)
    assert v1 == pytest.approx(0.1)
    assert np.all(pair.lambda_mult >= lam1)
    assert pair.lambda_mult[1] == 0.0  # open point stays at zero
    assert (pair.lambda_mult[0] > 0.0)
