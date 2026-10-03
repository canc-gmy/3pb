"""Stress recovery and failure-onset regressions without DOLFINx."""

import numpy as np
import pytest

from sandwich3pb.geometry import FEPoints, build_mesh_data
from sandwich3pb.materials import localized_quad_form
from sandwich3pb.postprocess import (
    beam_axis_profile,
    build_cell_fe_point_map,
    failure_indices,
    midspan_profile,
    recover_nodal_stress,
)
from sandwich3pb.results import first_failure_onset
from tests.conftest import make_config


def _recover(cfg, field, order):
    mesh = build_mesh_data(cfg)
    points = FEPoints.build(mesh, order)
    point_map = np.arange(points.xyz.shape[0] * 3).reshape(-1, 3)
    dofs = np.zeros(point_map.size)
    dofs[point_map] = field(points.xyz)
    node_map = point_map[:mesh.x.shape[0]]
    cell_points = build_cell_fe_point_map(mesh, points) if order == 2 else None
    quads = [
        localized_quad_form(cfg.materials[layer.material], layer.fibre_orientation)
        for layer in cfg.stackup
    ]
    stress, counts = recover_nodal_stress(
        mesh, dofs, node_map, mesh.cell_tags, quads,
        fe_points=points, point_map=point_map, element_order=order,
        cell_fe_points=cell_points,
    )
    return mesh, points, point_map, stress, counts, quads


@pytest.mark.parametrize("order", [1, 2])
def test_recover_stress_from_affine_axial_displacement(order):
    cfg = make_config()
    alpha = 2.0e-4
    mesh, _, _, stress, counts, quads = _recover(
        cfg, lambda xyz: np.column_stack((alpha * xyz[:, 0],
                                           np.zeros(len(xyz)),
                                           np.zeros(len(xyz)))), order
    )
    for layer, material_layer in enumerate(cfg.stackup):
        nodes = counts[:, layer] > 0
        expected = quads[layer].C @ np.array([alpha, 0, 0, 0, 0, 0])
        assert np.allclose(stress[nodes, layer], expected, rtol=1e-11, atol=1e-11)


def test_q2_recovery_keeps_quadratic_strain():
    cfg = make_config()
    beta = 3.0e-6
    _, _, _, stress, counts, _ = _recover(
        cfg,
        lambda xyz: np.column_stack((beta * xyz[:, 0] ** 2,
                                      np.zeros(len(xyz)), np.zeros(len(xyz)))),
        order=2,
    )
    # d(beta*x^2)/dx = 2*beta*x, including mid-side Q2 values.
    # Use the bottom face layer's nodal value at each x station.
    layer = 0
    nodes = np.flatnonzero(counts[:, layer] > 0)
    mesh = build_mesh_data(cfg)
    for node in nodes:
        expected_strain = 2.0 * beta * mesh.x[node, 0]
        # Compare with the layer's full Q2 constitutive response (including
        # Poisson coupling), not E1 times an assumed uniaxial stress state.
        quad = localized_quad_form(cfg.materials["glass_epoxy"], 0.0)
        expected = quad.C @ np.array([expected_strain, 0, 0, 0, 0, 0])
        assert stress[node, layer] == pytest.approx(expected, rel=1e-10)


def test_midspan_profile_uses_width_average_and_full_beam_midpoint_for_half_model():
    cfg = make_config()
    cfg.half_model = True
    mesh = build_mesh_data(cfg)
    layer = np.asarray(mesh.cell_tags)
    stress = np.zeros((len(mesh.x), len(cfg.stackup), 6))
    counts = np.ones((len(mesh.x), len(cfg.stackup)), dtype=int)
    stress[:, :, 0] = mesh.x[:, 1, None]
    stress[:, :, 4] = 2.0 * mesh.x[:, 1, None]
    profile = midspan_profile(mesh, stress, counts, cfg)
    assert np.allclose(profile["x"], 0.0)
    assert np.allclose(profile["sigma_xx"], 0.0)
    assert np.allclose(profile["tau_xz"], 0.0)

    span = beam_axis_profile(stress, counts, mesh, cfg)
    assert np.min(span["x"]) == pytest.approx(0.0)
    assert np.max(span["x"]) == pytest.approx(cfg.geometry.length / 2.0)


def test_failure_indices_report_the_hotspot_and_material_stress():
    cfg = make_config()
    mesh = build_mesh_data(cfg)
    stress = np.zeros((len(mesh.x), len(cfg.stackup), 6))
    counts = np.ones((len(mesh.x), len(cfg.stackup)), dtype=int)
    peak_node = 0
    stress[peak_node, 0, 0] = 9000.0
    failure = failure_indices(cfg, mesh, stress, counts)
    face = failure["layer_0"]
    hotspot = np.asarray(face["max_tsai_wu_location_mm"])
    assert np.allclose(hotspot, mesh.x[peak_node])
    assert len(face["max_tsai_wu_stress_material_MPa"]) == 6


def test_first_failure_onset_ignores_unconverged_steps_and_reports_all_crossings():
    history = [
        {"step": 1, "converged": False, "failure_by_layer": {
            "layer_0": {"role": "face", "max_tsai_wu": 2.0}}},
        {"step": 2, "converged": True, "travel_mm": 0.4, "force_N": 20.0,
         "deflection_mm": 0.3, "failure_by_layer": {
             "layer_0": {"role": "face", "max_tsai_wu": 1.1},
             "layer_1": {"role": "core", "max_shear_ratio": 1.0,
                         "max_crushing_ratio": 0.2}}},
    ]
    onset = first_failure_onset(history)
    assert onset["step"] == 2
    assert set(onset["criterion_keys"]) == {"layer_0:tsai_wu", "layer_1:shear"}
