"""Stress recovery, failure indices and derived engineering quantities.

Stress recovery: stresses are reported at the *grid nodes* (the layer-
sharing points). The hex8 elements are (isoparametric) trilinear, so the
strain at a grid node is evaluated directly from the element shape
functions at the node's local coordinates. Nodal stresses are averaged
per *stackup layer* (not globally) so that nodes shared between two layers
do not mix materials.

For quadratic displacement spaces (element_order = 2, superparametric on
the hex8 geometry), shape-function gradients are evaluated from the full
Q2 displacement field at the physical grid corners. This retains the
quadratic solution's strain rather than projecting displacement onto the
corner nodes first.

Through-thickness profiles are taken at the mid-span cross-section (at the
symmetry plane for half models).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .materials import (
    MaterialSpec,
    core_crushing_ratio,
    core_shear_ratio,
    localized_quad_form,
    tsai_wu_3d,
    von_mises,
)

# Shape function derivatives of the trilinear hex at its 8 corners.
# Node j local coords (xi_j, eta_j, zeta_j) follow the basix hexahedron
# vertex convention (lexicographic order), matching the cell connectivity
# emitted by geometry.build_mesh_data.
_SIGN = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [-1, +1, -1],
        [+1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [-1, +1, +1],
        [+1, +1, +1],
    ],
    dtype=float,
)

_Q2_NODES = np.array(
    [(x, y, z) for x in (-1.0, 0.0, 1.0)
     for y in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)],
    dtype=float,
)


def _hex_shape_derivs() -> np.ndarray:
    """dN/dxi evaluated at the 8 nodes: shape (8_eval, 8_fn, 3)."""
    dN = np.zeros((8, 8, 3))
    for j in range(8):          # evaluation point
        for i in range(8):      # shape function
            s_i = _SIGN[i]
            s_j = _SIGN[j]
            dN[j, i, 0] = 0.125 * s_i[0] * (1 + s_i[1] * s_j[1]) * (1 + s_i[2] * s_j[2])
            dN[j, i, 1] = 0.125 * (1 + s_i[0] * s_j[0]) * s_i[1] * (1 + s_i[2] * s_j[2])
            dN[j, i, 2] = 0.125 * (1 + s_i[0] * s_j[0]) * (1 + s_i[1] * s_j[1]) * s_i[2]
    return dN


_DN = _hex_shape_derivs()


def _q2_shape_derivs() -> np.ndarray:
    """Q2 basis derivatives at hex corners; basis nodes are lexicographic."""
    derivs = np.empty((8, 27, 3), dtype=float)
    for j, point in enumerate(_SIGN):
        values = np.array([
            [0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)]
            for x in point
        ])
        slopes = np.array([[x - 0.5, -2.0 * x, x + 0.5] for x in point])
        for i, node in enumerate(_Q2_NODES):
            a, b, c = (int(v + 1) for v in node)
            derivs[j, i, 0] = slopes[0, a] * values[1, b] * values[2, c]
            derivs[j, i, 1] = values[0, a] * slopes[1, b] * values[2, c]
            derivs[j, i, 2] = values[0, a] * values[1, b] * slopes[2, c]
    return derivs


_Q2_DN = _q2_shape_derivs()


def build_cell_fe_point_map(mesh_data, fe_points) -> np.ndarray:
    """Return each cell's 27 Q2 point ids in tensor-product basis order."""
    X = mesh_data.x[np.asarray(mesh_data.cell_conn)]
    starts = np.column_stack((
        np.searchsorted(fe_points.xs, X[:, 0, 0]),
        np.searchsorted(fe_points.ys, X[:, 0, 1]),
        np.searchsorted(fe_points.zs, X[:, 0, 2]),
    ))
    offsets = _Q2_NODES.astype(np.intp) + 1
    return fe_points.station_grid[
        starts[:, 0, None] + offsets[None, :, 0],
        starts[:, 1, None] + offsets[None, :, 1],
        starts[:, 2, None] + offsets[None, :, 2],
    ]


def recover_nodal_stress(
    mesh_data, u_dofs: np.ndarray, node2dof: np.ndarray,
    cell_layer_tags: np.ndarray, quads, fe_points=None,
    point_map: Optional[np.ndarray] = None, element_order: int = 1,
    cell_fe_points: Optional[np.ndarray] = None,
    cell_sizes: Optional[np.ndarray] = None,
    layer_cell_ids: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nodal Voigt stress per stackup layer.

    Returns
    -------
    stress : (n_nodes, n_layers, 6) stresses in global axes; entries are 0
             where a node does not belong to the layer (check ``counts``).
    counts : (n_nodes, n_layers) number of element-node samples averaged.
    """
    cells = np.asarray(mesh_data.cell_conn)
    n_cells = cells.shape[0]
    n_nodes = mesh_data.x.shape[0]
    n_layers = int(cell_layer_tags.max())

    X = mesh_data.x[cells]                              # (nc, 8, 3)
    # The structured mesh consists of axis-aligned affine hexes. Compute
    # physical side lengths once; the reference-to-physical Jacobian is
    # diagonal and constant, so no per-node matrix inversions are needed.
    lengths = cell_sizes if cell_sizes is not None else np.column_stack((
        X[:, 1, 0] - X[:, 0, 0],
        X[:, 2, 1] - X[:, 0, 1],
        X[:, 4, 2] - X[:, 0, 2],
    ))
    inv_jac = 2.0 / lengths

    if element_order == 2:
        if fe_points is None or point_map is None:
            raise ValueError("Q2 stress recovery requires FE points and point map")
        if cell_fe_points is None:
            cell_fe_points = build_cell_fe_point_map(mesh_data, fe_points)
        dN_ref = _Q2_DN
        u_cell = u_dofs[point_map[cell_fe_points]]
    else:
        dN_ref = _DN
        u_cell = u_dofs[node2dof[cells]]              # (nc, 8, 3)

    # B operator rows (Voigt: 11,22,33,23,13,12)
    eps = np.zeros((n_cells, 8, 6))
    for j in range(8):
        g = dN_ref[j][None, :, :] * inv_jac[:, None, :]
        e11 = np.einsum("ca,ca->c", g[:, :, 0], u_cell[:, :, 0])
        e22 = np.einsum("ca,ca->c", g[:, :, 1], u_cell[:, :, 1])
        e33 = np.einsum("ca,ca->c", g[:, :, 2], u_cell[:, :, 2])
        g23 = np.einsum("ca,ca->c", g[:, :, 2], u_cell[:, :, 1]) + \
              np.einsum("ca,ca->c", g[:, :, 1], u_cell[:, :, 2])
        g13 = np.einsum("ca,ca->c", g[:, :, 2], u_cell[:, :, 0]) + \
              np.einsum("ca,ca->c", g[:, :, 0], u_cell[:, :, 2])
        g12 = np.einsum("ca,ca->c", g[:, :, 1], u_cell[:, :, 0]) + \
              np.einsum("ca,ca->c", g[:, :, 0], u_cell[:, :, 1])
        eps[:, j, 0] = e11
        eps[:, j, 1] = e22
        eps[:, j, 2] = e33
        eps[:, j, 3] = g23
        eps[:, j, 4] = g13
        eps[:, j, 5] = g12

    C_g = np.stack([q.C for q in quads])                # (n_layers, 6, 6)
    layer_of_cell = np.asarray(cell_layer_tags, dtype=np.intp) - 1
    # Each cell uses its own layer stiffness; avoid a temporary with a
    # separate copy of every layer's stress for every cell.
    sig = np.einsum(
        "cjd,cde->cje", eps, C_g[layer_of_cell], optimize=True
    )

    stress_sum = np.zeros((n_layers, n_nodes, 6))
    counts = np.zeros((n_layers, n_nodes), dtype=np.int64)
    if layer_cell_ids is None:
        layer_cell_ids = [
            np.flatnonzero(layer_of_cell == layer) for layer in range(n_layers)
        ]
    for layer, cell_ids in enumerate(layer_cell_ids):
        if cell_ids.size == 0:
            continue
        layer_cells = cells[cell_ids]
        layer_sig = sig[cell_ids]
        node_ids = layer_cells.reshape(-1)
        np.add.at(stress_sum[layer], node_ids, layer_sig.reshape(-1, 6))
        np.add.at(counts[layer], node_ids, 1)
    valid = counts > 0
    stress_sum[valid] /= counts[valid, None]
    return np.moveaxis(stress_sum, 0, 1), counts.T


def midspan_profile(mesh_data, stress: np.ndarray, counts: np.ndarray,
                    cfg: Config, u_z_fe: Optional[np.ndarray] = None,
                    fe_points=None) -> Dict[str, np.ndarray]:
    """Stress and deflection along the thickness at mid-span.

    With ``u_z_fe`` (u_z at every FE point, order 2) the profile also
    reports the deflection at the mid-span FE points, so the quadratic
    deflection shape is visible between grid stations.
    """
    # Mid-span is the load roller axis in both full and symmetry models.
    mid_x = 0.0
    sel = np.isclose(mesh_data.x[:, 0], mid_x, atol=1e-6 * cfg.geometry.length)
    node_ids = np.where(sel)[0]
    order = np.argsort(mesh_data.x[node_ids, 2])
    node_ids = node_ids[order]

    z: List[float] = []
    sxx: List[float] = []
    sxz: List[float] = []
    layer_idx: List[int] = []
    # The plotted profile is a width-average at each thickness station;
    # aggregating by both z and layer avoids duplicate, out-of-order traces.
    z_values = np.unique(mesh_data.x[node_ids, 2])
    for z_value in z_values:
        at_z = node_ids[np.isclose(mesh_data.x[node_ids, 2], z_value)]
        for layer in range(stress.shape[1]):
            layer_nodes = at_z[counts[at_z, layer] > 0]
            if layer_nodes.size == 0:
                continue
            z.append(float(z_value))
            sxx.append(float(stress[layer_nodes, layer, 0].mean()))
            sxz.append(float(stress[layer_nodes, layer, 4].mean()))
            layer_idx.append(layer)
    out = {
        "z": np.asarray(z),
        "sigma_xx": np.asarray(sxx),
        "tau_xz": np.asarray(sxz),
        "layer": np.asarray(layer_idx, dtype=int),
        "x": np.zeros(len(z)),
        "y": np.zeros(len(z)),
    }
    if u_z_fe is not None and fe_points is not None:
        pm = np.isclose(fe_points.xyz[:, 0], mid_x,
                        atol=1e-6 * cfg.geometry.length)
        pids = np.where(pm)[0]
        order = np.argsort(fe_points.xyz[pids, 2])
        pids = pids[order]
        out["z_fe"] = fe_points.xyz[pids, 2]
        out["u_z_fe"] = u_z_fe[pids]
    return out


def beam_axis_profile(stress: np.ndarray, counts: np.ndarray, mesh_data,
                      cfg: Config) -> Dict[str, object]:
    """Stress along the beam axis at representative layer stations.

    Three-point bending puts bending stress and shear stress at different
    places along the span, so a single mid-span section hides half the
    picture. This samples the same recovered nodal stress along the whole
    beam at the mid-thickness of the bottom face, the core (if any) and
    the top face -- which is how the classic pair of diagrams (sigma_xx
    peaking at mid-span, tau_xz peaking at the supports) is drawn.

    Stations whose nodes are missing are skipped, so the result stays
    usable for coarse meshes and half models.
    """
    bounds = cfg.layer_z_bounds
    roles = cfg.resolve_roles()
    core = [i for i, r in enumerate(roles) if r == "core"]
    picks = [0]
    if core:
        picks.append(core[len(core) // 2])
    picks.append(len(cfg.stackup) - 1)

    out: Dict[str, object] = {
        "x": None,
        "x_station": [],
        "station_layer": [],
        "station_z": [],
        "station_label": [],
        "sigma_xx": [],
        "tau_xz": [],
    }
    tol = 1.0e-6 * max(bounds[-1], 1.0)
    for layer in picks:
        z_mid = 0.5 * (bounds[layer] + bounds[layer + 1])
        candidates = np.flatnonzero(counts[:, layer] > 0)
        if candidates.size == 0:
            continue
        # Use the mesh station nearest the layer midpoint. This works for
        # coarse/oddly subdivided layers where the exact midpoint is absent.
        station_z = np.unique(mesh_data.x[candidates, 2])
        z_pick = station_z[np.argmin(np.abs(station_z - z_mid))]
        nodes = candidates[np.abs(mesh_data.x[candidates, 2] - z_pick) < tol]
        if nodes.size == 0:
            continue
        # A station picks up every node on that z-plane, which is one per
        # element column across the width. Averaging them collapses each
        # x station to a single value -- otherwise the "line" doubles back
        # on itself and the CSV carries duplicate x rows.
        xs = mesh_data.x[nodes, 0]
        uniq, inverse = np.unique(xs, return_inverse=True)
        per_station = np.bincount(inverse).astype(float)
        sxx = np.bincount(
            inverse, weights=stress[nodes, layer, 0]
        ) / per_station
        sxz = np.bincount(
            inverse, weights=stress[nodes, layer, 4]
        ) / per_station
        if out["x"] is None:
            out["x"] = uniq.copy()
        out["x_station"].append(uniq)
        out["station_layer"].append(layer)
        out["station_z"].append(float(z_pick))
        out["station_label"].append(cfg.stackup[layer].material)
        out["sigma_xx"].append(sxx)
        out["tau_xz"].append(sxz)
    return out


def failure_indices(cfg: Config, mesh_data, stress: np.ndarray,
                    counts: np.ndarray,
                    quads: Optional[List] = None) -> Dict[str, Dict]:
    """Per-layer failure maxima over the model (material-axis stresses).

    ``quads`` (localized stiffness of each layer, in stackup order) is
    reused when given instead of rebuilding the rotation per layer.
    """
    roles = cfg.resolve_roles()
    out: Dict[str, Dict] = {}
    n_layers = stress.shape[1]

    for l in range(n_layers):
        layer = cfg.stackup[l]
        mat = cfg.materials[layer.material]
        mask = counts[:, l] > 0
        if not mask.any():
            continue
        sig_l = stress[mask, l, :]  # global axes
        node_ids = np.flatnonzero(mask)
        coords = np.asarray(mesh_data.x)

        def location(index: int) -> List[float]:
            return [float(v) for v in coords[node_ids[index]]]

        # rotate stresses to material axes using the layer's quad
        if quads is not None:
            quad = quads[l]
        else:
            quad = localized_quad_form(mat, layer.fibre_orientation)
        sig_m = quad.stress_material_from_global(sig_l)  # (n, 6)

        if roles[l] == "core":
            shear = np.abs(sig_m[:, 4])  # tau_xz in material axes (theta=0 iso)
            crushing = np.maximum(0.0, -sig_m[:, 2])
            shear_i = int(np.argmax(shear))
            crushing_i = int(np.argmax(crushing))
            out[f"layer_{l}"] = {
                "role": "core",
                "material": layer.material,
                "max_shear_ratio": float(shear[shear_i] / mat.shear),
                "max_shear_location_mm": location(shear_i),
                "max_shear_stress_MPa": float(sig_m[shear_i, 4]),
                "max_crushing_ratio": float(crushing[crushing_i] / mat.compression),
                "max_crushing_location_mm": location(crushing_i),
                "max_crushing_stress_MPa": float(sig_m[crushing_i, 2]),
            }
        else:
            tw = np.array([tsai_wu_3d(s, mat) for s in sig_m])
            tw_i = int(np.argmax(tw))
            out[f"layer_{l}"] = {
                "role": "face",
                "material": layer.material,
                "orientation_deg": layer.fibre_orientation,
                "max_tsai_wu": float(tw[tw_i]),
                "max_tsai_wu_location_mm": location(tw_i),
                "max_tsai_wu_stress_material_MPa": sig_m[tw_i].tolist(),
            }
    return out


def layup_flexural_rigidity(cfg: Config) -> Dict[str, float]:
    """Analytic beam flexural rigidity from the stackup (classic laminate).

    Uses the uniaxial modulus of each (possibly rotated) layer along the beam
    axis, the parallel-axis theorem, and perfect bonding. Returns per-width
    (N·mm/mm) and total (N·mm²) rigidity plus the neutral axis position.
    """
    widths: List[float] = []
    centroids: List[float] = []
    E_x: List[float] = []
    z0 = 0.0
    for layer in cfg.stackup:
        mat = cfg.materials[layer.material]
        quad = localized_quad_form(mat, layer.fibre_orientation)
        S_g = np.linalg.inv(quad.C)
        E_x.append(1.0 / S_g[0, 0])  # uniaxial modulus along the beam axis
        centroids.append(z0 + layer.thickness / 2.0)
        widths.append(layer.thickness)
        z0 += layer.thickness

    E_t = np.array(E_x) * np.array(widths)
    z_na = float(np.dot(E_t, centroids) / E_t.sum())
    EI_pw = float(
        sum(
            E * (t**3 / 12.0 + t * (zc - z_na) ** 2)
            for E, t, zc in zip(E_x, widths, centroids)
        )
    )
    return {
        "flexural_rigidity_per_width": EI_pw,   # N·mm (per mm width)
        "flexural_rigidity": EI_pw * cfg.geometry.width,  # N·mm²
        "neutral_axis_z": z_na,
    }
