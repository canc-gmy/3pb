"""Stress recovery, failure indices and derived engineering quantities.

Stress recovery: stresses are reported at the *grid nodes* (the layer-
sharing points). The hex8 elements are (isoparametric) trilinear, so the
strain at a grid node is evaluated directly from the element shape
functions at the node's local coordinates. Nodal stresses are averaged
per *stackup layer* (not globally) so that nodes shared between two layers
do not mix materials: every node stores one stress Voigt vector per
adjacent layer.

For quadratic displacement spaces (element_order = 2, superparametric on
the hex8 geometry) the same trilinear evaluation applies; the recovered
stress field is the exact equilibrium stress of the *projected* trilinear
part of the Q2 solution.

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


def recover_nodal_stress(
    mesh_data, u_dofs: np.ndarray, node2dof: np.ndarray,
    cell_layer_tags: np.ndarray, quads,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nodal Voigt stress per stackup layer.

    Returns
    -------
    stress : (n_nodes, n_layers, 6) stresses in global axes; entries are 0
             where a node does not belong to the layer (check ``counts``).
    counts : (n_nodes, n_layers) number of element-node samples averaged.
    """
    cells = np.asarray(mesh_data.cell_conn)
    n_cells, npe = cells.shape
    n_nodes = mesh_data.x.shape[0]
    n_layers = int(cell_layer_tags.max())

    X = mesh_data.x[cells]                              # (nc, 8, 3)
    # Jacobians at all 8 evaluation nodes: (nc, 8, 3, 3)
    J = np.einsum("cia,jib->cjab", X, _DN)
    Jinv = np.linalg.inv(J)
    dNdx = np.einsum("jib,cjba->cjia", _DN, Jinv)       # (nc, 8, 8, 3)

    # B operator rows (Voigt: 11,22,33,23,13,12)
    u_cell = u_dofs[node2dof[cells]]                    # (nc, 8, 3)
    eps = np.zeros((n_cells, 8, 6))
    for j in range(8):
        g = dNdx[:, j, :, :]                            # (nc, 8, 3)
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
    sig = np.einsum("cjd,lde->cjle", eps, C_g)          # then pick layer below
    layer_of_cell = cell_layer_tags - 1                 # 0-based

    stress = np.zeros((n_nodes, n_layers, 6))
    counts = np.zeros((n_nodes, n_layers), dtype=int)
    for j in range(8):
        nodes_j = cells[:, j]
        lay_j = layer_of_cell
        for l in range(n_layers):
            mask = lay_j == l
            if not mask.any():
                continue
            np.add.at(stress[:, l, :], nodes_j[mask], sig[mask, j, l, :])
            np.add.at(counts[:, l], nodes_j[mask], 1)
    valid = counts > 0
    stress[valid] /= counts[valid][..., None]
    return stress, counts


def midspan_profile(mesh_data, stress: np.ndarray, counts: np.ndarray,
                    cfg: Config, u_z_fe: Optional[np.ndarray] = None,
                    fe_points=None) -> Dict[str, np.ndarray]:
    """Stress and deflection along the thickness at mid-span.

    With ``u_z_fe`` (u_z at every FE point, order 2) the profile also
    reports the deflection at the mid-span FE points, so the quadratic
    deflection shape is visible between grid stations.
    """
    mid_x = 0.0 if not cfg.half_model else cfg.geometry.span / 2.0
    sel = np.isclose(mesh_data.x[:, 0], mid_x, atol=1e-6 * cfg.geometry.length)
    node_ids = np.where(sel)[0]
    order = np.argsort(mesh_data.x[node_ids, 2])
    node_ids = node_ids[order]

    z: List[float] = []
    sxx: List[float] = []
    sxz: List[float] = []
    layer_idx: List[int] = []
    for n in node_ids:
        for l in range(stress.shape[1]):
            if counts[n, l] > 0:
                z.append(mesh_data.x[n, 2])
                sxx.append(stress[n, l, 0])
                sxz.append(stress[n, l, 4])
                layer_idx.append(l)
    out = {
        "z": np.array(z),
        "sigma_xx": np.array(sxx),
        "tau_xz": np.array(sxz),
        "layer": np.array(layer_idx, dtype=int),
        "x": mesh_data.x[node_ids, 0],
        "y": mesh_data.x[node_ids, 1],
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

        # rotate stresses to material axes using the layer's quad
        if quads is not None:
            quad = quads[l]
        else:
            quad = localized_quad_form(mat, layer.fibre_orientation)
        sig_m = quad.stress_material_from_global(sig_l)  # (n, 6)

        if roles[l] == "core":
            shear = np.abs(sig_m[:, 4])  # tau_xz in material axes (theta=0 iso)
            crushing = np.maximum(0.0, -sig_m[:, 2])
            out[f"layer_{l}"] = {
                "role": "core",
                "material": layer.material,
                "max_shear_ratio": float((shear / mat.shear).max()),
                "max_crushing_ratio": float((crushing / mat.compression).max()),
            }
        else:
            tw = np.array([tsai_wu_3d(s, mat) for s in sig_m])
            out[f"layer_{l}"] = {
                "role": "face",
                "material": layer.material,
                "orientation_deg": layer.fibre_orientation,
                "max_tsai_wu": float(tw.max()),
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
