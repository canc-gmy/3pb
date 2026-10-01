"""Structured hexahedral mesh generation with layer and surface tagging.

The sandwich beam occupies
    x in [-L/2, L/2]  (beam axis; length L)
    y in [-w/2,  w/2] (width)
    z in [0, t]       (through thickness, stackup given bottom -> top)

The mesh is a conforming structured hex grid: layers share interface nodes,
which enforces perfect bonding between facesheets and core.

Cell tags (cell_data "layer"):
    k -> stackup layer k-1 (1-based layer index; the stackup is ordered
    bottom -> top, so tag 1 is the bottom-most layer). Every layer gets its
    own tag so that multi-ply facesheets with different fibre orientations
    receive the correct stiffness.

Facet tags (facet_data "surfaces"):
    1 -> bottom surface (z = 0)
    2 -> top surface (z = t)
    3 -> load roller contact zone (top surface under the load roller)
    4 -> support roller contact zones (bottom surface under the supports)
    5 -> symmetry plane (half models only, x = 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config

# dolfinx is imported lazily by build_dolfinx_mesh() so that config/material
# utilities stay usable without a FEniCSx installation.


CELL_TAG_BOTTOM_FACE = 1
CELL_TAG_CORE = 2
CELL_TAG_TOP_FACE = 3

FACET_TAG_BOTTOM = 1
FACET_TAG_TOP = 2
FACET_TAG_LOAD = 3
FACET_TAG_SUPPORT = 4
FACET_TAG_SYMMETRY = 5


@dataclass
class RollerAxis:
    """One rigid roller: an infinite cylinder with axis along the width (y)."""

    center_x: float
    center_y: float
    radius: float
    initial_gap: float   # distance from deformed surface datum to cylinder skin
    side: str            # "top" or "bottom"

    @property
    def inward_normal_sign(self) -> float:
        """Sign of the z-component of the inward (pushing) normal."""
        return -1.0 if self.side == "top" else +1.0


@dataclass
class FEPoints:
    """Finite-element point set: the points where displacement dofs live.

    Order 1: the FE points are exactly the structured grid nodes and the
    first ``n_grid`` ids are the mesh node ids.
    Order 2: quadratic Lagrange displacement on the (linear hex8) geometry -
    a superparametric formulation that removes hex8 shear locking. The FE
    point set is the grid refined by 2 (mid-edge / mid-face / cell-centre
    points added); corner points keep their grid node ids.
    """

    xyz: np.ndarray            # (n_fe, 3) point coordinates
    n_grid: int                # ids [0, n_grid) are the grid (corner) nodes
    xs: np.ndarray             # refined station coordinates per axis
    ys: np.ndarray
    zs: np.ndarray
    station_grid: np.ndarray   # (na, nb, nc) fe point id per station triple
    station_of_id: np.ndarray  # (n_fe, 3) station index per axis per point
    refinement: int            # 1 or 2

    @classmethod
    def build(cls, mesh_data: "MeshData", order: int) -> "FEPoints":
        xl = np.unique(mesh_data.x[:, 0])
        yl = np.unique(mesh_data.x[:, 1])
        zl = np.unique(mesh_data.x[:, 2])
        r = 1 if order <= 1 else 2

        def stations(lines: np.ndarray) -> np.ndarray:
            if r == 1:
                return lines
            mids = 0.5 * (lines[:-1] + lines[1:])
            return np.sort(np.unique(np.concatenate([lines, mids])))

        xs, ys, zs = stations(xl), stations(yl), stations(zl)
        na, nb, nc = len(xs), len(ys), len(zs)
        n_grid = mesh_data.x.shape[0]
        nx1, ny1 = len(xl), len(yl)

        station_grid = np.empty((na, nb, nc), dtype=np.int64)
        # corner points keep the grid node numbering (F-order formula used
        # by build_mesh_data.node_index)
        for i in range(nx1):
            for j in range(ny1):
                for k in range(len(zl)):
                    station_grid[r * i, r * j, r * k] = (
                        i + nx1 * (j + ny1 * k)
                    )
        # remaining points, in a deterministic order
        nxt = n_grid
        for c in range(nc):
            for b in range(nb):
                for a in range(na):
                    if a % r == 0 and b % r == 0 and c % r == 0:
                        continue
                    station_grid[a, b, c] = nxt
                    nxt += 1

        xyz = np.empty((nxt, 3), dtype=np.float64)
        xyz[:n_grid] = mesh_data.x
        station_of_id = np.empty((nxt, 3), dtype=np.int64)
        for c in range(nc):
            for b in range(nb):
                for a in range(na):
                    pid = station_grid[a, b, c]
                    xyz[pid] = (xs[a], ys[b], zs[c])
                    station_of_id[pid] = (a, b, c)

        return cls(xyz=xyz, n_grid=n_grid, xs=xs, ys=ys, zs=zs,
                   station_grid=station_grid, station_of_id=station_of_id,
                   refinement=r)

    def local_coords(self, point_ids: np.ndarray) -> np.ndarray:
        """Local (xi, eta, zeta) of points within any incident cell: -1 at
        grid lines (even stations), 0 at mid-side stations. Valid for every
        incident cell because the grid is structured and conforming."""
        st = self.station_of_id[point_ids]
        return np.where(st % self.refinement == 0, -1.0, 0.0)


def surface_patch_fe(mesh_data: "MeshData", fe: FEPoints,
                     facet_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """FE points lying on a tagged surface patch, with tributary areas.

    The patch is the union of the tagged facets; on the (planar, axis-
    aligned) surface the FE points form a structured sub-grid, so each
    facet splits into ``refinement^2`` sub-quads and every FE point receives
    a quarter of each incident sub-quad area. Total area is conserved
    exactly, and for order 1 this reproduces the classic per-facet
    quarter-area rule.
    """
    sub = fe.refinement
    facet_ids = np.where(facet_mask)[0]
    if facet_ids.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    fac = mesh_data.facet_conn[facet_ids]

    z_plane = mesh_data.x[fac[0, 0], 2]
    c0 = int(np.argmin(np.abs(fe.zs - z_plane)))
    if abs(fe.zs[c0] - z_plane) > 1e-9 * max(1.0, abs(z_plane)):
        raise RuntimeError("surface patch is not on a mesh station plane")

    xl = np.unique(mesh_data.x[:, 0])
    yl = np.unique(mesh_data.x[:, 1])

    areas = np.zeros(fe.xyz.shape[0])
    seen = set()
    for f in fac:
        x0, y0 = mesh_data.x[f[0], 0], mesh_data.x[f[0], 1]
        ix = int(np.argmin(np.abs(xl - x0)))
        iy = int(np.argmin(np.abs(yl - y0)))
        if (ix, iy) in seen:
            continue
        seen.add((ix, iy))
        for sa in range(sub):
            for sb in range(sub):
                a, b = sub * ix + sa, sub * iy + sb
                corners = (
                    fe.station_grid[a, b, c0],
                    fe.station_grid[a + 1, b, c0],
                    fe.station_grid[a + 1, b + 1, c0],
                    fe.station_grid[a, b + 1, c0],
                )
                q = 0.25 * (fe.xs[a + 1] - fe.xs[a]) * (fe.ys[b + 1] - fe.ys[b])
                areas[list(corners)] += q

    sel = areas > 0.0
    return np.where(sel)[0], areas[sel]


@dataclass
class MeshData:
    """Raw structured mesh data before handing over to dolfinx."""

    x: np.ndarray                     # (nnodes, 3) node coordinates
    cells: np.ndarray                 # (ncells, nnodes_per_cell) connectivity
    cell_tags: np.ndarray             # (ncells,) layer tags
    facet_tags: np.ndarray            # (nfacets,) surface tags (1 per facet)
    facet_conn: np.ndarray            # (nfacets, 4) facet -> cell corners
    cell_conn: np.ndarray             # (ncells, 8) cell -> corner nodes
    is_half: bool

    # ---- geometry helpers ---------------------------------------------------
    @property
    def x_span(self) -> Tuple[float, float]:
        return float(self.x[:, 0].min()), float(self.x[:, 0].max())

    @property
    def z_span(self) -> Tuple[float, float]:
        return float(self.x[:, 2].min()), float(self.x[:, 2].max())

    def bottom_nodes(self) -> np.ndarray:
        return np.where(np.isclose(self.x[:, 2], self.z_span[0]))[0]

    def top_nodes(self) -> np.ndarray:
        return np.where(np.isclose(self.x[:, 2], self.z_span[1]))[0]


def build_mesh_data(cfg: Config) -> MeshData:
    """Build the structured mesh arrays from the configuration."""
    geom = cfg.geometry
    roles = cfg.resolve_roles()

    if cfg.half_model:
        # half of the physical beam: symmetry plane at x = 0 (load
        # roller), support at x = span/2, overhang kept out to x = length/2
        x_min, x_max = 0.0, geom.length / 2.0
    else:
        x_min, x_max = -geom.length / 2.0, geom.length / 2.0
    y_min, y_max = -geom.width / 2.0, geom.width / 2.0

    nx = cfg.mesh.elements_x
    nw = cfg.mesh.elements_w

    # -- node grid: independent z-lines per layer, shared interface nodes ----
    x_lines = np.linspace(x_min, x_max, nx + 1)
    y_lines = np.linspace(y_min, y_max, nw + 1)

    # Roller crowns must coincide with mesh nodes: the nodal-contact
    # formulation collocates the constraint at nodes, so a crown falling
    # between nodes would let the beam bridge over the roller. Insert the
    # crown stations into the x-line list if they are not already nodes.
    dx_total = x_max - x_min
    crown_tol = 1e-9 * dx_total
    if cfg.half_model:
        crown_stations = [0.0, geom.span / 2.0]
    else:
        crown_stations = [0.0, -geom.span / 2.0, geom.span / 2.0]
    # (half models: load crown at the symmetry plane, support crown at
    # span/2 - both are interior stations of the [0, length/2] domain)
    for station in crown_stations:
        if not np.any(np.abs(x_lines - station) <= crown_tol):
            x_lines = np.sort(np.append(x_lines, station))
    # element counts follow the (possibly refined) line lists
    nx = len(x_lines) - 1

    z_bounds = cfg.layer_z_bounds
    z_lines: List[np.ndarray] = []
    for i, role in enumerate(roles):
        n_elems = max(1, int(cfg.mesh.elements_per_layer.get(role, 1)))
        z_lines.append(np.linspace(z_bounds[i], z_bounds[i + 1], n_elems + 1))

    z_unique: List[float] = [z_lines[0][0]]
    for line in z_lines:
        z_unique.extend(line[1:].tolist())

    xs, ys, zs = np.meshgrid(x_lines, y_lines, np.array(z_unique), indexing="ij")
    xs, ys, zs = xs.ravel(order="F"), ys.ravel(order="F"), zs.ravel(order="F")
    coords = np.stack([xs, ys, zs], axis=1)
    n_nodes = coords.shape[0]

    def node_index(ix: int, iy: int, iz: int) -> int:
        # meshgrid with indexing='ij': shape (nx+1, ny+1, nz+1), ravel order 'F'
        return int(ix + (len(x_lines)) * (iy + (len(y_lines)) * iz))

    nzc = len(z_unique) - 1
    # -- cells ----------------------------------------------------------------
    cells: List[np.ndarray] = []
    cell_tags: List[int] = []

    z_offset = 0
    for i, role in enumerate(roles):
        n_elems = max(1, int(cfg.mesh.elements_per_layer.get(role, 1)))
        tag = i + 1  # 1-based stackup layer index
        for iz in range(n_elems):
            gz = z_offset + iz
            for ix in range(nx):
                for iy in range(nw):
                    # NOTE: basix hexahedron vertices follow the lexicographic
                    # convention - bottom quad (-,-), (+,-), (-,+), (+,+), then
                    # the top quad in the same order. Emitting a ring order
                    # would twist every cell into a degenerate "butterfly" hex.
                    n = [
                        node_index(ix, iy, gz),
                        node_index(ix + 1, iy, gz),
                        node_index(ix, iy + 1, gz),
                        node_index(ix + 1, iy + 1, gz),
                        node_index(ix, iy, gz + 1),
                        node_index(ix + 1, iy, gz + 1),
                        node_index(ix, iy + 1, gz + 1),
                        node_index(ix + 1, iy + 1, gz + 1),
                    ]
                    cells.append(np.array(n, dtype=np.int64))
                    cell_tags.append(tag)
        z_offset += n_elems

    # -- facets ---------------------------------------------------------------
    facet_tags: List[int] = []
    facet_conn: List[np.ndarray] = []

    def add_facet(node_ids: List[int], tag: int) -> None:
        facet_conn.append(np.array(node_ids, dtype=np.int64))
        facet_tags.append(tag)

    for ix in range(nx):
        for iy in range(nw):
            n = [
                node_index(ix, iy, 0),
                node_index(ix + 1, iy, 0),
                node_index(ix + 1, iy + 1, 0),
                node_index(ix, iy + 1, 0),
            ]
            # Tag contact facets by their corner nodes: a facet is a support
            # facet if any corner lies within the support roller radius of
            # the support axis. Node-based tagging guarantees the roller-
            # crown nodes always belong to tagged facets even on coarse
            # meshes (the contact pairs are then trimmed to nodes actually
            # inside the footprint).
            x_corners = x_lines[ix : ix + 2]
            if cfg.half_model:
                # half models: the physical support is at x = span/2
                Rs = cfg.contact.roller_radius_support
                in_support = bool(
                    np.any(np.abs(x_corners - geom.span / 2.0) <= Rs)
                )
                tag = FACET_TAG_SUPPORT if in_support else FACET_TAG_BOTTOM
            else:
                half_span = geom.span / 2.0
                Rs = cfg.contact.roller_radius_support
                in_support = bool(
                    np.any(np.abs(np.abs(x_corners) - half_span) <= Rs)
                )
                tag = FACET_TAG_SUPPORT if in_support else FACET_TAG_BOTTOM
            add_facet(n, tag)

    top_z_index = nzc
    for ix in range(nx):
        for iy in range(nw):
            n = [
                node_index(ix, iy, top_z_index),
                node_index(ix + 1, iy, top_z_index),
                node_index(ix + 1, iy + 1, top_z_index),
                node_index(ix, iy + 1, top_z_index),
            ]
            if cfg.half_model:
                # load roller at the symmetry plane x = 0 (node-based
                # tagging like the full model, mirrored footprint)
                x_face = x_lines[ix : ix + 2]
                in_load = np.any(np.abs(x_face) <= cfg.contact.roller_radius_load)
                tag = FACET_TAG_LOAD if in_load else FACET_TAG_TOP
            else:
                x_face = x_lines[ix : ix + 2]
                in_load = np.any(np.abs(x_face) <= cfg.contact.roller_radius_load)
                tag = FACET_TAG_LOAD if in_load else FACET_TAG_TOP
            add_facet(n, tag)

    if cfg.half_model:
        for iy in range(nw):
            for iz in range(nzc):
                n = [
                    node_index(0, iy, iz),
                    node_index(0, iy + 1, iz),
                    node_index(0, iy + 1, iz + 1),
                    node_index(0, iy, iz + 1),
                ]
                add_facet(n, FACET_TAG_SYMMETRY)

    return MeshData(
        x=coords,
        cells=np.array(cells, dtype=np.int64),
        cell_tags=np.array(cell_tags, dtype=np.int32),
        facet_tags=np.array(facet_tags, dtype=np.int32),
        facet_conn=np.array(facet_conn, dtype=np.int64),
        cell_conn=np.array(cells, dtype=np.int64),
        is_half=cfg.half_model,
    )


def build_dolfinx_mesh(mesh_data: MeshData):
    """Create a dolfinx Mesh with cell and facet tags from MeshData.

    The geometric mesh is always the linear hex8 grid; ``element_order``
    only selects the displacement interpolation (see ``FEPoints``), so
    order-2 runs use a superparametric formulation on this same mesh.

    Returns (mesh, cell_tags, facet_tags, cell_input_ids) where
    ``cell_input_ids`` maps each dolfinx cell to its row in
    ``mesh_data.cells`` (dolfinx may permute cells during construction).
    Serial (MPI.COMM_SELF) only.
    """
    from mpi4py import MPI
    import basix
    import basix.ufl
    import dolfinx
    import dolfinx.mesh as dmesh
    import ufl

    coord_element = basix.ufl.element(
        "Lagrange", basix.CellType.hexahedron, 1, shape=(3,)
    )
    domain = ufl.Mesh(coord_element)
    mesh = dmesh.create_mesh(
        MPI.COMM_SELF,
        np.ascontiguousarray(mesh_data.cells, dtype=np.int64),
        domain,
        np.ascontiguousarray(mesh_data.x, dtype=np.float64),
    )

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    num_facets_local = mesh.topology.index_map(fdim).size_local
    num_cells_local_total = mesh.topology.index_map(tdim).size_local

    # Build facet tag array aligned with dolfinx facet ordering by matching
    # vertex sets (stable topology API, no version-dependent helpers).
    mesh.topology.create_connectivity(fdim, 0)
    facet_vertex_conn = mesh.topology.connectivity(fdim, 0)

    def key(nodes) -> Tuple[int, ...]:
        return tuple(sorted(int(v) for v in nodes))

    tag_by_key: Dict[Tuple[int, ...], int] = {}
    for nodes, tag in zip(mesh_data.facet_conn, mesh_data.facet_tags):
        tag_by_key[key(nodes)] = int(tag)

    facet_tag_values = np.zeros(num_facets_local, dtype=np.int32)
    for f in range(num_facets_local):
        k = key(facet_vertex_conn.links(f))
        facet_tag_values[f] = tag_by_key.get(k, 0)

    # Cell tags: dolfinx may permute cells, so match each dolfinx cell to
    # the input cell by its (sorted) node set via the input global indices.
    igi = np.asarray(mesh.geometry.input_global_indices)
    mesh.topology.create_connectivity(tdim, 0)
    cell_vertex_conn = mesh.topology.connectivity(tdim, 0)
    tag_by_node_set: Dict[Tuple[int, ...], int] = {}
    input_id_by_node_set: Dict[Tuple[int, ...], int] = {}
    for row, (nodes, tag) in enumerate(
            zip(mesh_data.cells, mesh_data.cell_tags)):
        k = key(nodes)
        tag_by_node_set[k] = int(tag)
        input_id_by_node_set[k] = row
    cell_tag_values = np.zeros(num_cells_local_total, dtype=np.int32)
    cell_input_ids = np.full(num_cells_local_total, -1, dtype=np.int64)
    for c in range(num_cells_local_total):
        nodes = tuple(sorted(int(igi[v]) for v in cell_vertex_conn.links(c)))
        cell_tag_values[c] = tag_by_node_set.get(nodes, 0)
        cell_input_ids[c] = input_id_by_node_set.get(nodes, -1)
    if (cell_input_ids < 0).any():
        raise RuntimeError("some dolfinx cells could not be matched to "
                           "the input mesh data")

    cell_tags = dmesh.meshtags(
        mesh, tdim, np.arange(num_cells_local_total, dtype=np.int32),
        cell_tag_values,
    )
    facet_tags = dmesh.meshtags(
        mesh, fdim, np.arange(num_facets_local, dtype=np.int32),
        facet_tag_values,
    )
    return mesh, cell_tags, facet_tags, cell_input_ids


def roller_axes(cfg: Config) -> Dict[str, List[RollerAxis]]:
    """Rigid roller geometry for the configured beam.

    Half models cover x in [0, length/2]: x = 0 is the symmetry plane at
    mid-span (load roller), x = span/2 the physical support, and the
    overhang out to length/2 is kept. The reported load-roller force of a
    half model is doubled in the solver so that all reported forces refer
    to the full (symmetric) beam.
    """
    geom = cfg.geometry
    x_load = 0.0   # full: mid-span; half: symmetry plane (same place)
    top_gap = cfg.total_thickness
    rollers: Dict[str, List[RollerAxis]] = {"load": [], "support": []}

    rollers["load"].append(
        RollerAxis(
            center_x=x_load,
            center_y=0.0,
            radius=cfg.contact.roller_radius_load,
            initial_gap=top_gap,
            side="top",
        )
    )

    if cfg.half_model:
        support_positions = [geom.span / 2.0]
    else:
        half_span = geom.span / 2.0
        support_positions = [-half_span, half_span]

    for x_s in support_positions:
        rollers["support"].append(
            RollerAxis(
                center_x=x_s,
                center_y=0.0,
                radius=cfg.contact.roller_radius_support,
                initial_gap=0.0,
                side="bottom",
            )
        )
    return rollers


def locate_entities(mesh, cell_tags, facet_tags):
    """Convenience accessor returning (tdim, fdim, mesh) for later use."""
    return mesh.topology.dim, mesh.topology.dim - 1, mesh
