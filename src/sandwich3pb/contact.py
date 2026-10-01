"""Rigid-roller frictionless contact, collocation at the FE points.

Each roller is a rigid cylinder with its axis along the width (y) direction.
Contact is one-way rigid/deformable and frictionless. The constraint is
enforced at the contact-surface FE points (grid nodes, plus mid-side
points for quadratic displacement spaces):

    g_a(u) >= 0,  p_a >= 0,  p_a * g_a = 0            (per point a)

with the signed gap g >= 0 when open. The nodal pressure is

    p_a = lambda_a + k * <-g_a(u)>                    (penalty / AL)

with k the penalty stiffness (N/mm^3) and lambda_a >= 0 an augmented-
Lagrangian multiplier updated between outer iterations.

Gap definitions (R = roller radius, dx = horizontal offset of the node
from the roller axis; only nodes with |dx| <= R take part in contact)
---------------------------------------------------------------------
Top roller (side = "top"), roller centre travel d (downwards, >= 0):
    roller surface height at offset dx:  t + R - d - sqrt(R^2 - dx^2)
    g_a = R - d - sqrt(R^2 - dx^2) - u_z_a
so the roller first touches the undeformed beam at midspan (dx = 0).
g < 0 means penetration depth -g.

Support rollers (side = "bottom"), fixed:
    roller surface height at offset dx:  -R + sqrt(R^2 - dx^2)
    g_a = u_z_a + R - sqrt(R^2 - dx^2)
(the beam bottom may not penetrate the roller from above).

Because contact points coincide with FE interpolation points, the nodal
forces enter the global residual directly at the z-dofs and the contact
tangent (derivative of the penalty force w.r.t. the point displacement) is
diagonal for the linear (hex8) space:
    dF_a/du_a = k * A_a   for active points, 0 otherwise.
For the quadratic space the mid-side point gaps additionally depend on the
element corner-point displacements (through the trilinear z(x,y,z) map,
see solver._newton for the tangent correction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .geometry import FEPoints, MeshData, RollerAxis, surface_patch_fe

LOAD = "load"
SUPPORT = "support"


@dataclass
class ContactPair:
    """One rigid roller acting on a set of contact-surface FE points."""

    roller: RollerAxis
    point_ids: np.ndarray           # global FE point ids on the surface
    point_areas: np.ndarray         # tributary area of each point (mm^2)
    point_dx: np.ndarray            # x-offset of each point from the axis
    side: str                       # "top" | "bottom"
    travel: float = 0.0             # roller centre travel (mm, >= 0)
    lambda_mult: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if self.lambda_mult.size != self.point_ids.size:
            self.lambda_mult = np.zeros(self.point_ids.size)

    # -- geometry -------------------------------------------------------------
    @property
    def normal_sign(self) -> float:
        """z-sign of the force the roller exerts on the beam."""
        return -1.0 if self.side == "top" else +1.0

    def cylinder_drop(self) -> np.ndarray:
        """sqrt(R^2 - dx^2): how far the cylinder surface is below its crown."""
        return np.sqrt(np.maximum(self.roller.radius**2 - self.point_dx**2, 0.0))

    def gaps(self, u_z_points: np.ndarray) -> np.ndarray:
        """Signed gap (mm) at each contact point for current displacements."""
        u = u_z_points[self.point_ids]
        cyl = self.cylinder_drop()
        if self.side == "top":
            return self.roller.radius - self.travel - cyl - u
        return self.roller.radius + u - cyl

    # -- pressures and forces --------------------------------------------------
    def pressures(self, gaps: np.ndarray, penalty: float) -> np.ndarray:
        """Nodal contact pressure (MPa, >= 0)."""
        return self.lambda_mult + penalty * np.maximum(0.0, -gaps)

    def nodal_forces_z(self, gaps: np.ndarray, penalty: float) -> np.ndarray:
        """Nodal z-forces (N) on the beam, signed (+up)."""
        p = self.pressures(gaps, penalty)
        return self.normal_sign * p * self.point_areas

    def resultant_force(self, gaps: np.ndarray, penalty: float) -> float:
        """Magnitude of the total contact force transmitted (N, >= 0)."""
        p = self.pressures(gaps, penalty)
        return float(np.dot(p, self.point_areas))

    def update_multipliers(self, gaps: np.ndarray, penalty: float,
                           rate: float) -> float:
        """AL multiplier update. Returns the max penetration (mm)."""
        violation = np.maximum(0.0, -gaps)
        self.lambda_mult = np.maximum(
            0.0, self.lambda_mult + rate * penalty * violation
        )
        return float(violation.max()) if violation.size else 0.0


@dataclass
class ContactSystem:
    """All contact pairs of the model."""

    pairs: Dict[str, List[ContactPair]]
    penalty: float

    def gaps_all(self, u_z_nodes: np.ndarray) -> Dict[str, List[np.ndarray]]:
        return {
            name: [pair.gaps(u_z_nodes) for pair in self.pairs[name]]
            for name in (LOAD, SUPPORT)
        }

    def resultant_forces(self, gaps: Dict[str, List[np.ndarray]]
                         ) -> Dict[str, List[float]]:
        return {
            name: [pair.resultant_force(g, self.penalty)
                   for pair, g in zip(self.pairs[name], gaps[name])]
            for name in (LOAD, SUPPORT)
        }

    def max_violation(self, gaps: Dict[str, List[np.ndarray]]) -> float:
        worst = 0.0
        for name in (LOAD, SUPPORT):
            for g in gaps[name]:
                if g.size:
                    worst = max(worst, float(np.maximum(0.0, -g).max()))
        return worst

    def all_nodal_forces(self, u_z_nodes: np.ndarray
                         ) -> tuple[Dict[str, List[np.ndarray]],
                                    Dict[str, np.ndarray]]:
        """Gaps and signed nodal z-forces per pair (for the residual)."""
        gaps = self.gaps_all(u_z_nodes)
        forces = {
            name: [pair.nodal_forces_z(g, self.penalty)
                   for pair, g in zip(self.pairs[name], gaps[name])]
            for name in (LOAD, SUPPORT)
        }
        return gaps, forces


def build_contact_system(mesh_data: MeshData, rollers: Dict[str, List[RollerAxis]],
                         load_tag: int, support_tag: int,
                         penalty: float,
                         fe_points: Optional[FEPoints] = None
                         ) -> ContactSystem:
    """Create the contact system from raw mesh data.

    With ``fe_points`` (order 2) the contact points are the FE points of the
    tagged surface patches - including mid-side points, which carry the
    z-mapping term -w * w'' of the trilinear through-thickness map.
    """
    pairs: Dict[str, List[ContactPair]] = {LOAD: [], SUPPORT: []}

    for name, tag in ((LOAD, load_tag), (SUPPORT, support_tag)):
        for roller in rollers[name]:
            mask = mesh_data.facet_tags == tag
            if fe_points is not None:
                patch_ids, patch_areas = surface_patch_fe(mesh_data,
                                                          fe_points, mask)
                dx_all = fe_points.xyz[patch_ids, 0] - roller.center_x
                keep = np.abs(dx_all) <= roller.radius
                point_ids = np.array(patch_ids[keep], dtype=int)
                areas = patch_areas[keep]
            else:
                candidates = np.unique(mesh_data.facet_conn[mask].ravel())
                # only nodes inside the cylinder's horizontal extent contact
                dx = mesh_data.x[candidates, 0] - roller.center_x
                point_ids = np.array(
                    sorted(int(n) for n, d in zip(candidates, dx)
                           if abs(d) <= roller.radius),
                    dtype=int,
                )
                areas = _tributary_areas(mesh_data, point_ids, mask)
            if point_ids.size == 0:
                raise ValueError(
                    f"roller '{name}' at x={roller.center_x} touches no mesh "
                    "nodes; increase its radius or refine the mesh"
                )
            coords = fe_points.xyz if fe_points is not None else mesh_data.x
            point_dx = coords[point_ids, 0] - roller.center_x

            # NOTE: use the roller's own side ("top"/"bottom"), not the
            # pair-group name ("load"/"support").
            pairs[name].append(
                ContactPair(roller=roller, point_ids=point_ids,
                            point_areas=areas, point_dx=point_dx,
                            side=roller.side)
            )

    return ContactSystem(pairs=pairs, penalty=penalty)


def _tributary_areas(mesh_data: MeshData, node_ids: np.ndarray,
                     facet_mask: np.ndarray) -> np.ndarray:
    """Tributary area (mm^2) of each node over the tagged facet patch.

    Quarter-area allocation from each incident quad: exact total-area
    conservation for any quad layout, and exact nodal areas on uniform grids.
    """
    areas = np.zeros(node_ids.size)
    index = {int(n): i for i, n in enumerate(node_ids)}
    for fac in mesh_data.facet_conn[facet_mask]:
        a = mesh_data.x[fac]
        # planar quad area via two triangles (nodes in ring order)
        t1 = 0.5 * np.linalg.norm(np.cross(a[1] - a[0], a[2] - a[0]))
        t2 = 0.5 * np.linalg.norm(np.cross(a[2] - a[0], a[3] - a[0]))
        quarter = 0.25 * (t1 + t2)
        for n in fac:
            i = index.get(int(n))
            if i is not None:  # nodes outside the roller footprint carry no area
                areas[i] += quarter
    return areas
