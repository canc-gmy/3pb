"""Displacement-controlled three-point-bending solver (FEniCSx + SciPy).

Nonlinearity comes from contact only (materials are linear elastic). At each
imposed roller travel d we solve

    r(u) = K u - F_contact(u, d) = 0        (Newton)

with
  * K the global linear stiffness assembled once from per-layer 6x6
    constitutive matrices (Voigt form, so rotated orthotropic plies are
    handled exactly), and
  * F_contact the nodal forces of the rigid-roller penalty/AL contact model.

The Newton tangent is K plus a *diagonal* contact contribution k*A_a on the
active contact nodes (the node-based contact formulation makes this exact).

Outer loop per step (augmented Lagrangian): after Newton converges, gaps are
evaluated, multipliers are updated, and the solve repeats until the maximum
penetration is below tolerance (a single pass when AL is disabled).

The displacement space is quadratic Lagrange (Q2, hex27-equivalent dofs)
when ``mesh.element_order == 2``; the geometric mesh always stays linear
hex8, so the formulation is *superparametric*. This removes the transverse-
shear locking of fully-linear hex8 beams. Contact and boundary conditions
are handled on the FE point set (grid nodes + mid-side points for Q2).
Because the geometry map is multilinear, a contact point's deformed
position depends only on its own u_z, so the contact formulation (gaps,
forces, diagonal tangent) is identical for both element orders.

Boundary conditions are applied by dof selection on the SciPy matrix:
  * full models: uy = 0 at both end cross-sections only. No ux constraint:
    frictionless rollers exert no axial force, so ux is a zero-energy rigid
    mode, and constraining a whole end section would clamp the beam slope
    (ux ~ -z w' in beam kinematics), bracing the overhangs and stiffening
    the span;
  * half models: ux = 0 on the symmetry plane x = 0, uy = 0 elsewhere.

The remaining rigid modes (x translation, and z before the supports
engage) are stabilized with weak ground springs (stiffness 1e-6 x the
mean diagonal of the free stiffness block) so the Newton tangent is
nonsingular at every iteration. The springs are ~6 orders of magnitude
below the contact stiffness and do not measurably affect results; they
select the symmetric branch of the (otherwise indeterminate) rigid
modes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple  # noqa: F401

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from .config import Config
from .contact import LOAD, SUPPORT, ContactSystem, build_contact_system
from .geometry import (
    FACET_TAG_LOAD,
    FACET_TAG_SUPPORT,
    FEPoints,
    MeshData,
    build_dolfinx_mesh,
    build_mesh_data,
    roller_axes,
    surface_patch_fe,
)
from .materials import LocalizedQuad, localized_quad_form


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One converged load step."""

    step: int
    travel: float                 # imposed roller travel (mm)
    force_load: float             # total force of the load roller (N)
    support_forces: List[float]   # resultant force per support roller (N)
    max_deflection: float         # |u_z| at mid-span top node (mm)
    max_violation: float          # max contact penetration (mm)
    newton_iterations: int
    al_iterations: int
    converged: bool


@dataclass
class Solution:
    """Everything downstream modules need."""

    cfg: Config
    mesh_data: MeshData
    u_dofs: np.ndarray                    # (n_dofs,)
    node2dof: np.ndarray                  # (nnodes, 3) node/component -> dof
    contact: ContactSystem
    fe_points: Optional[FEPoints] = None
    element_order: int = 1
    history: List[StepRecord] = field(default_factory=list)
    assembly_time: float = 0.0
    solve_time: float = 0.0
    n_dofs: int = 0
    n_nodes: int = 0
    cell_layer_tags: Optional[np.ndarray] = None  # (ncells,) stackup layer idx
    solver_ref: Optional["ThreePointBendingSolver"] = field(
        default=None, repr=False
    )

    def u_z_nodes(self) -> np.ndarray:
        """u_z at the *grid nodes* (layer-sharing points; stress output)."""
        return self.u_dofs[self.node2dof[:, 2]]

    def u_z_fe(self) -> Optional[np.ndarray]:
        """u_z at every FE point (grid + mid-side for quadratic elements)."""
        ref = self.solver_ref
        if ref is None or ref.point_map is None:
            return None
        return self.u_dofs[ref.point_map[:, 2]]

    def last_record(self) -> Optional[StepRecord]:
        return self.history[-1] if self.history else None

    @property
    def converged(self) -> bool:
        return bool(self.history) and self.history[-1].converged


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


class ThreePointBendingSolver:
    """Builds the model once, then runs the load history on demand."""

    def __init__(self, cfg: Config, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose

        self.mesh_data = build_mesh_data(cfg)
        self.rollers = roller_axes(cfg)
        # FE points are needed before the contact system for order 2 (the
        # contact points are the FE points of the surface patches).
        self.fe_points = FEPoints.build(self.mesh_data, cfg.mesh.element_order)
        self.element_order = int(cfg.mesh.element_order)
        self.contact = build_contact_system(
            self.mesh_data,
            self.rollers,
            load_tag=FACET_TAG_LOAD,
            support_tag=FACET_TAG_SUPPORT,
            penalty=cfg.contact.penalty,
            fe_points=self.fe_points,
        )
        self.point_map: Optional[np.ndarray] = None

        t0 = time.perf_counter()
        self._build_fe_model()
        self.assembly_time = time.perf_counter() - t0
        if self.verbose:
            print(
                f"[sandwich3pb] model built: {self.n_nodes} nodes, "
                f"{self.n_dofs} dofs, assembly {self.assembly_time:.1f} s"
            )

    # -- model construction ---------------------------------------------------
    def _build_fe_model(self) -> None:
        import dolfinx.fem as fem
        import ufl

        cfg = self.cfg
        mesh, cell_tags, facet_tags, cell_input_ids = \
            build_dolfinx_mesh(self.mesh_data)
        self.cell_input_ids = cell_input_ids
        self.mesh, self.cell_tags, self.facet_tags = mesh, cell_tags, facet_tags

        # ---- function space and FE point -> dof map -------------------------
        import basix
        import basix.ufl

        order = int(self.element_order)
        vector_element = basix.ufl.element(
            "Lagrange", basix.CellType.hexahedron, order, shape=(3,)
        )
        self.V = fem.functionspace(mesh, vector_element)
        bs = self.V.dofmap.index_map_bs
        if bs != 3:
            raise RuntimeError(f"unexpected vector block size {bs}")

        # tabulate_dof_coordinates: one (x, y, z) row per *block* (bs = 3),
        # indexed by the same block numbering used in cell_dofs. dolfinx
        # reorders dofs w.r.t. basix and permutes geometry points, so the
        # robust point -> dof map is pure coordinate matching (corner
        # coordinates reproduce the input nodes exactly; mid-side stations
        # are exact binary averages, so a rounding to 1e-9 is safe).
        n_grid = self.mesh_data.x.shape[0]
        fe = self.fe_points
        n_fe = fe.xyz.shape[0]
        block_coords = np.asarray(self.V.tabulate_dof_coordinates())
        n_blocks = block_coords.shape[0]
        if n_blocks * bs != int(self.V.dofmap.index_map.size_local * bs):
            raise RuntimeError("dof coordinate tabulation is not per block")
        block_of_coord: Dict[Tuple[float, float, float], int] = {}
        for block in range(n_blocks):
            c = tuple(np.round(block_coords[block], 9))
            if c in block_of_coord:
                raise RuntimeError("duplicated dof coordinates")
            block_of_coord[c] = block
        if len(block_of_coord) != n_fe:
            raise RuntimeError(
                f"{n_blocks} dof blocks but {n_fe} FE points - element "
                "order and mesh do not agree"
            )
        point_map = np.full((n_fe, 3), -1, dtype=np.int64)
        for pid in range(n_fe):
            c = tuple(np.round(fe.xyz[pid], 9))
            block = block_of_coord.get(c)
            if block is None:
                raise RuntimeError(
                    f"FE point {pid} at {fe.xyz[pid]} has no matching dof"
                )
            point_map[pid, 0] = block * bs
            point_map[pid, 1] = block * bs + 1
            point_map[pid, 2] = block * bs + 2
        self.node2dof = point_map[:n_grid]
        self.point_map = point_map
        self.n_nodes = int(n_grid)
        self.n_dofs = int(self.V.dofmap.index_map.size_local * bs)

        # ---- per-layer constitutive quads ------------------------------------
        self.quads: List[LocalizedQuad] = []
        for layer in cfg.stackup:
            mat = cfg.materials[layer.material]
            self.quads.append(localized_quad_form(mat, layer.fibre_orientation))

        # ---- bilinear form, layer by layer -----------------------------------
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        def voigt_strain(t):
            return ufl.as_vector(
                [t[0, 0], t[1, 1], t[2, 2], 2 * t[1, 2], 2 * t[0, 2], 2 * t[0, 1]]
            )

        def voigt_stress(C, t):
            s = C * voigt_strain(t)
            return ufl.as_tensor(
                [[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]]
            )

        def eps(w):
            return ufl.sym(ufl.grad(w))

        dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
        a = 0
        for i, quad in enumerate(self.quads):
            C = ufl.as_matrix(quad.C.tolist())
            a = a + ufl.inner(voigt_stress(C, eps(u)), eps(v)) * dx(i + 1)

        # ---- assemble K once (no BC objects; BCs by dof selection) -----------
        K = _assemble_bilinear(fem.form(a))
        self.K = K.tocsr()
        self.free_dofs, self.fixed_dofs = self._boundary_dofs()
        K_ff = self.K[self.free_dofs][:, self.free_dofs].tocsc()

        # Ground springs on the rigid modes that contact does not
        # restrain (x translation; z before the supports engage). Without
        # them K_ff is singular: the first Newton iteration of a step
        # would be unconstrained in z, and the beam could slide in x
        # (asymmetric reactions). eps must be tiny: summed over all FE
        # points the springs still carry load, which would bleed force out
        # of the contact reactions (1e-6 keeps ~5% of P, 1e-9 < 0.01%).
        g2f = -np.ones(self.n_dofs, dtype=np.int64)
        g2f[self.free_dofs] = np.arange(self.free_dofs.size)
        eps = 1.0e-9 * float(np.abs(K_ff.diagonal()).mean())
        springs = np.zeros(self.free_dofs.size)
        for component in (0, 2):  # ux, uz
            rows = g2f[self.point_map[:, component]]
            valid = rows >= 0
            springs[rows[valid]] = eps
        self.K_ff = (K_ff + diags(springs)).tocsc()
        # The Newton tangent (K_ff + active contact stiffness) is factored
        # per iteration inside _newton.

    def grid_node_point(self, node_id: int) -> int:
        """FE point id of a grid node (order 1: identity)."""
        return int(node_id)

    def _boundary_dofs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Dirichlet selection on FE point dofs (see module docstring).

        Note the *left end only* ux constraint: ux at BOTH ends would clamp
        the beam slope (ux ~ -z w') and stiffen the span by roughly a
        factor two through rotational overhang springs.
        """
        cfg = self.cfg
        fe = self.fe_points
        x_all = fe.xyz
        tol = 1e-6 * max(1.0, cfg.geometry.length)
        free_mask = np.ones(self.n_dofs, dtype=bool)

        def fix(component: int, point_mask: np.ndarray) -> None:
            dofs = self.point_map[point_mask, component]
            free_mask[dofs] = False

        xmax = x_all[:, 0].max()
        xmin = self.mesh_data.x_span[0]
        if cfg.half_model:
            sym_points = np.isclose(x_all[:, 0], 0.0, atol=tol)
            fix(0, sym_points)          # ux = 0 on symmetry plane
            fix(1, ~sym_points)         # uy = 0 elsewhere
        else:
            end_points = np.isclose(x_all[:, 0], xmin, atol=tol) | np.isclose(
                x_all[:, 0], xmax, atol=tol
            )
            # no ux constraint (see module docstring)
            fix(1, end_points)          # uy = 0 at both ends
        # u_z is restrained by the support rollers through contact.
        return np.where(free_mask)[0], np.where(~free_mask)[0]

    # -- stepping --------------------------------------------------------------
    def solve(self) -> Solution:
        """Run the full load history and return a Solution."""
        cfg = self.cfg
        u = np.zeros(self.n_dofs)
        history: List[StepRecord] = []

        travels = np.linspace(
            0.0, cfg.loading.max_indentation, cfg.loading.n_steps + 1
        )[1:]
        t0 = time.perf_counter()
        for step, travel in enumerate(travels, start=1):
            record, u = self._solve_step(step, float(travel), u)
            history.append(record)
            if self.verbose:
                status = "ok " if record.converged else "NOT CONVERGED"
                print(
                    f"[sandwich3pb] step {step:3d}/{len(travels)} "
                    f"d={record.travel:7.3f} mm  P={record.force_load:10.1f} N  "
                    f"w={record.max_deflection:7.3f} mm  "
                    f"it={record.newton_iterations:2d}  "
                    f"AL={record.al_iterations}  viol={record.max_violation:.1e}"
                    f"  {status}"
                )
        solve_time = time.perf_counter() - t0

        cell_layer = np.asarray(self.mesh_data.cell_tags, dtype=int)  # 1-based
        return Solution(
            cfg=cfg,
            mesh_data=self.mesh_data,
            u_dofs=u,
            node2dof=self.node2dof,
            contact=self.contact,
            fe_points=self.fe_points,
            element_order=self.element_order,
            history=history,
            assembly_time=self.assembly_time,
            solve_time=solve_time,
            n_dofs=self.n_dofs,
            n_nodes=self.n_nodes,
            cell_layer_tags=cell_layer,
            solver_ref=self,
        )

    def _solve_step(self, step: int, travel: float,
                    u0: np.ndarray) -> Tuple[StepRecord, np.ndarray]:
        cfg = self.cfg
        al = cfg.contact.augmented_lagrangian
        u = u0.copy()

        for pair in self.contact.pairs[LOAD]:
            pair.travel = travel

        newton_iters = 0
        al_iters = 0
        converged = False
        gaps_final: Optional[Dict[str, List[np.ndarray]]] = None

        max_outer = al.max_outer if al.enabled else 1
        for _outer in range(max_outer):
            u, iters, ok = self._newton(u, travel)
            newton_iters += iters
            al_iters += 1

            u_z_points = u[self.point_map[:, 2]]
            gaps = self.contact.gaps_all(u_z_points)
            violation = self.contact.max_violation(gaps)
            gaps_final = gaps

            if not ok:
                break
            if not al.enabled or violation <= al.tol:
                converged = True
                break
            for name in (LOAD, SUPPORT):
                for pair, g in zip(self.contact.pairs[name], gaps[name]):
                    pair.update_multipliers(
                        g, self.contact.penalty, al.update_rate
                    )

        forces = self.contact.resultant_forces(gaps_final)
        # half models carry half the load roller (symmetry); scale so all
        # reported forces refer to the full beam
        scale = 2.0 if cfg.half_model else 1.0
        force_load = scale * float(sum(forces[LOAD]))
        support_forces = [scale * float(f) for f in forces[SUPPORT]]

        # mid-span is x = 0 in full AND half models (half models span
        # x in [0, span/2] with the symmetry plane at x = 0)
        mid_node = _nearest_node(self.mesh_data.x, 0.0, 0.0,
                                 cfg.total_thickness)
        w_mid = -float(u[self.node2dof[mid_node, 2]])

        record = StepRecord(
            step=step,
            travel=travel,
            force_load=force_load,
            support_forces=support_forces,
            max_deflection=w_mid,
            max_violation=self.contact.max_violation(gaps_final),
            newton_iterations=newton_iters,
            al_iterations=al_iters,
            converged=converged,
        )
        return record, u

    def _newton(self, u: np.ndarray, travel: float
                ) -> Tuple[np.ndarray, int, bool]:
        """Newton iterations with an active-set contact tangent.

        Contact is collocated at the FE points. Because the hex8 geometry
        map is multilinear (linear along every coordinate line), the
        deformed surface position of a contact point is exactly its own
        u_z - also for the quadratic displacement space, where mid-side
        FE points act as additional collocation points. The contact
        tangent is therefore diagonal (k * tributary area) for every
        element order.
        """
        cfg = self.cfg
        pm = self.point_map
        z_dofs = pm[:, 2]
        fe = self.fe_points
        g2f = -np.ones(self.n_dofs, dtype=np.int64)
        g2f[self.free_dofs] = np.arange(self.free_dofs.size)

        converged = False
        ref_force = 1.0
        iters = 0
        for it in range(cfg.solver.max_newton_iterations):
            u_z_points = u[z_dofs]
            gaps, forces_by_pair = self.contact.all_nodal_forces(u_z_points)

            forces_z = np.zeros(fe.xyz.shape[0])
            diag_contact = np.zeros(fe.xyz.shape[0])
            for name in (LOAD, SUPPORT):
                for pair, g, fz in zip(self.contact.pairs[name],
                                       gaps[name], forces_by_pair[name]):
                    forces_z[pair.point_ids] += fz
                    ref_force = max(ref_force, float(np.abs(fz).sum()))
                    # touching points (g <= 0) count as active: the support
                    # crowns start exactly at g = 0 and must restrain z from
                    # the first Newton iteration of the first step
                    active = g <= 0.0
                    diag_contact[pair.point_ids[active]] += (
                        self.contact.penalty * pair.point_areas[active]
                    )

            # (the deformed-surface map is multilinear, so no corner
            # coupling terms exist - see the docstring above)

            r_full = self.K @ u
            r_full[z_dofs] -= forces_z
            r_free = r_full[self.free_dofs]
            if np.max(np.abs(r_free)) <= cfg.solver.rtol * ref_force:
                converged = True
                iters = it
                break

            d_free = np.zeros(self.free_dofs.size)
            # map nodal contact stiffness onto the free z-dof diagonal
            z_free_positions = g2f[z_dofs]
            valid = z_free_positions >= 0
            np.add.at(
                d_free,
                z_free_positions[valid],
                diag_contact[np.where(valid)[0]],
            )
            T = (self.K_ff + diags(d_free)).tocsc()
            delta = _solve_tangent(T, -r_free)
            u[self.free_dofs] += delta
            iters = it + 1

        return u, iters, converged


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _solve_tangent(T, rhs: np.ndarray) -> np.ndarray:
    """Solve the Newton tangent system (nonsingular thanks to the ground
    springs on the rigid modes and the touching-inclusive contact
    activation)."""
    return splu(T).solve(rhs)


def _nearest_node(coords: np.ndarray, x: float, y: float, z: float) -> int:
    d = (coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2 \
        + (coords[:, 2] - z) ** 2
    return int(np.argmin(d))


def _assemble_bilinear(form):
    """Assemble a bilinear form to a SciPy CSR matrix (no BCs applied)."""
    import numpy as _np

    try:  # dolfinx <= 0.10
        from dolfinx.fem.petsc import assemble_matrix as _assemble
    except ImportError:  # newer dolfinx
        from dolfinx.fem import assemble_matrix as _assemble  # type: ignore

    A = _assemble(form)
    A.assemble()
    indptr, indices, data = A.getValuesCSR()
    size = A.getSize()[0]
    from scipy.sparse import csr_matrix

    M = csr_matrix((data, indices, indptr), shape=(size, size))
    A.destroy()
    return M
