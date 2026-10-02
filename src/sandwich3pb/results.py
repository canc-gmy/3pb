"""Case results container: engineering quantities extracted from a run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import Config
from .postprocess import (
    beam_axis_profile,
    failure_indices,
    layup_flexural_rigidity,
    midspan_profile,
    recover_nodal_stress,
)
from .solver import Solution


def _stiffness_gradient(force: np.ndarray, deflection: np.ndarray
                        ) -> tuple[float, float, int]:
    """dP/dw by iteratively refitted linear regression.

    Returns (gradient, r_squared, n_points_used). Points deviating more than
    2.5 sigma from the current fit are dropped (contact softening at the
    start of the curve and any nonlinear tail).
    """
    if force.size < 2:
        return float("nan"), float("nan"), int(force.size)
    x, y = deflection, force
    mask = np.ones_like(x, dtype=bool)
    for _ in range(5):
        xm, ym = x[mask], y[mask]
        if xm.size < 2:
            break
        A = np.vstack([xm, np.ones_like(xm)]).T
        (slope, intercept), res, *_ = np.linalg.lstsq(A, ym, rcond=None)
        pred = slope * x + intercept
        resid = y - pred
        sigma = resid[mask].std(ddof=1) if mask.sum() > 2 else 0.0
        if sigma == 0.0:
            break
        new_mask = np.abs(resid) <= 2.5 * sigma
        if new_mask.sum() < 2 or (new_mask == mask).all():
            break
        mask = new_mask
    xm, ym = x[mask], y[mask]
    A = np.vstack([xm, np.ones_like(xm)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, ym, rcond=None)
    pred = slope * xm + intercept
    ss_res = float(np.sum((ym - pred) ** 2))
    ss_tot = float(np.sum((ym - ym.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2), int(mask.sum())


@dataclass
class CaseResults:
    """Extracted engineering results of one three-point-bending run."""

    cfg: Config
    summary: Dict[str, Any]
    history: List[Dict[str, Any]]
    profile: Dict[str, np.ndarray]
    span_profile: Dict[str, object] = field(default_factory=dict)
    stress_nodal: Optional[np.ndarray] = None   # (n_nodes, n_layers, 6)
    stress_counts: Optional[np.ndarray] = None
    solution: Optional[Solution] = field(default=None, repr=False)

    # -- factory ---------------------------------------------------------------
    @classmethod
    def from_solution(cls, solution: Solution) -> "CaseResults":
        cfg = solution.cfg
        hist = solution.history

        force = np.array([r.force_load for r in hist])
        deflection = np.array([r.max_deflection for r in hist])
        travels = np.array([r.travel for r in hist])

        gradient, r2, n_fit = _stiffness_gradient(force, deflection)
        L = cfg.geometry.span
        D_app = gradient * L**3 / 48.0  # 3PB closed form
        layup = layup_flexural_rigidity(cfg)

        last = hist[-1]
        support_total = float(sum(last.support_forces))
        balance_residual = abs(support_total - last.force_load) / max(
            last.force_load, 1e-12
        )

        stress, counts = recover_nodal_stress(
            solution.mesh_data,
            solution.u_dofs,
            solution.node2dof,
            solution.cell_layer_tags,
            solution.solver_ref.quads,
        )
        failure = failure_indices(cfg, solution.mesh_data, stress, counts,
                                  quads=solution.solver_ref.quads)
        profile = midspan_profile(
            solution.mesh_data, stress, counts, cfg,
            u_z_fe=solution.u_z_fe(),
            fe_points=solution.fe_points,
        )
        span_profile = beam_axis_profile(stress, counts, solution.mesh_data, cfg)

        roles = cfg.resolve_roles()
        face_tw = [
            d["max_tsai_wu"] for d in failure.values() if d["role"] == "face"
        ]
        core_shear = [
            d["max_shear_ratio"] for d in failure.values()
            if d["role"] == "core"
        ]
        core_crush = [
            d["max_crushing_ratio"] for d in failure.values()
            if d["role"] == "core"
        ]

        summary: Dict[str, Any] = {
            "case_name": cfg.name,
            "max_force_N": float(force.max()) if force.size else 0.0,
            "deflection_at_max_force_mm": float(
                deflection[int(np.argmax(force))]
            )
            if force.size
            else 0.0,
            "max_deflection_mm": float(deflection.max()) if deflection.size else 0.0,
            "final_travel_mm": float(travels[-1]) if travels.size else 0.0,
            "force_gradient_N_per_mm": gradient,
            "gradient_fit_r2": r2,
            "gradient_fit_points": n_fit,
            "apparent_flexural_rigidity_Nmm2": D_app,
            "layup_flexural_rigidity_Nmm2": layup["flexural_rigidity"],
            "layup_flexural_rigidity_per_width_Nmm": layup[
                "flexural_rigidity_per_width"
            ],
            "neutral_axis_z_mm": layup["neutral_axis_z"],
            "rigidity_ratio_FE_over_layup": (
                D_app / layup["flexural_rigidity"]
                if layup["flexural_rigidity"] > 0
                else float("nan")
            ),
            "load_roller_force_N": last.force_load,
            "support_reactions_N": last.support_forces,
            "support_reaction_total_N": support_total,
            "force_balance_residual": balance_residual,
            "max_contact_penetration_mm": last.max_violation,
            "max_tsai_wu_faces": max(face_tw) if face_tw else None,
            "max_core_shear_ratio": max(core_shear) if core_shear else None,
            "max_core_crushing_ratio": max(core_crush) if core_crush else None,
            "n_steps": len(hist),
            "all_steps_converged": all(r.converged for r in hist),
            "n_nodes": solution.n_nodes,
            "n_dofs": solution.n_dofs,
            "element_order": int(getattr(solution, "element_order", 1)),
            "assembly_time_s": solution.assembly_time,
            "solve_time_s": solution.solve_time,
            "failure_by_layer": failure,
        }
        return cls(
            cfg=cfg,
            summary=summary,
            history=[
                {
                    "step": r.step,
                    "travel_mm": r.travel,
                    "force_N": r.force_load,
                    "deflection_mm": r.max_deflection,
                    "support_forces_N": r.support_forces,
                    "max_violation_mm": r.max_violation,
                    "newton_iterations": r.newton_iterations,
                    "al_iterations": r.al_iterations,
                    "converged": r.converged,
                }
                for r in hist
            ],
            profile=profile,
            span_profile=span_profile,
            stress_nodal=stress,
            stress_counts=counts,
            solution=solution,
        )

    # -- outputs ---------------------------------------------------------------
    def write_outputs(self, out_dir: str) -> Dict[str, str]:
        """Write CSV/JSON/XDMF/plots/report; returns the artifact paths."""
        from . import io as _io
        from .report import write_report

        paths: Dict[str, str] = {}
        paths.update(_io.write_case_outputs(self, out_dir))
        paths.update(write_report(self, out_dir))
        return paths
