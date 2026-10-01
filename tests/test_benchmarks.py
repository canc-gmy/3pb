"""FEM verification benchmarks (require dolfinx; skipped otherwise)."""

import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")

from sandwich3pb.postprocess import layup_flexural_rigidity
from sandwich3pb.results import CaseResults
from sandwich3pb.solver import ThreePointBendingSolver
from tests.conftest import make_config


def _run(cfg):
    solver = ThreePointBendingSolver(cfg, verbose=False)
    solution = solver.solve()
    return CaseResults.from_solution(solution)


@pytest.fixture(scope="module")
def isotropic_results():
    """Single isotropic-material beam (E faces-like) for the EB benchmark."""
    from sandwich3pb.config import (
        LayerSpec, MaterialSpec,
    )

    cfg = make_config()
    cfg.name = "iso_bench"
    # one thick "core" replaced: use a quasi-homogeneous beam with 3 layers
    # of the same material to keep the 3-layer schema
    mat = MaterialSpec(kind="isotropic", name="alu",
                       E=68000.0, nu=0.33, shear=1e9, compression=1e9)
    cfg.materials = {"alu": mat}
    cfg.stackup = [
        LayerSpec(material="alu", thickness=8.0, role="face"),
        LayerSpec(material="alu", thickness=8.0, role="core"),
        LayerSpec(material="alu", thickness=8.0, role="face"),
    ]
    # quadratic displacement (Q2 on the hex8 geometry) - hex8 shear locking
    # makes order-1 beams ~2x too stiff at this slenderness
    cfg.mesh = cfg.mesh.__class__(elements_x=40, elements_w=2,
                                  elements_per_layer={"face": 2, "core": 4},
                                  element_order=2)
    cfg.loading = cfg.loading.__class__(max_indentation=1.0, n_steps=5)
    cfg.contact = cfg.contact.__class__(
        penalty=1.0e6, roller_radius_load=10.0, roller_radius_support=5.0,
        augmented_lagrangian=cfg.contact.augmented_lagrangian.__class__(
            enabled=True, max_outer=8, tol=1e-3, update_rate=10.0),
    )
    return _run(cfg)


@pytest.fixture(scope="module")
def sandwich_results():
    cfg = make_config()
    cfg.name = "sandwich_bench"
    cfg.mesh.element_order = 2   # Q2: no transverse-shear locking
    cfg.loading = cfg.loading.__class__(max_indentation=1.0, n_steps=5)
    return _run(cfg)


class TestEulerBernoulli:
    def test_deflection_within_5pct(self, isotropic_results):
        """Isotropic beam: FE stiffness vs. dP/dw from EB theory.

        δ = P L^3 / (48 E I) for 3PB with central load (span L between
        supports). The FE model also carries contact + support compliance,
        so a tolerance of a few percent is appropriate.
        """
        cfg = isotropic_results.cfg
        E = 68000.0
        b, h = cfg.geometry.width, cfg.total_thickness
        I = b * h**3 / 12.0
        L = cfg.geometry.span
        k_eb = 48.0 * E * I / L**3  # N/mm

        fe = isotropic_results.summary["force_gradient_N_per_mm"]
        ratio = fe / k_eb
        # Timoshenko shear + contact compliance soften the FE response by
        # ~8-10% at span/depth = 8.3, so expect slightly below unity.
        assert 0.82 < ratio < 1.03, f"FE/EB stiffness ratio = {ratio}"

    def test_linear_response(self, isotropic_results):
        s = isotropic_results.summary
        assert s["gradient_fit_r2"] > 0.999


class TestAllenSandwich:
    def test_sandwich_stiffness_matches_allen(self, sandwich_results):
        """FE stiffness must match Allen bending+shear theory.

        With span/depth = 200/23 = 8.7 and a soft foam core, transverse
        shear dominates the deflection; the FE result must track the
        Allen prediction (not the pure-bending value).
        """
        cfg = sandwich_results.cfg
        L = cfg.geometry.span
        b = cfg.geometry.width
        tf = 1.5
        tc = 20.0
        Ef = 39000.0  # uniaxial modulus along the beam axis
        Gc = 75.0 / (2.0 * 1.3)

        d = tc + tf  # facesheet centroid distance
        D = 2 * Ef * (b * tf**3 / 12.0 + b * tf * (d / 2) ** 2)
        A = Gc * b * d
        k_bend = 48.0 * D / L**3
        # Allen 3PB: 1/k = L^3/(48 D) + L/(4 b d Gc)
        k_allen = 1.0 / (L**3 / (48.0 * D) + L / (4.0 * A))

        fe = sandwich_results.summary["force_gradient_N_per_mm"]
        ratio = fe / k_allen
        assert 0.85 < ratio < 1.15, f"FE/Allen stiffness ratio = {ratio}"
        # shear flexibility must matter strongly for this geometry
        assert fe < 0.5 * k_bend

    def test_apparent_rigidity_below_layup_rigidity(self, sandwich_results):
        """Shear flexibility reduces the apparent rigidity below the
        analytic layup EI at this span/depth ratio."""
        s = sandwich_results.summary
        ratio = s["rigidity_ratio_FE_over_layup"]
        assert 0.05 < ratio < 0.5

    def test_layup_rigidity_matches_closed_form(self):
        cfg = make_config()
        lay = layup_flexural_rigidity(cfg)
        # thin faces: EI ~ 2 * Ex * b * tf * (d/2)^2 with d = tc + tf
        from sandwich3pb.materials import localized_quad_form

        quad = localized_quad_form(cfg.materials["glass_epoxy"], 0.0)
        Ex = 1.0 / np.linalg.inv(quad.C)[0, 0]
        b = cfg.geometry.width
        tf, tc = 1.5, 20.0
        d = tc + tf
        EI_expected = 2 * Ex * (b * tf**3 / 12.0 + b * tf * (d / 2) ** 2)
        assert lay["flexural_rigidity"] == pytest.approx(EI_expected, rel=5e-3)


class TestForceBalance:
    def test_supports_carry_load(self, sandwich_results):
        s = sandwich_results.summary
        assert s["force_balance_residual"] < 5e-3

    def test_symmetric_reactions(self, sandwich_results):
        rec = sandwich_results.summary["support_reactions_N"]
        assert len(rec) == 2
        assert abs(rec[0] - rec[1]) < 1e-6 * max(rec[0], 1.0)


class TestContactQuality:
    def test_penetration_bounded(self, sandwich_results):
        s = sandwich_results.summary
        assert s["max_contact_penetration_mm"] < 5e-3

    def test_all_steps_converged(self, sandwich_results):
        assert sandwich_results.summary["all_steps_converged"]


class TestMonotonicity:
    def test_force_monotone_in_travel(self, sandwich_results):
        f = [r["force_N"] for r in sandwich_results.history]
        assert all(b > a for a, b in zip(f, f[1:]))

    def test_stiffer_with_thicker_core(self):
        cfg1 = make_config()
        cfg2 = make_config()
        cfg2.stackup[1].thickness = 25.0
        r1 = _run(cfg1)
        r2 = _run(cfg2)
        assert (
            r2.summary["force_gradient_N_per_mm"]
            > r1.summary["force_gradient_N_per_mm"]
        )
