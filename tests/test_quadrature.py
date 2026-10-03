"""The pinned FFCx quadrature degree must be exact for this integrand.

The stiffness form is integrated with an explicitly pinned quadrature degree
(``solver._QUADRATURE_DEGREE``) because FFCx's own default -- the highest
degree present in the integrand -- over-integrates it by a factor ~2.7 in
point count for no accuracy gain. Pinning is only legitimate while the
integrand stays low-degree, which relies on two things that are easy to break
silently: the geometry map is trilinear hex8 within every cell, and the
constitutive matrix is constant within every cell. A curved geometry map or a
graded/smeared layer would raise the integrand degree and the pinned rule would
under-integrate *silently* -- no warning, just a wrong stiffness matrix.

These tests re-assemble the same form at a higher and a lower degree and check
both directions, so the pinned value has to be right rather than merely
effective. Deliberately coarse (elements_x = 12) and solve-free: they only
assemble, so they cost well under a second.
"""

import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")

from sandwich3pb.solver import (
    _QUADRATURE_DEGREE,
    ThreePointBendingSolver,
    _assemble_bilinear,
)
from tests.conftest import make_config

# Above every degree FFCx would pick for this form, so the reference is the
# converged value rather than another under-integrated one.
REFERENCE_DEGREE = 8


@pytest.fixture(scope="module")
def solver():
    cfg = make_config()
    cfg.name = "quadrature"
    cfg.mesh = cfg.mesh.__class__(
        elements_x=12, elements_w=2,
        elements_per_layer={"face": 1, "core": 2},
        element_order=2,  # Q2 on hex8: the production discretization
    )
    return ThreePointBendingSolver(cfg, verbose=False)


def _assemble_at_degree(solver, degree):
    """Re-assemble the stiffness form at an explicit quadrature degree."""
    import dolfinx.fem as fem

    form = fem.form(solver._stiffness_form(
        solver.mesh, solver.cell_tags, degree))
    return _assemble_bilinear(form).tocsr()


def _relative_difference(a, b) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_pinned_degree_matches_high_degree_reference(solver):
    """K at the pinned degree equals K at a converged degree to roundoff."""
    reference = _assemble_at_degree(solver, REFERENCE_DEGREE)
    assert solver.K.shape == reference.shape
    assert solver.K.nnz == reference.nnz
    rel = _relative_difference(solver.K, reference)
    assert rel < 1.0e-12, (
        f"quadrature degree {_QUADRATURE_DEGREE} is not exact for the "
        f"stiffness integrand (max|K - K_ref| / max|K_ref| = {rel:.3e}); "
        "the geometry map or the material assignment must have gained "
        "in-cell variation -- raise _QUADRATURE_DEGREE"
    )


def test_pinned_degree_is_the_cheapest_exact_one(solver):
    """One degree lower is already wrong, so the pin is in effect and minimal.

    This also guards the metadata itself: if ``Measure(..., metadata=...)``
    ever stopped reaching FFCx, the form would silently fall back to the
    default degree and this comparison would come back far too small.
    """
    coarser = _assemble_at_degree(solver, _QUADRATURE_DEGREE - 1)
    rel = _relative_difference(coarser, solver.K)
    assert rel > 1.0e-6, (
        f"degree {_QUADRATURE_DEGREE - 1} is as accurate as "
        f"_QUADRATURE_DEGREE (relative difference {rel:.3e}); either the "
        "pinned degree is not reaching FFCx or it can be lowered"
    )