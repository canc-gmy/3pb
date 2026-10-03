"""Exactness of the cached-contact tangent solve (no DOLFINx required).

The Newton tangent ``K + diag(d)`` is solved by reusing one factorization
of ``K + diag(d_all)`` and applying a Woodbury correction confined to the
contact DOFs. That is an optimization, so it must reproduce a direct
factorization of ``K + diag(d)`` exactly -- in particular for a *partly*
open contact set, and for the near-singular ``K`` the solver actually sees
(its rigid modes are held only by ~1e-9 ground springs).
"""

import numpy as np
import pytest
from scipy.sparse import diags, random as sparse_random
from scipy.sparse.linalg import splu

from sandwich3pb.solver import _ContactTangentSolver


def _operator(n=240, density=0.04, seed=0, near_singular=False):
    rng = np.random.default_rng(seed)
    a = sparse_random(n, n, density=density, format="csc", random_state=rng)
    k = (a + a.T).tocsc() + diags(np.full(n, 20.0))
    if near_singular:
        # mimic the ground springs that stabilize the rigid modes
        k = k + diags(np.full(n, 20.0 * 1e-9))
    return k, rng


def _contact_dofs(n, count, rng):
    return np.sort(rng.choice(n, size=count, replace=False).astype(np.intp))


@pytest.mark.parametrize("near_singular", [False, True])
@pytest.mark.parametrize("active_fraction", [1.0, 0.75, 0.5, 0.0])
def test_cached_solve_matches_direct(near_singular, active_fraction):
    """Every active-set configuration reproduces the direct solve."""
    k, rng = _operator(near_singular=near_singular)
    n = k.shape[0]
    contact = _contact_dofs(n, 24, rng)
    d_all = rng.uniform(1e4, 1e6, size=contact.size)
    tangent = _ContactTangentSolver(k, contact, d_all)

    d = np.zeros(n)
    n_active = int(round(contact.size * active_fraction))
    if n_active:
        d[contact[:n_active]] = d_all[:n_active]

    b = rng.standard_normal(n)
    expected = splu((k + diags(d)).tocsc()).solve(b)
    assert np.allclose(tangent.solve(b, d), expected, rtol=1e-8, atol=1e-10)


def test_cached_solve_matches_direct_off_contact_diagonal():
    """Contact stiffness outside the contact set must not be ignored.

    ``d_free`` is only ever nonzero on the contact DOFs, so this guards the
    contract that makes the cached path valid rather than an approximation.
    """
    k, rng = _operator(seed=3)
    n = k.shape[0]
    contact = _contact_dofs(n, 12, rng)
    d_all = rng.uniform(1e4, 1e6, size=contact.size)
    tangent = _ContactTangentSolver(k, contact, d_all)

    d = np.zeros(n)
    d[contact] = d_all
    b = rng.standard_normal(n)
    assert np.allclose(
        tangent.solve(b, d), splu((k + diags(d)).tocsc()).solve(b),
        rtol=1e-8, atol=1e-10,
    )


def test_direct_fallback_agrees_with_cached_branch():
    """The over-the-cap fallback is the same algebra, so results match."""
    k, rng = _operator(seed=5)
    n = k.shape[0]
    contact = _contact_dofs(n, 15, rng)
    d_all = rng.uniform(1e4, 1e6, size=contact.size)

    cached = _ContactTangentSolver(k, contact, d_all, max_cached_dofs=100)
    direct = _ContactTangentSolver(k, contact, d_all, max_cached_dofs=2)
    assert cached._use_cache and not direct._use_cache

    d = np.zeros(n)
    d[contact[:8]] = d_all[:8]
    b = rng.standard_normal(n)
    assert np.allclose(cached.solve(b, d), direct.solve(b, d),
                       rtol=1e-8, atol=1e-10)


def test_schur_block_is_symmetric():
    """``S = A^-1[C, C]`` is symmetric for symmetric positive-definite A."""
    k, rng = _operator(seed=7)
    n = k.shape[0]
    contact = _contact_dofs(n, 20, rng)
    d_all = rng.uniform(1e4, 1e6, size=contact.size)
    tangent = _ContactTangentSolver(k, contact, d_all)
    assert np.allclose(tangent._S, tangent._S.T, rtol=1e-10, atol=1e-12)


def test_single_factorization_reused_across_solves():
    """The whole point: one factorization, however many tangent solves."""
    k, rng = _operator(n=160, seed=9)
    n = k.shape[0]
    contact = _contact_dofs(n, 18, rng)
    d_all = rng.uniform(1e4, 1e6, size=contact.size)
    tangent = _ContactTangentSolver(k, contact, d_all)

    assert tangent.n_factorizations == 1
    d = np.zeros(n)
    d[contact] = d_all
    for _ in range(5):
        tangent.solve(rng.standard_normal(n), d)
    assert tangent.n_factorizations == 1
    assert tangent.n_solves == 5


def test_empty_contact_set_is_handled():
    """No contact DOFs at all still solves (and cannot be cached)."""
    k, rng = _operator(n=80, seed=11)
    tangent = _ContactTangentSolver(k, np.empty(0, np.intp),
                                    np.empty(0))
    assert not tangent._use_cache
    d = np.zeros(k.shape[0])
    b = rng.standard_normal(k.shape[0])
    assert np.allclose(tangent.solve(b, d), splu(k).solve(b),
                       rtol=1e-8, atol=1e-10)


def test_auto_penalty_scales_contact_area_out():
    """``penalty * tributary_area`` must track ``penalty_scale``.

    The penalty carries an inverse-area factor, so scaling it by the mean
    stiffness alone (without dividing by the mean contact area) would make
    the *nodal* contact stiffness - the only thing the structure feels -
    depend on mesh refinement and roller radius.
    """
    pytest.importorskip("dolfinx")
    from sandwich3pb.solver import ThreePointBendingSolver
    from tests.conftest import make_config

    nodal = {}
    for elements_w in (2, 8):
        cfg = make_config()
        cfg.mesh.elements_w = elements_w
        cfg.contact.penalty = -1.0
        cfg.contact.penalty_scale = 10.0
        solver = ThreePointBendingSolver(cfg, verbose=False)
        stiffness = solver.contact.penalty * float(
            solver.contact.pairs["support"][0].point_areas.mean()
        )
        bulk = float(abs(solver.K_ff.diagonal()).mean())
        nodal[elements_w] = stiffness / bulk

    # Contact is ~penalty_scale times the bulk stiffness in nodal terms,
    # and that ratio must not drift with the mesh refinement.
    for ratio in nodal.values():
        assert 5.0 < ratio < 20.0
    assert nodal[2] == pytest.approx(nodal[8], rel=0.05)