"""Tests for material tensors, rotations and failure criteria."""

import numpy as np
import pytest

from sandwich3pb.config import MaterialSpec
from sandwich3pb.materials import (
    isotropic_stiffness,
    localized_quad_form,
    rotation_matrix_y,
    stiffness_matrix,
    tensor4_to_voigt,
    tsai_wu_3d,
    von_mises,
    voigt_to_tensor4,
)


def test_isotropic_matrix_symmetry():
    C = isotropic_stiffness(1000.0, 0.3)
    assert np.allclose(C, C.T)
    assert C[0, 0] > 0 and np.all(np.linalg.eigvalsh(C) > 0)


def test_orthotropic_positive_definite():
    mat = MaterialSpec(
        kind="orthotropic", name="t", E1=39000, E2=9000, E3=9000,
        nu12=0.28, nu13=0.28, nu23=0.40,
        G12=3800, G13=3800, G23=3200,
    )
    C = stiffness_matrix(mat)
    assert np.allclose(C, C.T)
    assert np.all(np.linalg.eigvalsh(C) > 0)


def test_unphysical_poisson_rejected():
    mat = MaterialSpec(
        kind="orthotropic", name="t", E1=1000, E2=1000, E3=1000,
        nu12=0.9, nu13=0.9, nu23=0.9,
        G12=400, G13=400, G23=400,
    )
    with pytest.raises(ValueError, match="positive-definite"):
        stiffness_matrix(mat)


def test_uniaxial_modulus_roundtrip():
    """C1111^-1 in the uniaxial sense must reproduce E1 in material axes."""
    mat = MaterialSpec(
        kind="orthotropic", name="t", E1=39000, E2=9000, E3=9000,
        nu12=0.28, nu13=0.28, nu23=0.40,
        G12=3800, G13=3800, G23=3200,
    )
    C = stiffness_matrix(mat)
    S = np.linalg.inv(C)
    assert 1.0 / S[0, 0] == pytest.approx(39000.0, rel=1e-12)


def test_iso_rotation_invariance():
    """Rotating an isotropic material must not change its stiffness."""
    C0 = isotropic_stiffness(210000.0, 0.3)
    quad = localized_quad_form(
        MaterialSpec(kind="isotropic", name="steel", E=210000.0, nu=0.3), 37.0
    )
    assert np.allclose(quad.C, C0, atol=1e-8 * np.abs(C0).max())


def test_orthogonal_45_rotation_direction():
    """At ±45°, axial strain must couple to in-plane shear, and the axial
    stiffness must drop relative to the fibre direction (classical result
    for a unidirectional ply rotated off-axis)."""
    mat = MaterialSpec(
        kind="orthotropic", name="t", E1=39000, E2=9000, E3=9000,
        nu12=0.28, nu13=0.28, nu23=0.40,
        G12=3800, G13=3800, G23=3200,
    )
    quad0 = localized_quad_form(mat, 0.0)
    quad45 = localized_quad_form(mat, 45.0)
    quad_m45 = localized_quad_form(mat, -45.0)
    eps_g = np.array([1e-3, 0, 0, 0, 0, 0])
    s0 = quad0.C @ eps_g
    s45 = quad45.C @ eps_g
    s_m45 = quad_m45.C @ eps_g
    # rotation is about y: axial strain couples to xz-shear (Voigt idx 4)
    assert abs(s45[4]) > 1e-6 * abs(s0[0])
    # the coupling flips sign for the mirrored orientation
    assert s45[4] * s_m45[4] < 0.0
    # tau_xy (idx 5) stays zero: y is the rotation axis
    assert s45[5] == pytest.approx(0.0, abs=1e-9)
    assert quad45.C[0, 0] < quad0.C[0, 0]


def test_energy_consistency_rotation():
    """Strain energy must be identical in both frames for any rotation."""
    mat = MaterialSpec(
        kind="orthotropic", name="t", E1=39000, E2=9000, E3=9000,
        nu12=0.28, nu13=0.28, nu23=0.40,
        G12=3800, G13=3800, G23=3200,
    )
    quad = localized_quad_form(mat, 30.0)
    rng = np.random.default_rng(0)
    eps_g = rng.normal(size=6) * 1e-4
    # energy from global form
    w_g = quad.quad_form(eps_g)
    # energy computed in material axes
    eps_m = quad.strain_material_from_global(eps_g)
    w_m = 0.5 * float(eps_m @ quad.C_material @ eps_m)
    assert w_g == pytest.approx(w_m, rel=1e-10)


def test_tensor4_roundtrip():
    C = isotropic_stiffness(1000.0, 0.3)
    C4 = voigt_to_tensor4(C)
    C2 = tensor4_to_voigt(C4)
    assert np.allclose(C, C2)


def test_tsai_wu_hand_value():
    mat = MaterialSpec(
        kind="orthotropic", name="t",
        E1=1, E2=1, E3=1, nu12=0.0, nu13=0.0, nu23=0.0,
        G12=1, G13=1, G23=1,
        Xt=100.0, Xc=100.0, Yt=10.0, Yc=10.0, S=5.0,
    )
    # pure s11 = 50 -> F11*s^2 = 0.25
    assert tsai_wu_3d(np.array([50.0, 0, 0, 0, 0, 0]), mat) == pytest.approx(0.25)
    # pure s12 = 5 -> F66*s^2 = 1.0
    assert tsai_wu_3d(np.array([0, 0, 0, 0, 0, 5.0]), mat) == pytest.approx(1.0)


def test_von_mises_uniaxial():
    v = np.array([100.0, 0, 0, 0, 0, 0])
    assert von_mises(v) == pytest.approx(100.0)


def test_von_mises_pure_shear():
    v = np.array([0, 0, 0, 0, 0, 50.0])
    assert von_mises(v) == pytest.approx(np.sqrt(3.0) * 50.0)
